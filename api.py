"""
SAO Demo AI Assistant REST API
"""

import os
import uuid
import asyncio
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from setup_env import setup_environment
from domain_manager import DomainManager

from helpers.sao_api_helpers import (
    get_sao_project_id,
    get_sao_agent_stream_id,
    get_sao_app_url,
    create_sao_logger,
)
from helpers.agent_control_helpers import init_agent_control

from agent_factory import AgentFactory


# --------------------------------------------------
# Environment setup (same as Streamlit app)
# --------------------------------------------------

load_dotenv()

DOMAIN = "bank"

if not os.getenv("_SAO_ENV_LOADED"):
    dm = DomainManager()
    full_config = dm.load_domain_config(DOMAIN)

    setup_environment(
        domain_name=DOMAIN,
        domain_config=full_config.config
    )

    os.environ["_SAO_ENV_LOADED"] = "true"


# --------------------------------------------------
# Splunk AO project / agent stream information
# --------------------------------------------------

project_name = os.environ.get("SPLUNK_AO_PROJECT", "")
agent_stream_name = os.environ.get("SPLUNK_AO_AGENT_STREAM", "")

print("SPLUNK_AO_PROJECT:", project_name)
print("SPLUNK_AO_AGENT_STREAM:", agent_stream_name)

console_url = get_sao_app_url()

project_id = get_sao_project_id(project_name)

agent_stream_id = get_sao_agent_stream_id(
    project_id,
    agent_stream_name
)

print("SPLUNK AO PROJECT ID:", project_id)
print("SPLUNK AO AGENT STREAM ID:", agent_stream_id)


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Bank AI Assistant API"
)


class EndOtelTraceMiddleware(BaseHTTPMiddleware):
    """End the HTTP span and flush when the response is ready (one trace per request)."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        span = trace.get_current_span()
        if span and span.is_recording():
            attrs = getattr(request.state, "otel_sao_attrs", None)
            if attrs:
                for key, value in attrs.items():
                    if value is not None:
                        span.set_attribute(key, value)
            if response.status_code >= 400:
                span.set_status(Status(StatusCode.ERROR))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=5000)

        return response


app.add_middleware(EndOtelTraceMiddleware)


FRAMEWORK = "LangGraph"
PROVIDER = "bedrock"
MODEL = "mistral.ministral-3-14b-instruct"


class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


@dataclass
class ChatSession:
    session_id: str
    logger: object
    agent: object
    sao_session_id: Optional[str] = None
    messages: List[dict] = field(default_factory=list)


factory = AgentFactory()
sessions: Dict[str, ChatSession] = {}
# browser session_id -> SAO internal session UUID (survives logger recreation)
_sao_session_ids: Dict[str, str] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(session_id: str) -> ChatSession:
    with _sessions_lock:
        if session_id in sessions:
            print(f"[api] session cache hit session_id={session_id}")
            return sessions[session_id]

        print(f"[api] creating session cache session_id={session_id}")
        logger = create_sao_logger(
            project_name=project_name,
            agent_stream=agent_stream_name,
        )
        logger.enable_agent_control()
        init_agent_control(
            logger,
            project_name=project_name,
            agent_stream=agent_stream_name,
            agent_description=f"{DOMAIN.title()} demo agent",
        )

        agent = factory.create_agent(
            domain=DOMAIN,
            framework=FRAMEWORK,
            session_id=session_id,
            model_name=MODEL,
            sao_logger=logger,
            llm_provider=PROVIDER,
        )

        chat_session = ChatSession(
            session_id=session_id,
            logger=logger,
            agent=agent,
        )
        sessions[session_id] = chat_session
        return chat_session


def ensure_sao_session(chat_session: ChatSession) -> None:
    """
    Attach the logger to the Splunk AO session for this browser visit.

    Splunk AO creates one auto-generated session per trace unless a session is
    active on the logger. Re-attach before every /query (after flush clears
    traces) using the cached SAO session id or external_id lookup.
    """
    external_id = chat_session.session_id
    session_name = f"{DOMAIN.title()} Agent Demo"
    logger = chat_session.logger

    cached_sao_id = _sao_session_ids.get(external_id)
    if cached_sao_id:
        logger.set_session(cached_sao_id)
        chat_session.sao_session_id = cached_sao_id
        print(
            f"[api] sao set_session sao_id={cached_sao_id} "
            f"external_id={external_id}"
        )
        return

    sao_id = logger.start_session(
        name=session_name,
        external_id=external_id,
    )
    if not sao_id:
        sao_id = getattr(logger, "session_id", None)

    if sao_id:
        _sao_session_ids[external_id] = sao_id
        chat_session.sao_session_id = sao_id
        print(
            f"[api] sao start_session sao_id={sao_id} "
            f"external_id={external_id}"
        )
    else:
        print(
            f"[api] WARNING: no SAO session for external_id={external_id}; "
            "traces will each get their own auto-generated session"
        )


# --------------------------------------------------
# API endpoint
# --------------------------------------------------

@app.post("/query")
async def query(request_body: QueryRequest, request: Request):
    session_id = (request_body.session_id or "").strip() or str(uuid.uuid4())[:10]
    print(f"[api] /query received session_id={session_id!r} question={request_body.question!r}")

    chat_session = get_or_create_session(session_id)
    ensure_sao_session(chat_session)

    chat_session.messages.append(
        {"role": "user", "content": request_body.question}
    )

    response = await asyncio.to_thread(
        chat_session.agent.process_query,
        chat_session.messages,
    )

    chat_session.messages.append(
        {"role": "assistant", "content": response}
    )

    sao_id = chat_session.sao_session_id or getattr(
        chat_session.logger, "session_id", None
    )
    request.state.otel_sao_attrs = {
        "splunk_ao.project_id": project_id,
        "splunk_ao.agent_stream_id": agent_stream_id,
        "splunk_ao.session_id": session_id,
        "splunk_ao_user_session_id": sao_id,
        "splunk_ao.url": (
            f"{console_url}/project/{project_id}/agent-streams/{agent_stream_id}"
        ),
    }

    return {
        "answer": response,
        "session_id": session_id,
        "splunk_ao": {
            "project_id": project_id,
            "agent_stream_id": agent_stream_id,
            "url": (
                f"{console_url}/project/"
                f"{project_id}/agent-streams/{agent_stream_id}"
            )
        }
    }
