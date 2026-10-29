"""
SAO Demo AI Assistant REST API

Banking UI posts to POST /query. Start without the OTel auto-instrumentor:

    python start_api.py
    # or:
    uvicorn api:app --host 0.0.0.0 --port 8000

Session/project ids are stamped on the Splunk AO "Run Agent" trace (same
fields Galileo put on the HTTP span).
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
from helpers.agent_control_helpers import init_agent_control, string_metadata

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


def _correlation_metadata(session_id: str, logger=None, sao_id: Optional[str] = None) -> dict:
    """IDs for start_session / start_trace. Prefer live logger values over import-time lookup."""
    resolved_project_id = (
        getattr(logger, "project_id", None) if logger is not None else None
    ) or project_id
    resolved_stream_id = (
        (
            getattr(logger, "agent_stream_id", None)
            or getattr(logger, "log_stream_id", None)
        )
        if logger is not None
        else None
    ) or agent_stream_id
    return {
        "splunk_ao.project_id": resolved_project_id,
        "splunk_ao.agent_stream_id": resolved_stream_id,
        "splunk_ao.session_id": session_id,
        "splunk_ao_user_session_id": sao_id,
        "splunk_ao.url": (
            f"{console_url}/project/{resolved_project_id}/agent-streams/{resolved_stream_id}"
        ),
    }


# --------------------------------------------------
# FastAPI
# --------------------------------------------------

app = FastAPI(
    title="Bank AI Assistant API"
)


class EndOtelTraceMiddleware(BaseHTTPMiddleware):
    """Stamp SAO ids on the FastAPI HTTP span. Do not end or flush it.

    Galileo agent traces were a separate upload, so ending/flushing the OTel
    span here was safe. Splunk AO traces are OTel spans; ending or force_flush
    here would cut the agent tree or parent it under POST /query.
    """

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


# --------------------------------------------------
# API endpoint
# --------------------------------------------------

@app.post("/query")
async def query(request_body: QueryRequest, request: Request):
    session_id = (request_body.session_id or "").strip() or str(uuid.uuid4())[:10]
    print(f"[api] /query received session_id={session_id!r} question={request_body.question!r}")

    chat_session = get_or_create_session(session_id)

    trace_metadata = string_metadata(
        _correlation_metadata(
            session_id,
            chat_session.logger,
            sao_id=chat_session.sao_session_id,
        )
    )
    request.state.otel_sao_attrs = trace_metadata

    chat_session.messages.append(
        {"role": "user", "content": request_body.question}
    )

    response = await asyncio.to_thread(
        chat_session.agent.process_query,
        chat_session.messages,
        metadata=trace_metadata,
    )

    chat_session.sao_session_id = getattr(
        chat_session.logger, "session_id", None
    ) or chat_session.sao_session_id
    chat_session.messages.append(
        {"role": "assistant", "content": response}
    )

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
