"""
Environment Setup - Load secrets and set environment variables
"""
import os
import warnings
import toml
from pathlib import Path
from typing import Optional

# Silence SDK span-serialization noise. The ingestion models
# subclass the core span types and hold child spans in a discriminated union,
# which makes Pydantic 2.12 emit "PydanticSerializationUnexpectedValue" /
# "Pydantic serializer warnings" (field_name='spans') on every trace upload.
# It's cosmetic — traces still serialize and upload correctly, and it's provider
# independent (not related to Ollama/OpenAI/Bedrock). Registered here so it
# applies process-wide. Remove this filter if it ever masks a real
# serialization problem.
warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings",
    category=UserWarning,
    module="pydantic",
)


def _derive_sao_api_url(console_url: str, explicit_url: str = "") -> str:
    """Derive Splunk AO API URL from console URL when not set explicitly."""
    if explicit_url:
        return explicit_url.rstrip("/")
    console_url = (console_url or "").rstrip("/")
    if not console_url:
        return ""
    if "console." in console_url:
        return console_url.replace("console.", "api.", 1)
    return ""


def _derive_agent_control_url(console_url: str, explicit_url: str = "") -> str:
    """Derive hosted Agent Control proxy URL from the Splunk AO console URL."""
    if explicit_url:
        return explicit_url.rstrip("/")
    console_url = (console_url or "").rstrip("/")
    if not console_url:
        return ""
    return f"{console_url}/api/agent-control"


def get_domain_project_name(domain_name: str, domain_config: Optional[dict] = None) -> str:
    """
    Get the Splunk Agent Observability project name for a domain.

    Priority:
    1. If domain_config has a 'splunk_ao.project' field, use that (explicit)
    2. Otherwise, use default: "sao-demo-{domain_name}"
    """
    if domain_config and "splunk_ao" in domain_config and "project" in domain_config["splunk_ao"]:
        return domain_config["splunk_ao"]["project"]

    return f"sao-demo-{domain_name}"


def setup_environment(domain_name: Optional[str] = None, domain_config: Optional[dict] = None):
    """
    Load secrets from .streamlit/secrets.toml and set environment variables.

    If domain_name is provided:
        - Sets domain-specific SPLUNK_AO_PROJECT (from config or default)
        - Sets domain-specific SPLUNK_AO_AGENT_STREAM (from config or default)

    If domain_name is None (for CLI scripts):
        - Only sets API keys and non-domain-specific settings
        - Project name should be set by the calling script after loading domain config

    Do not mix on-prem (SPLUNK_AO_API_KEY + SPLUNK_AO_CONSOLE_URL) and SaaS
    (SPLUNK_AO_O11Y_TOKEN + SPLUNK_AO_REALM) environment variables.
    """
    secrets_path = Path(".streamlit/secrets.toml")

    if not secrets_path.exists():
        print("⚠️  .streamlit/secrets.toml not found. Please create it with your API keys.")

        secrets_path = Path("../.streamlit/secrets.toml")
        if not secrets_path.exists():
            return

    try:
        secrets = toml.load(secrets_path)

        on_prem_key = secrets.get("splunk_ao_api_key", "")
        console_url = secrets.get("splunk_ao_console_url", "")
        sao_api_url = _derive_sao_api_url(console_url, secrets.get("splunk_ao_api_url", ""))
        agent_control_url = _derive_agent_control_url(
            console_url, secrets.get("agent_control_url", "")
        )
        o11y_token = secrets.get("splunk_ao_o11y_token", "")
        o11y_api_token = secrets.get("splunk_ao_o11y_api_token", "")
        realm = secrets.get("splunk_ao_realm", "")

        use_on_prem = bool(on_prem_key and console_url)
        agent_control_api_key = on_prem_key if use_on_prem else (o11y_api_token or o11y_token)

        env_vars = {
            "OLLAMA_BASE_URL": secrets.get("ollama_base_url", ""),
            "OLLAMA_DEFAULT_CHAT_MODEL": secrets.get(
                "ollama_default_chat_model", "gemma4"
            ),
            "OLLAMA_EMBEDDING_MODEL": secrets.get(
                "ollama_embedding_model", "nomic-embed-text"
            ),
            "OPENAI_API_KEY": secrets.get("openai_api_key", ""),
            "OPENAI_DEFAULT_CHAT_MODEL": secrets.get(
                "openai_default_chat_model", "gpt-4o"
            ),
            "OPENAI_EMBEDDING_MODEL": secrets.get(
                "openai_embedding_model", "text-embedding-3-large"
            ),
            "OPENAI_EMBEDDING_DIMENSIONS": str(
                secrets.get("openai_embedding_dimensions", 768)
            ),
            "AWS_BEARER_TOKEN_BEDROCK": secrets.get("bedrock_api_key", ""),
            "AWS_REGION": secrets.get("aws_region", "us-east-1"),
            "BEDROCK_DEFAULT_CHAT_MODEL": secrets.get(
                "bedrock_default_chat_model", "mistral.ministral-3-14b-instruct"
            ),
            "BEDROCK_EMBEDDING_MODEL": secrets.get(
                "bedrock_embedding_model", "amazon.titan-embed-text-v2:0"
            ),
            "AGENT_CONTROL_URL": agent_control_url,
            "AGENT_CONTROL_API_KEY": agent_control_api_key,
            "AGENT_CONTROL_AGENT_NAME": secrets.get("agent_control_agent_name", ""),
            "AGENT_CONTROL_API_KEY_HEADER": secrets.get("agent_control_api_key_header", "Splunk-AO-API-Key"),
            "AGENT_CONTROL_RUNTIME_AUTH_MODE": secrets.get("agent_control_runtime_auth_mode", "jwt"),
            "AGENT_CONTROL_TARGET_TYPE": secrets.get("agent_control_target_type", "agent_stream"),
            "ADMIN_KEY": secrets.get("admin_key", ""),
            "POSTGRES_HOST": os.environ.get("POSTGRES_HOST")
            or secrets.get("postgres_host", "localhost"),
            "POSTGRES_PORT": os.environ.get("POSTGRES_PORT")
            or secrets.get("postgres_port", "5432"),
            "POSTGRES_USER": os.environ.get("POSTGRES_USER")
            or secrets.get("postgres_user", "postgres"),
            "POSTGRES_PASSWORD": os.environ.get("POSTGRES_PASSWORD")
            or secrets.get("postgres_password", ""),
            "POSTGRES_DB": os.environ.get("POSTGRES_DB")
            or secrets.get("postgres_db", "vectordb"),
            "ENVIRONMENT": secrets.get("environment", "local")
        }

        if use_on_prem:
            env_vars["SPLUNK_AO_API_KEY"] = on_prem_key
            env_vars["SPLUNK_AO_CONSOLE_URL"] = console_url
            env_vars["SPLUNK_AO_API_URL"] = sao_api_url
        else:
            env_vars["SPLUNK_AO_O11Y_TOKEN"] = o11y_token
            env_vars["SPLUNK_AO_REALM"] = realm
            env_vars["SPLUNK_AO_O11Y_API_TOKEN"] = o11y_api_token

        if domain_name:
            project_name = get_domain_project_name(domain_name, domain_config)
            agent_stream = "default"
            if domain_config and "splunk_ao" in domain_config and "agent_stream" in domain_config["splunk_ao"]:
                agent_stream = domain_config["splunk_ao"]["agent_stream"]

            env_vars["SPLUNK_AO_PROJECT"] = project_name
            env_vars["SPLUNK_AO_AGENT_STREAM"] = agent_stream

        _AUTHORITATIVE_KEYS = {
            "OLLAMA_BASE_URL",
            "OPENAI_API_KEY",
            "AWS_BEARER_TOKEN_BEDROCK",
        }
        for key, value in env_vars.items():
            if value:
                os.environ[key] = value
            elif key in _AUTHORITATIVE_KEYS:
                os.environ.pop(key, None)
                print(f"ℹ️  {key} not configured in secrets.toml (cleared)")
            else:
                print(f"⚠️  {key} not set (empty value)")

        embedding_provider = secrets.get("embedding_provider", "")
        if embedding_provider:
            os.environ["EMBEDDING_PROVIDER"] = embedding_provider

        if domain_name:
            project_name = get_domain_project_name(domain_name, domain_config)
            print(f"🔧 Environment setup complete for domain: {domain_name} (project: {project_name})")
        else:
            print("🔧 Environment setup complete (domain-agnostic mode)")

    except Exception as e:
        print(f"❌ Error loading secrets: {e}")


if __name__ == "__main__":
    setup_environment()
