"""
Splunk Agent Observability helper functions.

These helpers load only API credentials and console URL from secrets.
Domain-specific environment setup (like project names) is handled by the main app.
"""
import logging
import os
import toml
from pathlib import Path


_SECRET_PLACEHOLDERS = {
    "",
    "...",
    "https://console.example.com",
    "https://api.example.com",
}


def _secret_value(secrets: dict, key: str) -> str:
    value = str(secrets.get(key, "") or "").strip()
    if value in _SECRET_PLACEHOLDERS:
        return ""
    return value


def _load_secrets_if_needed():
    """
    Load secrets from .streamlit/secrets.toml if environment variables are not set.
    Only loads API credentials and console URL (not domain-specific settings).
    """
    on_prem_ready = os.environ.get("SPLUNK_AO_API_KEY") and os.environ.get("SPLUNK_AO_CONSOLE_URL")
    saas_ready = os.environ.get("SPLUNK_AO_O11Y_TOKEN") and os.environ.get("SPLUNK_AO_REALM")
    if on_prem_ready or saas_ready:
        return

    secrets_path = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return

    try:
        secrets = toml.load(secrets_path)

        mapping = {
            "splunk_ao_api_key": "SPLUNK_AO_API_KEY",
            "splunk_ao_console_url": "SPLUNK_AO_CONSOLE_URL",
            "splunk_ao_api_url": "SPLUNK_AO_API_URL",
            "splunk_ao_o11y_token": "SPLUNK_AO_O11Y_TOKEN",
            "splunk_ao_o11y_api_token": "SPLUNK_AO_O11Y_API_TOKEN",
            "splunk_ao_realm": "SPLUNK_AO_REALM",
        }
        for secret_key, env_key in mapping.items():
            value = _secret_value(secrets, secret_key)
            if value and not os.environ.get(env_key):
                os.environ[env_key] = value

    except Exception as e:
        logging.warning(f"Could not load secrets: {e}")


def get_sao_app_url() -> str:
    """
    Get the Splunk Agent Observability web console URL.

    Returns:
        str: The console URL without trailing slash

    Raises:
        ValueError: If neither SPLUNK_AO_CONSOLE_URL nor SPLUNK_AO_REALM is set
    """
    _load_secrets_if_needed()

    sao_url = os.environ.get("SPLUNK_AO_CONSOLE_URL")
    if sao_url:
        return sao_url.rstrip("/")

    realm = os.environ.get("SPLUNK_AO_REALM")
    if realm:
        return f"https://app.{realm}.observability.splunkcloud.com"

    raise ValueError("SPLUNK_AO_CONSOLE_URL or SPLUNK_AO_REALM environment variable is not set")


def get_sao_api_url() -> str:
    """
    Get the Splunk Agent Observability API URL.

    Returns:
        str: The API URL
    """
    explicit = os.environ.get("SPLUNK_AO_API_URL")
    if explicit:
        return explicit.rstrip("/")

    sao_url = get_sao_app_url()

    if sao_url.startswith("https://"):
        domain = sao_url[8:]
    elif sao_url.startswith("http://"):
        domain = sao_url[7:]
    else:
        domain = sao_url

    if domain.startswith("app."):
        domain = domain[4:]
    elif domain.startswith("console."):
        domain = domain[8:]
    domain = domain.split("/")[0]

    return f"https://api.{domain}"


def get_sao_api_key() -> str:
    """
    Get the Splunk Agent Observability API key or SaaS token.

    Returns:
        str: The API key / token

    Raises:
        ValueError: If no credential is set
    """
    _load_secrets_if_needed()

    api_key = (
        os.environ.get("SPLUNK_AO_API_KEY")
        or os.environ.get("SPLUNK_AO_O11Y_API_TOKEN")
        or os.environ.get("SPLUNK_AO_O11Y_TOKEN")
    )
    if not api_key:
        raise ValueError("SPLUNK_AO_API_KEY (or SPLUNK_AO_O11Y_TOKEN) environment variable is not set")
    return api_key


def resolve_logger_ids(logger) -> None:
    """Fill logger.project_id / agent_stream_id without starting a session.

    In SaaS (O11y) mode the SDK defers the name -> UUID lookup until the first
    session call. Agent Control needs logger.agent_stream_id up front, and
    start_session() would create a separate (empty) session record, so resolve
    the IDs directly. Names are looked up (or created) via the SDK's own helper.
    """
    if getattr(logger, "project_id", None) and getattr(logger, "agent_stream_id", None):
        return
    try:
        logger._ensure_session_crud_client()
    except Exception as exc:
        print(f"⚠️ Could not resolve Splunk AO project/agent stream IDs: {exc}")


def create_sao_logger(project_name: str, agent_stream: str, resolve_ids: bool = True):
    """
    Create a SplunkAOLogger for an existing project/agent stream.
    """
    from splunk_ao import SplunkAOLogger

    logger = SplunkAOLogger(
        project=project_name,
        agent_stream=agent_stream,
        mode="batch",
    )
    if resolve_ids:
        resolve_logger_ids(logger)
    return logger


def get_sao_project_id(project_name: str) -> str:
    """
    Resolve the Splunk AO project ID for a given project name via the SDK logger.
    """
    agent_stream = os.environ.get("SPLUNK_AO_AGENT_STREAM", "default")
    logger = create_sao_logger(project_name, agent_stream)
    return getattr(logger, "project_id", None)


def get_sao_agent_stream_id(project_id: str, agent_stream_name: str) -> str:
    """
    Resolve the Splunk AO agent stream ID for a given project and stream name.
    """
    project_name = os.environ.get("SPLUNK_AO_PROJECT", "")
    logger = create_sao_logger(project_name, agent_stream_name)
    return getattr(logger, "agent_stream_id", None) or getattr(logger, "log_stream_id", None)
