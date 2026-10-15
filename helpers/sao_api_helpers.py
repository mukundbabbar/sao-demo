"""
Splunk Agent Observability helper functions.

These helpers load only API credentials and console URL from secrets.
Domain-specific environment setup (like project names) is handled by the main app.
"""
import logging
import os
import toml
from pathlib import Path


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

        if "splunk_ao_api_key" in secrets and not os.environ.get("SPLUNK_AO_API_KEY"):
            os.environ["SPLUNK_AO_API_KEY"] = secrets["splunk_ao_api_key"]

        if "splunk_ao_console_url" in secrets and not os.environ.get("SPLUNK_AO_CONSOLE_URL"):
            os.environ["SPLUNK_AO_CONSOLE_URL"] = secrets["splunk_ao_console_url"]

        if "splunk_ao_api_url" in secrets and not os.environ.get("SPLUNK_AO_API_URL"):
            os.environ["SPLUNK_AO_API_URL"] = secrets["splunk_ao_api_url"]

        if "splunk_ao_o11y_token" in secrets and not os.environ.get("SPLUNK_AO_O11Y_TOKEN"):
            os.environ["SPLUNK_AO_O11Y_TOKEN"] = secrets["splunk_ao_o11y_token"]

        if "splunk_ao_o11y_api_token" in secrets and not os.environ.get("SPLUNK_AO_O11Y_API_TOKEN"):
            os.environ["SPLUNK_AO_O11Y_API_TOKEN"] = secrets["splunk_ao_o11y_api_token"]

        if "splunk_ao_realm" in secrets and not os.environ.get("SPLUNK_AO_REALM"):
            os.environ["SPLUNK_AO_REALM"] = secrets["splunk_ao_realm"]

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
        return f"https://app.{realm}.signalfx.com"

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


def create_sao_logger(project_name: str, agent_stream: str):
    """
    Create a SplunkAOLogger for an existing project/agent stream.
    """
    from splunk_ao import SplunkAOLogger

    return SplunkAOLogger(project=project_name, agent_stream=agent_stream)


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
