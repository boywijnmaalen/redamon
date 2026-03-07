"""
Agent Project Settings - Fetch agent configuration from webapp API

When PROJECT_ID and WEBAPP_API_URL are set as environment variables,
settings are fetched from the PostgreSQL database via webapp API.
Otherwise, falls back to DEFAULT_AGENT_SETTINGS for standalone usage.

Mirrors the pattern from recon/project_settings.py.
"""
import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# =============================================================================
# DEFAULT SETTINGS - Used as fallback for standalone usage and missing API fields
# =============================================================================

DEFAULT_AGENT_SETTINGS: dict[str, Any] = {
    # LLM Configuration
    'OPENAI_MODEL': 'claude-opus-4-6',
    'INFORMATIONAL_SYSTEM_PROMPT': '',
    'EXPL_SYSTEM_PROMPT': '',
    'POST_EXPL_SYSTEM_PROMPT': '',

    # Stealth Mode
    'STEALTH_MODE': False,

    # Phase Configuration
    'ACTIVATE_POST_EXPL_PHASE': True,
    'POST_EXPL_PHASE_TYPE': 'statefull',

    # Payload Direction
    'LHOST': '',       # Empty string = not set
    'LPORT': None,      # None = not set
    'BIND_PORT_ON_TARGET': None,  # None = not set (agent will ask user)
    'PAYLOAD_USE_HTTPS': False,
    'NGROK_TUNNEL_ENABLED': False,
    'CHISEL_TUNNEL_ENABLED': False,

    # Agent Limits
    'MAX_ITERATIONS': 100,
    'EXECUTION_TRACE_MEMORY_STEPS': 100,
    'TOOL_OUTPUT_MAX_CHARS': 20000,

    # Approval Gates
    'REQUIRE_APPROVAL_FOR_EXPLOITATION': True,
    'REQUIRE_APPROVAL_FOR_POST_EXPLOITATION': True,

    # Neo4j
    'CYPHER_MAX_RETRIES': 3,

    # LLM Parse Retry
    'LLM_PARSE_MAX_RETRIES': 3,

    # Debug
    'CREATE_GRAPH_IMAGE_ON_INIT': False,

    # Logging
    'LOG_MAX_MB': 10,
    'LOG_BACKUP_COUNT': 5,

    # Tool Phase Restrictions
    'TOOL_PHASE_MAP': {
        'query_graph': ['informational', 'exploitation', 'post_exploitation'],
        'execute_curl': ['informational', 'exploitation', 'post_exploitation'],
        'execute_naabu': ['informational', 'exploitation', 'post_exploitation'],
        'execute_nmap': ['informational', 'exploitation', 'post_exploitation'],
        'execute_nuclei': ['informational', 'exploitation', 'post_exploitation'],
        'kali_shell': ['informational', 'exploitation', 'post_exploitation'],
        'execute_code': ['informational', 'exploitation', 'post_exploitation'],
        'execute_hydra': ['exploitation', 'post_exploitation'],
        'metasploit_console': ['exploitation', 'post_exploitation'],
        'msf_restart': ['exploitation', 'post_exploitation'],
        'web_search': ['informational', 'exploitation', 'post_exploitation'],
    },

    # Hydra Brute Force
    'HYDRA_ENABLED': True,
    'HYDRA_THREADS': 16,
    'HYDRA_WAIT_BETWEEN_CONNECTIONS': 0,
    'HYDRA_CONNECTION_TIMEOUT': 32,
    'HYDRA_STOP_ON_FIRST_FOUND': True,
    'HYDRA_EXTRA_CHECKS': 'nsr',
    'HYDRA_VERBOSE': True,
    'HYDRA_MAX_WORDLIST_ATTEMPTS': 3,

    # Phishing / Social Engineering
    'PHISHING_SMTP_CONFIG': '',  # Free-text SMTP config for phishing email delivery (optional)

    # Legacy (deprecated — kept for backward compat)
    'BRUTE_FORCE_MAX_WORDLIST_ATTEMPTS': 3,
    'BRUTEFORCE_SPEED': 5,
}


def fetch_agent_settings(project_id: str, webapp_url: str) -> dict[str, Any]:
    """
    Fetch agent settings from webapp API.

    Args:
        project_id: The project ID to fetch settings for
        webapp_url: Base URL of the webapp API (e.g., http://localhost:3000)

    Returns:
        Dictionary of settings in SCREAMING_SNAKE_CASE format
    """
    import requests

    url = f"{webapp_url.rstrip('/')}/api/projects/{project_id}"
    logger.info(f"Fetching agent settings from {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    project = response.json()

    # Start with defaults, then override with API values
    settings = DEFAULT_AGENT_SETTINGS.copy()

    # Map camelCase API fields to SCREAMING_SNAKE_CASE
    settings['OPENAI_MODEL'] = project.get('agentOpenaiModel', DEFAULT_AGENT_SETTINGS['OPENAI_MODEL'])
    settings['INFORMATIONAL_SYSTEM_PROMPT'] = project.get('agentInformationalSystemPrompt', DEFAULT_AGENT_SETTINGS['INFORMATIONAL_SYSTEM_PROMPT'])
    settings['EXPL_SYSTEM_PROMPT'] = project.get('agentExplSystemPrompt', DEFAULT_AGENT_SETTINGS['EXPL_SYSTEM_PROMPT'])
    settings['POST_EXPL_SYSTEM_PROMPT'] = project.get('agentPostExplSystemPrompt', DEFAULT_AGENT_SETTINGS['POST_EXPL_SYSTEM_PROMPT'])
    settings['ACTIVATE_POST_EXPL_PHASE'] = project.get('agentActivatePostExplPhase', DEFAULT_AGENT_SETTINGS['ACTIVATE_POST_EXPL_PHASE'])
    settings['POST_EXPL_PHASE_TYPE'] = project.get('agentPostExplPhaseType', DEFAULT_AGENT_SETTINGS['POST_EXPL_PHASE_TYPE'])
    settings['LHOST'] = project.get('agentLhost', DEFAULT_AGENT_SETTINGS['LHOST'])
    settings['LPORT'] = project.get('agentLport', DEFAULT_AGENT_SETTINGS['LPORT'])
    settings['BIND_PORT_ON_TARGET'] = project.get('agentBindPortOnTarget', DEFAULT_AGENT_SETTINGS['BIND_PORT_ON_TARGET'])
    settings['PAYLOAD_USE_HTTPS'] = project.get('agentPayloadUseHttps', DEFAULT_AGENT_SETTINGS['PAYLOAD_USE_HTTPS'])
    settings['NGROK_TUNNEL_ENABLED'] = project.get('agentNgrokTunnelEnabled', DEFAULT_AGENT_SETTINGS['NGROK_TUNNEL_ENABLED'])
    settings['CHISEL_TUNNEL_ENABLED'] = project.get('agentChiselTunnelEnabled', DEFAULT_AGENT_SETTINGS['CHISEL_TUNNEL_ENABLED'])
    settings['MAX_ITERATIONS'] = project.get('agentMaxIterations', DEFAULT_AGENT_SETTINGS['MAX_ITERATIONS'])
    settings['EXECUTION_TRACE_MEMORY_STEPS'] = project.get('agentExecutionTraceMemorySteps', DEFAULT_AGENT_SETTINGS['EXECUTION_TRACE_MEMORY_STEPS'])
    settings['REQUIRE_APPROVAL_FOR_EXPLOITATION'] = project.get('agentRequireApprovalForExploitation', DEFAULT_AGENT_SETTINGS['REQUIRE_APPROVAL_FOR_EXPLOITATION'])
    settings['REQUIRE_APPROVAL_FOR_POST_EXPLOITATION'] = project.get('agentRequireApprovalForPostExploitation', DEFAULT_AGENT_SETTINGS['REQUIRE_APPROVAL_FOR_POST_EXPLOITATION'])
    settings['TOOL_OUTPUT_MAX_CHARS'] = project.get('agentToolOutputMaxChars', DEFAULT_AGENT_SETTINGS['TOOL_OUTPUT_MAX_CHARS'])
    settings['CYPHER_MAX_RETRIES'] = project.get('agentCypherMaxRetries', DEFAULT_AGENT_SETTINGS['CYPHER_MAX_RETRIES'])
    settings['LLM_PARSE_MAX_RETRIES'] = project.get('agentLlmParseMaxRetries', DEFAULT_AGENT_SETTINGS['LLM_PARSE_MAX_RETRIES'])
    settings['CREATE_GRAPH_IMAGE_ON_INIT'] = project.get('agentCreateGraphImageOnInit', DEFAULT_AGENT_SETTINGS['CREATE_GRAPH_IMAGE_ON_INIT'])
    settings['LOG_MAX_MB'] = project.get('agentLogMaxMb', DEFAULT_AGENT_SETTINGS['LOG_MAX_MB'])
    settings['LOG_BACKUP_COUNT'] = project.get('agentLogBackupCount', DEFAULT_AGENT_SETTINGS['LOG_BACKUP_COUNT'])
    settings['TOOL_PHASE_MAP'] = project.get('agentToolPhaseMap', DEFAULT_AGENT_SETTINGS['TOOL_PHASE_MAP'])
    settings['BRUTE_FORCE_MAX_WORDLIST_ATTEMPTS'] = project.get('agentBruteForceMaxWordlistAttempts', DEFAULT_AGENT_SETTINGS['BRUTE_FORCE_MAX_WORDLIST_ATTEMPTS'])
    settings['BRUTEFORCE_SPEED'] = project.get('agentBruteforceSpeed', DEFAULT_AGENT_SETTINGS['BRUTEFORCE_SPEED'])
    settings['HYDRA_ENABLED'] = project.get('hydraEnabled', DEFAULT_AGENT_SETTINGS['HYDRA_ENABLED'])
    settings['HYDRA_THREADS'] = project.get('hydraThreads', DEFAULT_AGENT_SETTINGS['HYDRA_THREADS'])
    settings['HYDRA_WAIT_BETWEEN_CONNECTIONS'] = project.get('hydraWaitBetweenConnections', DEFAULT_AGENT_SETTINGS['HYDRA_WAIT_BETWEEN_CONNECTIONS'])
    settings['HYDRA_CONNECTION_TIMEOUT'] = project.get('hydraConnectionTimeout', DEFAULT_AGENT_SETTINGS['HYDRA_CONNECTION_TIMEOUT'])
    settings['HYDRA_STOP_ON_FIRST_FOUND'] = project.get('hydraStopOnFirstFound', DEFAULT_AGENT_SETTINGS['HYDRA_STOP_ON_FIRST_FOUND'])
    settings['HYDRA_EXTRA_CHECKS'] = project.get('hydraExtraChecks', DEFAULT_AGENT_SETTINGS['HYDRA_EXTRA_CHECKS'])
    settings['HYDRA_VERBOSE'] = project.get('hydraVerbose', DEFAULT_AGENT_SETTINGS['HYDRA_VERBOSE'])
    settings['HYDRA_MAX_WORDLIST_ATTEMPTS'] = project.get('hydraMaxWordlistAttempts', DEFAULT_AGENT_SETTINGS['HYDRA_MAX_WORDLIST_ATTEMPTS'])
    settings['STEALTH_MODE'] = project.get('stealthMode', DEFAULT_AGENT_SETTINGS['STEALTH_MODE'])
    settings['PHISHING_SMTP_CONFIG'] = project.get('phishingSmtpConfig', DEFAULT_AGENT_SETTINGS['PHISHING_SMTP_CONFIG'])

    # Target scope (used by guardrail checks inside the agent)
    settings['TARGET_DOMAIN'] = project.get('targetDomain', '')
    settings['IP_MODE'] = project.get('ipMode', False)
    settings['TARGET_IPS'] = project.get('targetIps', [])

    logger.info(f"Loaded {len(settings)} agent settings for project {project_id}")
    return settings


def get_settings() -> dict[str, Any]:
    """
    Get current agent settings.

    Returns cached settings if loaded for a project, otherwise defaults.
    Use load_project_settings() to fetch settings for a specific project.

    Returns:
        Dictionary of settings in SCREAMING_SNAKE_CASE format
    """
    global _settings
    if _settings is not None:
        return _settings
    # Return defaults until a project is loaded
    logger.info("Using DEFAULT_AGENT_SETTINGS (no project loaded yet)")
    return DEFAULT_AGENT_SETTINGS.copy()


# Singleton settings instance
_settings: Optional[dict[str, Any]] = None
_current_project_id: Optional[str] = None


def load_project_settings(project_id: str) -> dict[str, Any]:
    """
    Fetch settings for a specific project from webapp API.

    Called by the orchestrator on every invocation to ensure settings
    reflect the latest values saved in the database.

    Args:
        project_id: The project ID received from the frontend

    Returns:
        Dictionary of settings in SCREAMING_SNAKE_CASE format
    """
    global _settings, _current_project_id

    webapp_url = os.environ.get('WEBAPP_API_URL')

    if not webapp_url:
        logger.warning("WEBAPP_API_URL not set, using DEFAULT_AGENT_SETTINGS")
        _settings = DEFAULT_AGENT_SETTINGS.copy()
        _current_project_id = project_id
        return _settings

    try:
        _settings = fetch_agent_settings(project_id, webapp_url)
        _current_project_id = project_id
        logger.info(f"Loaded {len(_settings)} agent settings from API for project {project_id}")
        return _settings

    except Exception as e:
        logger.error(f"Failed to fetch agent settings for project {project_id}: {e}")
        logger.warning("Falling back to DEFAULT_AGENT_SETTINGS")
        _settings = DEFAULT_AGENT_SETTINGS.copy()
        _current_project_id = project_id
        return _settings


def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a single agent setting value.

    Args:
        key: Setting name in SCREAMING_SNAKE_CASE
        default: Default value if setting not found

    Returns:
        Setting value or default
    """
    return get_settings().get(key, default)


def reload_settings(project_id: Optional[str] = None) -> dict[str, Any]:
    """Force reload of settings for a project."""
    global _settings, _current_project_id
    if project_id:
        _current_project_id = None  # Force refetch
        return load_project_settings(project_id)
    _settings = None
    _current_project_id = None
    return get_settings()


# =============================================================================
# TOOL PHASE RESTRICTION HELPERS (moved from params.py)
# =============================================================================

def is_tool_allowed_in_phase(tool_name: str, phase: str) -> bool:
    """Check if a tool is allowed in the given phase."""
    tool_phase_map = get_setting('TOOL_PHASE_MAP', {})
    allowed_phases = tool_phase_map.get(tool_name, [])
    return phase in allowed_phases


def get_allowed_tools_for_phase(phase: str) -> list:
    """Get list of tool names allowed in the given phase."""
    tool_phase_map = get_setting('TOOL_PHASE_MAP', {})
    return [
        tool_name
        for tool_name, allowed_phases in tool_phase_map.items()
        if phase in allowed_phases
    ]


def get_hydra_flags_from_settings() -> str:
    """Build Hydra CLI flags string from project settings.

    Returns a pre-formatted flag string like: -t 16 -f -e nsr -V
    Injected into brute force prompts so the LLM uses project-configured values.
    """
    parts = []
    parts.append(f"-t {get_setting('HYDRA_THREADS', 16)}")
    wait = get_setting('HYDRA_WAIT_BETWEEN_CONNECTIONS', 0)
    if wait > 0:
        parts.append(f"-W {wait}")
    timeout = get_setting('HYDRA_CONNECTION_TIMEOUT', 32)
    if timeout != 32:
        parts.append(f"-w {timeout}")
    if get_setting('HYDRA_STOP_ON_FIRST_FOUND', True):
        parts.append("-f")
    extra = get_setting('HYDRA_EXTRA_CHECKS', 'nsr')
    if extra:
        parts.append(f"-e {extra}")
    if get_setting('HYDRA_VERBOSE', True):
        parts.append("-V")
    return " ".join(parts)
