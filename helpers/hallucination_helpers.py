"""
Hallucination Demo Helpers

This module provides functions to log intentional hallucinations to Splunk Agent
Observability for demonstration purposes. Each domain can define hallucination
examples in their config.yaml under the `demo_hallucinations` key.
"""
import uuid
import logging
from typing import Optional, List, Dict, Any, Union
from splunk_ao import SplunkAOLogger
from helpers.sao_api_helpers import create_sao_logger

logger = logging.getLogger(__name__)


def log_hallucination(
    project_name: str,
    agent_stream: str,
    question: str,
    context_docs: List[str],
    hallucinated_answer: str,
    model: str = "gemma4",
    session_name: str = "Hallucination Demo",
    existing_logger: Optional[Union[SplunkAOLogger, Any]] = None,
) -> bool:
    """
    Log a hallucination trace to Splunk Agent Observability for demonstration purposes.
    
    This creates a complete trace with:
    - A retriever span showing the retrieved context
    - An LLM span showing the question and hallucinated (wrong) answer
    
    Args:
        project_name: The Splunk AO project name to log to
        agent_stream: The Splunk AO agent stream to log to
        question: The user's question
        context_docs: List of retrieved context documents (real content)
        hallucinated_answer: The intentionally wrong answer that contradicts the context
        model: The model name to log (default: gpt-4o)
        session_name: Name for the Splunk AO session (default: "Hallucination Demo")
        existing_logger: Optional existing logger or context to reuse (if session already started)
        
    Returns:
        bool: True if logging succeeded, False otherwise
    """
    try:
        logger.info(f"Logging hallucination to project: {project_name}, agent stream: {agent_stream}")
        
        # Use existing logger if provided, otherwise create a new one
        if existing_logger:
            logger.info("Using existing SAO session for hallucification demo")
            # If it's a decorator/context, get the logger instance
            if hasattr(existing_logger, 'get_logger_instance'):
                sao_logger = existing_logger.get_logger_instance()
                logger.info("Extracted logger instance from SAO context")
            else:
                sao_logger = existing_logger
            created_new_session = False
        else:
            logger.info("Creating new SAO session for hallucination demo")
            sao_logger = create_sao_logger(project_name, agent_stream)
            sao_logger.enable_agent_control()
            # Start a named session for easy identification
            session_id = str(uuid.uuid4())[:10]
            sao_logger.start_session(name=session_name, external_id=session_id)
            created_new_session = True
        
        # Start a workflow trace
        sao_logger.start_trace(
            input=question,
            name="Hallucination Demo",
        )
        
        # Add retriever span with the real context
        sao_logger.add_retriever_span(
            input=question,
            output=context_docs,
            name="RAG Retrieval",
            duration_ns=int(1.3e8),
            status_code=200
        )
        
        # Build the LLM input with context
        context_text = "\n\n".join(context_docs)
        llm_input = f"""Human: You are a helpful assistant. Given the context below, please answer the following question:

{context_text}

Question: {question}"""
        
        # Add LLM span with the hallucinated answer
        sao_logger.add_llm_span(
            input=llm_input,
            output=hallucinated_answer,
            model=model,
            name="LLM Response",
            num_input_tokens=len(llm_input.split()) * 2,  # Rough estimate
            num_output_tokens=len(hallucinated_answer.split()) * 2,
            total_tokens=len(llm_input.split()) * 2 + len(hallucinated_answer.split()) * 2,
            duration_ns=int(1.2e8),
            metadata={"temperature": "0.1", "demo_type": "hallucination"},
            temperature=0.1,
            status_code=200,
            time_to_first_token_ns=500000,
        )
        
        # Conclude the trace
        sao_logger.conclude(
            output=hallucinated_answer,
            duration_ns=int(2.5e8),
            status_code=200
        )
        
        # Only flush if we created a new session
        # if created_new_session:
        sao_logger.flush()
        
        logger.info(f"Successfully logged hallucination to project: {project_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to log hallucination: {e}")
        return False


def log_hallucination_for_domain(
    domain_name: str,
    domain_config: Dict[str, Any],
    rag_retriever_func: Optional[callable] = None,
    hallucination_index: int = 0,
    existing_logger: Optional[Union[SplunkAOLogger, Any]] = None,
) -> bool:
    """
    Log a hallucination for a specific domain using its config.
    
    This reads the hallucination examples from the domain's config.yaml
    and logs them to Splunk Agent Observability. If a RAG retriever function
    is provided, it will fetch real context; otherwise uses the pre-defined context.
    
    Args:
        domain_name: Name of the domain (e.g., "finance", "healthcare")
        domain_config: The domain's configuration dictionary
        rag_retriever_func: Optional function to retrieve real context from RAG
        hallucination_index: Which hallucination example to use (default: 0)
        existing_logger: Optional existing logger or context to reuse (if session already started)
        
    Returns:
        bool: True if logging succeeded, False otherwise
    """
    sao_config = domain_config.get("splunk_ao", {})
    project_name = sao_config.get("project", f"sao-demo-{domain_name}")
    agent_stream = sao_config.get("agent_stream", "default")
    
    # Get hallucination examples
    hallucinations = domain_config.get("demo_hallucinations", [])
    
    if not hallucinations:
        logger.warning(f"No hallucination examples defined for domain: {domain_name}")
        return False
    
    if hallucination_index >= len(hallucinations):
        hallucination_index = 0
        
    hallucination = hallucinations[hallucination_index]
    question = hallucination.get("question", "")
    hallucinated_answer = hallucination.get("hallucinated_answer", "")
    context_docs = hallucination.get("context", [])
    
    if not question or not hallucinated_answer:
        logger.error(f"Invalid hallucination config for domain: {domain_name}")
        return False
    
    # If RAG retriever is provided and no pre-defined context, fetch real context
    if rag_retriever_func and not context_docs:
        try:
            context_docs = rag_retriever_func(question)
        except Exception as e:
            logger.warning(f"Failed to retrieve context from RAG: {e}")
            context_docs = ["[Context retrieval failed]"]
    
    # Use pre-defined context if available
    if not context_docs:
        context_docs = ["[No context available]"]
    
    # Create a session name that includes the domain
    session_name = f"{domain_name.title()} Hallucination Demo"
    
    return log_hallucination(
        project_name=project_name,
        agent_stream=agent_stream,
        question=question,
        context_docs=context_docs,
        hallucinated_answer=hallucinated_answer,
        session_name=session_name,
        existing_logger=existing_logger,
    )
