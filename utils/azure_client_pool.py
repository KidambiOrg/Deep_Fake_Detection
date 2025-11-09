"""
Azure AI Client Pool for Agent Framework

Provides singleton Azure AI client to avoid redundant authentication and connection overhead.
All agents share a single client instance, reducing initialization time from 3-5s to <100ms.
"""

import os
import logging
import threading
from typing import Optional
from agent_framework.azure import AzureAIAgentClient
from azure.identity.aio import AzureCliCredential

logger = logging.getLogger(__name__)

# Global singleton client
_global_azure_client: Optional[AzureAIAgentClient] = None
_client_lock = threading.Lock()


def get_or_create_azure_client() -> AzureAIAgentClient:
    """
    Get or create global Azure AI client instance (singleton pattern).
    
    Thread-safe initialization ensures only one client is created even with
    concurrent requests. Subsequent calls return the cached client immediately.
    
    Benefits:
    - Reduces agent initialization time from 3-5s to <100ms
    - Shares authentication credentials across all agents
    - Reuses HTTP connection pool
    
    Returns:
        AzureAIAgentClient instance configured for Azure AI Foundry
    """
    global _global_azure_client
    
    # Fast path: if client already exists, return immediately
    if _global_azure_client is not None:
        logger.debug("Reusing cached Azure AI client")
        return _global_azure_client
    
    # Slow path: need to create client (with lock to prevent concurrent initialization)
    with _client_lock:
        # Double-check after acquiring lock
        if _global_azure_client is not None:
            logger.debug("Reusing cached Azure AI client (after lock)")
            return _global_azure_client
        
        logger.info("Initializing Azure AI client for the first time...")
        
        # Get configuration from environment
        project_endpoint = os.getenv('AZURE_AI_PROJECT_ENDPOINT')
        deployment_name = os.getenv('AZURE_AI_MODEL_DEPLOYMENT_NAME')
        
        if not project_endpoint or not deployment_name:
            raise ValueError(
                "Missing Azure AI configuration. Set AZURE_AI_PROJECT_ENDPOINT "
                "and AZURE_AI_MODEL_DEPLOYMENT_NAME environment variables."
            )
        
        # Create client with async credential
        _global_azure_client = AzureAIAgentClient(
            project_endpoint=project_endpoint,
            model_deployment_name=deployment_name,
            async_credential=AzureCliCredential()
        )
        
        logger.info(f"✅ Azure AI client initialized successfully")
        logger.info(f"   Endpoint: {project_endpoint}")
        logger.info(f"   Deployment: {deployment_name}")
        
        return _global_azure_client


def reset_azure_client():
    """
    Reset the global Azure AI client (for testing or reinitialization).
    
    Warning: This will force all agents to recreate their client on next use.
    """
    global _global_azure_client
    with _client_lock:
        _global_azure_client = None
        logger.info("Azure AI client reset")
