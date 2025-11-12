"""
Agent and Thread Administration Utilities for Azure AI Foundry

This module provides functions to manage agents and threads in an Azure AI Foundry project.
Includes utilities for finding, creating, deleting, and reusing agents and threads.
Based on: https://github.com/frdeange/AifoundryAgentsCleaner/blob/main/main.py
"""

import os
from typing import Optional
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient


def find_agent_by_name(project_client: AIProjectClient, agent_name: str) -> Optional[str]:
    """
    Find an agent by name and return its ID.
    
    Args:
        project_client: An instance of AIProjectClient connected to your AI Foundry project
        agent_name: The name of the agent to find
        
    Returns:
        Agent ID if found, None otherwise
    """
    try:
        agents_list = list(project_client.agents.list_agents())
        for agent in agents_list:
            if agent.name == agent_name:
                return agent.id
        return None
    except Exception as e:
        print(f"Error finding agent by name '{agent_name}': {e}")
        return None


def find_thread_by_metadata(project_client: AIProjectClient, metadata_key: str, metadata_value: str) -> Optional[str]:
    """
    Find a thread by metadata key-value pair and return its ID.
    
    Args:
        project_client: An instance of AIProjectClient connected to your AI Foundry project
        metadata_key: The metadata key to search for
        metadata_value: The metadata value to match
        
    Returns:
        Thread ID if found, None otherwise
    """
    try:
        threads = list(project_client.agents.threads.list())
        for thread in threads:
            if hasattr(thread, 'metadata') and thread.metadata:
                if thread.metadata.get(metadata_key) == metadata_value:
                    return thread.id
        return None
    except Exception as e:
        print(f"Error finding thread by metadata '{metadata_key}={metadata_value}': {e}")
        return None


def delete_all_agents(project_client: AIProjectClient) -> None:
    """
    Delete all agents from the Azure AI Foundry project.
    
    Args:
        project_client: An instance of AIProjectClient connected to your AI Foundry project
    """
    print("Listing agents...")
    agents_list = list(project_client.agents.list_agents())  # Convert to static list
    
    if not agents_list:
        print("No agents found.")
        return
    
    print(f"Found {len(agents_list)} agent(s) to delete.")
    print (f"Deleting {len(agents_list)} agent(s)...")
    for agent in agents_list:
        try:
            
            project_client.agents.delete_agent(agent.id)
        except Exception as e:
            print(f"Warning: Could not delete agent {agent.id}: {e}")


def delete_all_threads(project_client: AIProjectClient) -> None:
    """
    Delete all threads from the Azure AI Foundry project.
    
    Args:
        project_client: An instance of AIProjectClient connected to your AI Foundry project
    """
    print("Listing threads...")
    # Note: returns a pageable; convert to static list to avoid pagination issues
    threads = list(project_client.agents.threads.list())  # Convert to static list
    
    if not threads:
        print("No threads found.")
        return
    
    print(f"Found {len(threads)} thread(s) to delete.")
    print (f"Deleting {len(threads)} thread(s)...")
    for th in threads:
        try:
            
            project_client.agents.threads.delete(th.id)
        except Exception as e:
            print(f"Warning: Could not delete thread {th.id}: {e}")


def cleanup_ai_foundry_resources(endpoint: Optional[str] = None) -> None:
    """
    Main function to clean up all agents and threads from Azure AI Foundry.
    
    Args:
        endpoint: The AI Foundry endpoint URL. If not provided, will be read from 
                 AZURE_AI_PROJECT_ENDPOINT environment variable.
    
    Example:
        >>> from utils.agents_delete import cleanup_ai_foundry_resources
        >>> cleanup_ai_foundry_resources("https://your-project.api.azureml.ms")
    """
    load_dotenv()
    
    if endpoint is None:
        endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
    
    if not endpoint:
        raise ValueError(
            "AI Foundry endpoint not provided. Set AZURE_AI_PROJECT_ENDPOINT environment "
            "variable or pass endpoint parameter."
        )
    
    print(f"Connecting to AI Foundry endpoint: {endpoint}")
    credential = DefaultAzureCredential()
    project_client = AIProjectClient(endpoint=endpoint, credential=credential)
    
    delete_all_agents(project_client)
    delete_all_threads(project_client)
    
    print("Cleanup process finished.")


if __name__ == "__main__":
    # When run directly, perform cleanup using environment variables
    cleanup_ai_foundry_resources()
