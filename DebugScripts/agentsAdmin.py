"""
Agent and Thread Administration Utilities for Azure AI Foundry

This module provides functions to manage agents and threads in an Azure AI Foundry project.
Includes utilities for finding, creating, deleting, and reusing agents and threads.
Based on: https://github.com/frdeange/AifoundryAgentsCleaner/blob/main/main.py
"""

import os
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient


def find_agent_by_name(agents_client: AgentsClient, agent_name: str) -> Optional[str]:
    """
    Find an agent by name and return its ID.
    
    Args:
        agents_client: An instance of AgentsClient connected to your AI Foundry project
        agent_name: The name of the agent to find
        
    Returns:
        Agent ID if found, None otherwise
    """
    try:
        agents_response = agents_client.list_agents()
        agents_list = list(agents_response)
        for agent in agents_list:
            if agent.name == agent_name:
                return agent.id
        return None
    except Exception as e:
        print(f"Error finding agent by name '{agent_name}': {e}")
        return None


def find_thread_by_metadata(agents_client: AgentsClient, metadata_key: str, metadata_value: str) -> Optional[str]:
    """
    Find a thread by metadata key-value pair and return its ID.
    
    Args:
        agents_client: An instance of AgentsClient connected to your AI Foundry project
        metadata_key: The metadata key to search for
        metadata_value: The metadata value to match
        
    Returns:
        Thread ID if found, None otherwise
    """
    try:
        threads_response = agents_client.threads.list()
        threads = list(threads_response)
        for thread in threads:
            if hasattr(thread, 'metadata') and thread.metadata:
                if thread.metadata.get(metadata_key) == metadata_value:
                    return thread.id
        return None
    except Exception as e:
        print(f"Error finding thread by metadata '{metadata_key}={metadata_value}': {e}")
        return None


def delete_all_agents(agents_client: AgentsClient) -> None:
    """
    Delete all agents from the Azure AI Foundry project.
    
    Args:
        agents_client: An instance of AgentsClient connected to your AI Foundry project
    """
    print("Listing agents...")
    try:
        # List all agents - list_agents() returns ItemPaged, need to convert to list
        agents_response = agents_client.list_agents()
        print(f"  Response type: {type(agents_response)}")
        
        # Convert ItemPaged to list by iterating
        agents_list = list(agents_response)
        print(f"  Agents list length: {len(agents_list)}")
        
        if not agents_list:
            print("No agents found.")
            return               
        
        print(f"\nDeleting {len(agents_list)} agent(s) in parallel...")
        
        def delete_agent(agent):
            try:
                agents_client.delete_agent(agent.id)
                return (True, agent.id, None)
            except Exception as e:
                return (False, agent.id, str(e))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(delete_agent, agent): agent for agent in agents_list}
            for future in as_completed(futures):
                success, agent_id, error = future.result()
                if not success:
                    print(f"    ✗ Warning: Could not delete agent {agent_id}: {error}")
    except Exception as e:
        print(f"Error listing agents: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


def delete_all_threads(agents_client: AgentsClient) -> None:
    """
    Delete all threads from the Azure AI Foundry project.
    
    Args:
        agents_client: An instance of AgentsClient connected to your AI Foundry project
    """
    print("Listing threads...")
    try:
        threads_response = agents_client.threads.list()
        # Convert ItemPaged to list
        threads = list(threads_response)
        
        if not threads:
            print("No threads found.")
            return
        
        print(f"Found {len(threads)} thread(s) to delete.")
        print(f"Deleting {len(threads)} thread(s) in parallel...")
        
        def delete_thread(thread):
            try:
                agents_client.threads.delete(thread.id)
                return (True, thread.id, None)
            except Exception as e:
                return (False, thread.id, str(e))
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(delete_thread, thread): thread for thread in threads}
            for future in as_completed(futures):
                success, thread_id, error = future.result()
                if not success:
                    print(f"    ✗ Warning: Could not delete thread {thread_id}: {error}")
    except Exception as e:
        print(f"Error with threads: {e}")
        print("Note: Thread operations may not be available.")


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
        print("\n" + "="*80)
        print("ERROR: Azure AI Project Endpoint not configured")
        print("="*80)
        print("\nThe AZURE_AI_PROJECT_ENDPOINT environment variable is not set.")
        print("\nTo find your endpoint:")
        print("  1. Go to https://ai.azure.com/")
        print("  2. Select your project")
        print("  3. Go to 'Settings' -> 'Project properties'")
        print("  4. Copy the 'Project connection string' or 'Endpoint URL'")
        print("     Format: https://<resource>.services.ai.azure.com/api/projects/<project-name>")
        print("\nTo set the endpoint:")
        print("  Option 1: Create a .env file in the project root:")
        print('            AZURE_AI_PROJECT_ENDPOINT="your-endpoint-here"')
        print("  Option 2: Set as environment variable:")
        print('            set AZURE_AI_PROJECT_ENDPOINT=your-endpoint-here')
        print("="*80 + "\n")
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT not configured")
    
    print(f"Connecting to AI Foundry endpoint: {endpoint}")
    print(f"Using DefaultAzureCredential for authentication...")
    
    try:
        credential = DefaultAzureCredential()
        with AgentsClient(endpoint=endpoint, credential=credential) as agents_client:
            print(f"✓ Successfully created AgentsClient")
            print(f"  Client type: {type(agents_client)}")
            
            delete_all_agents(agents_client)
            delete_all_threads(agents_client)
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("Cleanup process finished.")


if __name__ == "__main__":
    # When run directly, perform cleanup using environment variables
    cleanup_ai_foundry_resources()
