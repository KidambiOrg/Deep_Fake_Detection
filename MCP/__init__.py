"""
MCP (Model Context Protocol) package for Deepfake Detection
Contains all MCP tool trigger endpoints.
"""
from .mcp_deepfake_image import register_mcp_image_endpoints

__all__ = ['register_mcp_image_endpoints']
