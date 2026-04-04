"""Reverse tunnel management for exposing model containers via public URLs using pyngrok."""

import asyncio
import os
from typing import Any, Dict, Optional

from pyngrok import ngrok


async def start_tunnel(local_port: int, model_id: str) -> tuple[Optional[str], None]:
    """Start a pyngrok (ngrok) reverse tunnel for the given local port.
    
    Args:
        local_port: The local port to expose (e.g., 8001)
        model_id: Model identifier for logging
    
    Returns:
        Tuple of (public_url, None) - ngrok doesn't expose PID, only URL
    """
    try:
        # Connect to ngrok and get public URL
        tunnel_url = await asyncio.to_thread(
            ngrok.connect,
            local_port,
            "http",
        )
        
        # Extract the actual URL string
        if hasattr(tunnel_url, "public_url"):
            public_url = tunnel_url.public_url
        else:
            public_url = str(tunnel_url)
        
        return public_url, None
    
    except Exception as exc:
        raise RuntimeError(f"Failed to start ngrok tunnel: {str(exc)}")


async def stop_tunnel(tunnel_url: str) -> bool:
    """Stop a running ngrok tunnel.
    
    Args:
        tunnel_url: Public URL of the tunnel to stop
    
    Returns:
        True if stopped successfully, False otherwise
    """
    try:
        await asyncio.to_thread(ngrok.disconnect, tunnel_url)
        return True
    except Exception:
        return False


def get_tunnel_status(model_id: str) -> Dict[str, Any]:
    """Get the status of a tunnel.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dict with tunnel status and URL if available
    """
    try:
        tunnels = ngrok.get_tunnels()
        for tunnel in tunnels:
            if model_id in str(tunnel):
                return {"status": "running", "url": tunnel.public_url}
        return {"status": "not_found", "url": None}
    except Exception:
        return {"status": "error", "url": None}


