"""Reverse tunnel management for exposing model containers via public URLs using official ngrok SDK."""

import asyncio
import os
from typing import Any, Dict, Optional

import ngrok


_ngrok_initialized = False


def _ensure_ngrok_initialized() -> None:
    """Initialize ngrok with auth token."""
    global _ngrok_initialized
    if _ngrok_initialized:
        return

    token = os.environ.get("NGROK_AUTHTOKEN") or os.environ.get("PYNGROK_AUTHTOKEN")
    if not token:
        raise RuntimeError(
            "Tunnel requested but NGROK_AUTHTOKEN is not set. "
            "Set NGROK_AUTHTOKEN in your environment to enable public tunnel URLs."
        )

    # Set auth token for ngrok (sync function)
    ngrok.set_auth_token(token)  # type: ignore[attr-defined]
    _ngrok_initialized = True


async def start_tunnel(local_port: int, model_id: str) -> tuple[str, None]:
    """Start an ngrok reverse tunnel for the given local port.
    
    Args:
        local_port: The local port to expose (e.g., 8000)
        model_id: Model identifier for logging
    
    Returns:
        Tuple of (public_url, None) - ngrok doesn't expose PID, only URL
    """
    try:
        _ensure_ngrok_initialized()
        
        print(f"[Tunnel] Disconnecting all existing ngrok tunnels...")
        # Disconnect any existing tunnels to ensure fresh connection
        try:
            listeners = await asyncio.to_thread(ngrok.get_listeners)  # type: ignore[attr-defined]
            print(f"[Tunnel] Found {len(listeners)} active tunnels")
            if listeners:
                for listener in listeners:
                    print(f"[Tunnel] Disconnecting: {listener.url()}")
                    await asyncio.to_thread(ngrok.disconnect, listener.url())
        except Exception as e:
            print(f"[Tunnel] Could not disconnect existing tunnels: {e}")

        # Wait a moment for cleanup
        await asyncio.sleep(1)
        
        print(f"[Tunnel] Starting NEW ngrok tunnel to http://127.0.0.1:{local_port}...")
        
        # Build ngrok connect parameters
        addr = f"http://127.0.0.1:{local_port}"
        
        # Check if custom domain is configured
        custom_domain = os.environ.get("NGROK_CUSTOM_DOMAIN")
        
        # Connect tunnel (blocking call, run in thread)
        if custom_domain:
            print(f"[Tunnel] Using custom domain: {custom_domain}")
            listener = await asyncio.to_thread(
                ngrok.connect,
                addr,
                domain=custom_domain,
            )
        else:
            print(f"[Tunnel] No custom domain configured, ngrok will auto-generate")
            listener = await asyncio.to_thread(
                ngrok.connect,
                addr,
            )
        
        print(f"[Tunnel] Response type: {type(listener)}")
        print(f"[Tunnel] Response object: {listener}")
        
        # Extract the public URL from Listener.url() - it's a method
        # Returns a string like https://xxxx.ngrok.io
        public_url = listener.url()
        print(f"[Tunnel] Got public_url: {public_url}")
        
        print(f"✓ [Tunnel] Successfully created tunnel: http://127.0.0.1:{local_port} -> {public_url}")
        
        # CRITICAL: Give ngrok time to establish and stabilize the tunnel
        print(f"[Tunnel] Waiting for tunnel to stabilize (3 seconds)...")
        await asyncio.sleep(3)
        
        # Test the tunnel with retries
        print(f"[Tunnel] Testing tunnel connectivity...")
        
        # First, verify local endpoint is working
        print(f"[Tunnel] Verifying local endpoint at http://127.0.0.1:{local_port}/...")
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://127.0.0.1:{local_port}/")
                print(f"[Tunnel] Local endpoint returned: {response.status_code}")
        except Exception as e:
            print(f"[Tunnel] WARNING: Local endpoint not responding: {e}")
            print(f"[Tunnel] This may cause tunnel issues!")
        
        # Now test the tunnel URL
        max_test_attempts = 5
        test_attempt = 0
        tunnel_ready = False
        
        while test_attempt < max_test_attempts and not tunnel_ready:
            test_attempt += 1
            try:
                import httpx
                async with httpx.AsyncClient(
                    follow_redirects=True, 
                    timeout=httpx.Timeout(10.0, connect=15.0)
                ) as client:
                    response = await client.get(f"{public_url}/")
                    if response.status_code < 500:
                        print(f"[Tunnel] Test request returned: {response.status_code} - tunnel is ready")
                        tunnel_ready = True
                    else:
                        print(f"[Tunnel] Test request returned: {response.status_code} - retrying...")
            except Exception as test_err:
                print(f"[Tunnel] Test attempt {test_attempt}/{max_test_attempts} failed: {test_err}")
                if test_attempt < max_test_attempts:
                    await asyncio.sleep(2)
                else:
                    print(f"[Tunnel] WARNING: Could not verify tunnel after {max_test_attempts} attempts")
                    print(f"[Tunnel] This may be expected if the tunnel service is under load")
                    print(f"[Tunnel] The tunnel URL is still valid and may work with retries: {public_url}")
        
        return public_url, None
    
    except Exception as exc:
        print(f"✗ [Tunnel] Failed to start ngrok tunnel: {str(exc)}")
        import traceback
        traceback.print_exc()
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
        listeners = ngrok.get_listeners()  # type: ignore[attr-defined]
        if listeners:
            for listener in listeners:
                if model_id in str(listener):
                    return {"status": "running", "url": listener.url()}
        return {"status": "not_found", "url": None}
    except Exception:
        return {"status": "error", "url": None}



