"""
Model management command module.

Tools for managing embedding model lifecycle (load/unload/status).
Primarily useful for NV-Embed-v2 which requires explicit GPU memory management.
"""

from typing import Any, Dict, List

from mcp.types import Tool

from semantic_kinematics.mcp.state_manager import StateManager


def get_tools() -> List[Tool]:
    """Return model management tool definitions."""
    return [
        Tool(
            name="model_status",
            description=(
                "Report current embedding model status: backend type, model name, "
                "dimensions, whether loaded in memory, and embedding cache size."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="model_load",
            description=(
                "Load the embedding model into memory. For nv_embed backend, this loads "
                "the model to GPU VRAM (~28GB fp32). For lmstudio, initializes the API "
                "client. Optionally switch backend before loading."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "description": (
                            "Switch to this backend before loading. "
                            "Options: 'nv_embed', 'lmstudio', 'sentence_transformers'"
                        ),
                    },
                    "base_url": {
                        "type": "string",
                        "description": "API URL for lmstudio backend (e.g. http://192.168.137.2:1234/v1)",
                    },
                    "model_name": {
                        "type": "string",
                        "description": "Model name for API backends",
                    },
                },
            },
        ),
        Tool(
            name="model_unload",
            description=(
                "Unload the embedding model from memory. For nv_embed, frees GPU VRAM. "
                "Also clears the embedding cache."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "clear_cache": {
                        "type": "boolean",
                        "description": "Also clear the embedding cache (default: true)",
                        "default": True,
                    },
                },
            },
        ),
    ]


async def model_status(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Report current model status."""
    result = {
        "backend": manager._backend,
        "cache_size": len(manager._embedding_cache),
    }

    if manager._adapter is not None:
        adapter = manager._adapter
        result["model_name"] = adapter.model_name
        result["is_loaded"] = adapter.is_loaded
        try:
            result["dimensions"] = adapter.dimensions
        except Exception:
            result["dimensions"] = "unknown (model not loaded)"
    else:
        result["model_name"] = "not initialized"
        result["is_loaded"] = False
        result["dimensions"] = "unknown"

    return result


async def model_load(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Load the embedding model into memory."""
    backend = args.get("backend")
    base_url = args.get("base_url")
    model_name = args.get("model_name")

    # Switch backend if requested
    if backend:
        kwargs = {}
        # Only pass kwargs that the target backend accepts
        if backend == "lmstudio":
            if base_url:
                kwargs["base_url"] = base_url
            # Fall back to EMBEDDING_MODEL env var if no model_name provided
            if not model_name:
                import os
                model_name = os.environ.get("EMBEDDING_MODEL")
            if model_name:
                kwargs["model_name"] = model_name
        elif backend == "nv_embed":
            # NVEmbedAdapter accepts: model_path, device, use_fp16, max_length, unload_after_use
            pass
        elif backend == "sentence_transformers":
            if model_name:
                kwargs["model_name"] = model_name
        manager.set_backend(backend, **kwargs)

    # Force adapter initialization
    adapter = manager.get_adapter()

    # For NV-Embed, explicitly trigger model load
    if hasattr(adapter, "_load_model"):
        adapter._load_model()

    return {
        "status": "loaded",
        "backend": manager._backend,
        "model_name": adapter.model_name,
        "is_loaded": adapter.is_loaded,
    }


async def model_unload(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Unload the embedding model from memory."""
    clear_cache = args.get("clear_cache", True)

    cache_cleared = 0
    if clear_cache:
        cache_cleared = manager.clear_cache()

    if manager._adapter is not None:
        manager._adapter.unload()
        return {
            "status": "unloaded",
            "cache_entries_cleared": cache_cleared,
        }

    return {
        "status": "no model was loaded",
        "cache_entries_cleared": cache_cleared,
    }
