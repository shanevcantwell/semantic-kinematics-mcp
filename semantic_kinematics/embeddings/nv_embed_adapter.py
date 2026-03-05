"""
Direct NV-Embed-v2 adapter using transformers.

Bypasses sentence-transformers to avoid API incompatibilities with NV-Embed-v2's
custom code. Loads model in fp32 by default (requires ~28GB VRAM, e.g. RTX-8000).

NV-Embed-v2's custom modeling code (BidirectionalMistralModel) was written for
transformers ~4.42. Two compatibility issues with transformers >=4.45:
  1. DynamicCache.get_usable_length was removed (renamed to get_seq_length)
  2. MistralDecoderLayer now expects pre-computed position_embeddings parameter

We fix both by patching the BidirectionalMistralModel.forward after loading,
rather than pinning transformers to an old version.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from semantic_kinematics.embeddings.base import EmbeddingAdapter


def _patch_bidirectional_mistral(embedding_model):
    """
    Patch BidirectionalMistralModel.forward for transformers>=4.45 compatibility.

    The custom forward() was written for transformers ~4.42 where MistralDecoderLayer
    computed its own rotary embeddings from position_ids. In >=4.45, the parent model
    must compute position_embeddings and pass them to each layer.

    Also disables use_cache (embedding models never need KV cache).
    """
    embedding_model.config.use_cache = False

    # Check if the patched API is needed (position_embeddings parameter exists)
    import inspect
    layer = embedding_model.layers[0]
    sig = inspect.signature(layer.forward)
    if 'position_embeddings' not in sig.parameters:
        return  # Old transformers, no patch needed

    from transformers.modeling_attn_mask_utils import (
        _prepare_4d_attention_mask,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Build 4D attention mask for eager attention
        attention_mask = _prepare_4d_attention_mask(
            attention_mask, inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds

        # Compute rotary embeddings once, pass to all layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    import types
    embedding_model.forward = types.MethodType(patched_forward, embedding_model)


class NVEmbedAdapter(EmbeddingAdapter):
    """
    Direct adapter for NV-Embed-v2 using transformers.

    Uses the model's native encode method which handles pooling correctly.
    Loads in fp16 by default (~14GB VRAM). fp32 requires ~28GB and OOMs on
    48GB GPUs during batch inference due to activation memory.

    Memory management:
        - Model lazy-loads to GPU on first embed
        - Unloads completely after use (frees VRAM)
        - Set unload_after_use=False to keep model resident (faster, uses VRAM)
    """

    DEFAULT_MODEL_PATH = "nvidia/NV-Embed-v2"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = True,
        max_length: int = 32768,
        unload_after_use: bool = True,
    ):
        """
        Initialize NV-Embed-v2 adapter.

        Args:
            model_path: Path to model directory
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_fp16: Use float16 precision (default: True). fp32 needs ~28GB
                      VRAM and OOMs during batch inference on 48GB GPUs.
            max_length: Maximum sequence length
            unload_after_use: Fully unload model after each embed (default: True)
        """
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._use_fp16 = use_fp16
        self._max_length = max_length
        self._unload_after_use = unload_after_use
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load model and tokenizer to GPU."""
        if self._model is None:
            from transformers import AutoModel, AutoTokenizer

            dtype = torch.float16 if self._use_fp16 else torch.float32

            self._model = AutoModel.from_pretrained(
                self._model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )

            # Force fp16 on ALL parameters and buffers - custom code may ignore torch_dtype
            if self._use_fp16:
                self._model = self._model.half()
                for name, buf in self._model.named_buffers():
                    if buf.dtype == torch.float32:
                        buf.data = buf.data.half()

            # Patch BidirectionalMistralModel for transformers>=4.45 compatibility
            if hasattr(self._model, 'embedding_model'):
                _patch_bidirectional_mistral(self._model.embedding_model)

            self._model = self._model.to(self._device)
            self._model.eval()

            # Tokenizer stays resident (small)
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)

        return self._model, self._tokenizer

    def unload(self):
        """Fully unload model from memory (frees VRAM)."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()

    @property
    def model_name(self) -> str:
        return f"NVEmbed:{self._model_path}"

    @property
    def dimensions(self) -> int:
        return 4096  # NV-Embed-v2 fixed dimension

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._model is not None

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for text.

        Uses the model's native encode() which handles tokenization, instruction
        masking, pool_mask construction, and latent attention pooling correctly.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector (4096 dims)
        """
        model, _ = self._load_model()

        try:
            # model.encode() is @torch.no_grad() internally
            embeddings = model.encode([text], max_length=self._max_length)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings[0].cpu().float().numpy()
        finally:
            if self._unload_after_use:
                self.unload()

    def embed_batch(self, texts: List[str], chunk_size: int = 8) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Loads model, processes texts in chunks to limit memory, then unloads.
        Model stays loaded across chunks for efficiency.

        Args:
            texts: List of input texts
            chunk_size: Max texts per forward pass (default: 8, limits VRAM usage)

        Returns:
            Array of shape (len(texts), 4096)
        """
        if not texts:
            return np.array([])

        model, _ = self._load_model()

        try:
            all_embeddings = []

            for i in range(0, len(texts), chunk_size):
                chunk = texts[i:i + chunk_size]
                embeddings = model.encode(chunk, max_length=self._max_length)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().float().numpy())

            return np.concatenate(all_embeddings, axis=0)
        finally:
            if self._unload_after_use:
                self.unload()
