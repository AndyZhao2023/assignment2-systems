#!/usr/bin/env python3
"""
Enhanced NVTX annotations for more detailed profiling.
This module provides additional annotated functions for key model components.
"""

import contextlib
from typing import Optional

import einx
import torch
from einops import einsum, rearrange
from jaxtyping import Float, Int
from torch import Tensor
from torch.cuda import nvtx

from logging_config import create_profiling_logger

# Set up logging
logger = create_profiling_logger(__name__)

# Import cs336_basics components (may not resolve in some environments)
try:
    import cs336_basics.model
    from cs336_basics.model import scaled_dot_product_attention
    CS336_AVAILABLE = True
except ImportError:
    CS336_AVAILABLE = False
    # Create dummy functions to prevent import errors
    cs336_basics = None
    scaled_dot_product_attention = None

# Check NVTX availability
NVTX_AVAILABLE = False  # Initialize before try block
try:
    NVTX_AVAILABLE = torch.cuda.is_available()
    if NVTX_AVAILABLE:
        # Test if NVTX actually works
        try:
            with nvtx.range("test"):
                pass
        except (RuntimeError, AttributeError):
            NVTX_AVAILABLE = False
except (ImportError, RuntimeError, AttributeError):
    NVTX_AVAILABLE = False

if not NVTX_AVAILABLE:
    class DummyNVTX:
        """Dummy NVTX implementation for CPU/non-CUDA environments."""
        @staticmethod
        def range(_):
            """Return a no-op context manager."""
            return contextlib.nullcontext()
    nvtx = DummyNVTX()  # Override the imported nvtx with dummy


def annotated_linear_forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
    """NVTX-annotated version of Linear.forward."""
    weight_shape = f"linear_{self.weight.shape[0]}x{self.weight.shape[1]}"
    with nvtx.range(weight_shape):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


def annotated_rmsnorm_forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
    """NVTX-annotated version of RMSNorm.forward."""
    with nvtx.range("rmsnorm"):
        # Original implementation
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return (self.weight * x).to(in_dtype)


def annotated_swiglu_forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
    """NVTX-annotated version of SwiGLU.forward."""
    with nvtx.range("swiglu_ffn"):
        with nvtx.range("swiglu_gate_proj"):
            gate = self.w1(x)

        with nvtx.range("swiglu_up_proj"):
            up = self.w3(x)

        with nvtx.range("swiglu_activation"):
            # SiLU activation
            gate_activated = gate * torch.sigmoid(gate)
            gated = gate_activated * up

        with nvtx.range("swiglu_down_proj"):
            return self.w2(gated)


def annotated_rope_forward(
    self, x: Float[Tensor, "... seq d"], pos_ids: Int[Tensor, "... seq"]
) -> Float[Tensor, "... seq d"]:
    """NVTX-annotated version of RotaryEmbedding.forward."""
    with nvtx.range("rope_positional_encoding"):
        with nvtx.range("rope_split"):
            x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2)

        with nvtx.range("rope_get_frequencies"):
            freq_pattern = 'cos_sin [pos] half_dim, ... -> cos_sin ... half_dim'
            cos, sin = einx.get_at(freq_pattern, self._freq_cis_cache, pos_ids)

        with nvtx.range("rope_rotation"):
            x1_rot = cos * x1 - sin * x2
            x2_rot = sin * x1 + cos * x2
            rearrange_pattern = '... x_half, ... x_half -> ... (x_half (1 + 1))'
            result = einx.rearrange(rearrange_pattern, x1_rot, x2_rot).contiguous()

        return result


def annotated_attention_forward(
    self,
    x: Float[Tensor, "... seq d_model"],
    token_positions: Optional[Int[Tensor, "... seq"]] = None
) -> Float[Tensor, "... seq d_model"]:
    """NVTX-annotated version of CausalMultiHeadSelfAttention.forward."""
    with nvtx.range("multihead_attention"):
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        with nvtx.range("attention_qkv_projection"):
            Q = self.q_proj(x)
            K = self.k_proj(x)
            V = self.v_proj(x)

        with nvtx.range("attention_reshape_heads"):
            Q, K, V = (
                rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
                for X in (Q, K, V)
            )

        with nvtx.range("attention_position_setup"):
            if token_positions is None:
                seq_range = torch.arange(sequence_length, device=x.device)
                token_positions = einx.rearrange(
                    "seq -> b... seq", seq_range, b=[1] * len(b)
                )
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        with nvtx.range("attention_rope"):
            Q = self.positional_encoder(Q, token_positions)
            K = self.positional_encoder(K, token_positions)

        with nvtx.range("attention_causal_mask"):
            seq = torch.arange(sequence_length, device=x.device)
            qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
            kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
            causal_mask = qi >= kj

        with nvtx.range("attention_scaled_dot_product"):
            attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        with nvtx.range("attention_concat_heads"):
            concat_pattern = "batch heads seq d_v -> batch seq (heads d_v)"
            attn_output = rearrange(attn_output, concat_pattern).contiguous()

        with nvtx.range("attention_output_projection"):
            output = self.output_proj(attn_output)

        return output


def annotated_transformer_block_forward(self, x: Float[Tensor, "... seq d_model"]) -> Float[Tensor, "... seq d_model"]:
    """NVTX-annotated version of TransformerBlock.forward."""
    with nvtx.range("transformer_block"):
        with nvtx.range("attention_residual"):
            with nvtx.range("pre_attention_norm"):
                attn_input = self.ln1(x)

            attn_output = self.attn(attn_input)
            x = x + attn_output

        with nvtx.range("ffn_residual"):
            with nvtx.range("pre_ffn_norm"):
                ffn_input = self.ln2(x)

            ffn_output = self.ffn(ffn_input)
            x = x + ffn_output

        return x


def apply_enhanced_annotations(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply enhanced NVTX annotations to a model.
    This replaces key methods with annotated versions for detailed profiling.

    Args:
        model: The model to apply annotations to

    Returns:
        The same model with enhanced annotations applied
    """
    if not NVTX_AVAILABLE:
        logger.warning("NVTX not available - enhanced annotations will be no-ops")
        return model

    if not CS336_AVAILABLE:
        logger.warning("cs336_basics not available - cannot apply annotations")
        return model

    # Replace class methods
    cs336_basics.model.Linear.forward = annotated_linear_forward
    cs336_basics.model.RMSNorm.forward = annotated_rmsnorm_forward
    cs336_basics.model.SwiGLU.forward = annotated_swiglu_forward
    cs336_basics.model.RotaryEmbedding.forward = annotated_rope_forward
    cs336_basics.model.CausalMultiHeadSelfAttention.forward = annotated_attention_forward
    cs336_basics.model.TransformerBlock.forward = annotated_transformer_block_forward

    logger.info("Enhanced NVTX annotations applied to all model components")
    return model


def restore_original_methods() -> None:
    """Restore original methods (for cleanup)."""
    # Note: In practice, this would need to store original methods first
    # For now, just a placeholder since the original methods are overwritten
    logger.info("Note: Original methods have been overwritten. Restart to restore.")


if __name__ == "__main__":
    logger.info("Enhanced profiling annotations module")
    logger.info(f"NVTX available: {NVTX_AVAILABLE}")

    if NVTX_AVAILABLE:
        logger.info("✅ Ready for detailed GPU profiling")
    else:
        logger.warning("⚠️  NVTX not available - annotations will be no-ops")
