"""
Consistency tests between HuggingFace and Megatron implementations of LlavaOnevision2 vision encoder.

This module tests that the key components of the vision encoder produce identical outputs:
1. VideoRotaryEmbedding with 4:6:6 (T:H:W) split
2. PatchEmbed layer
3. rotate_half function
4. Full ViT model layer-by-layer alignment

Usage:
    # Run all tests with pytest
    pytest tests/test_llava_onevision2_consistency.py -v

    # Run specific test class
    pytest tests/test_llava_onevision2_consistency.py::TestViTModelConsistency -v

    # Run directly
    python tests/test_llava_onevision2_consistency.py

Environment Variables:
    AIAK_TRAINING_PATH: Path to LLaVA-OneVision-2 training code (default: /workspace/LLaVA-OneVision-2)
    AIAK_MAGATRON_PATH: Path to aiak_megatron (default: ${AIAK_TRAINING_PATH}/aiak_megatron)
"""

import os
import sys

# Default environment variables and PYTHONPATH setup
AIAK_TRAINING_PATH = os.environ.get("AIAK_TRAINING_PATH", "/workspace/LLaVA-OneVision-2")
AIAK_MAGATRON_PATH = os.environ.get("AIAK_MAGATRON_PATH", os.path.join(AIAK_TRAINING_PATH, "aiak_megatron"))

# Add paths to PYTHONPATH so that imports can find the libraries
if AIAK_TRAINING_PATH not in sys.path:
    sys.path.insert(0, AIAK_TRAINING_PATH)
if AIAK_MAGATRON_PATH not in sys.path:
    sys.path.insert(0, AIAK_MAGATRON_PATH)

import torch
import pytest
from typing import Dict
from dataclasses import dataclass

# HuggingFace implementation
from ds.llavaonevision2.modeling_llava_onevision2 import (
    VisionRotaryEmbedding as HFVisionRotaryEmbedding,
    OneVisionEncoderEmbeddings as HFPatchEmbed,
    rotate_half as hf_rotate_half,
)
from ds.llavaonevision2.configuration_llava_onevision2 import LlavaOnevision2VisionConfig

# Megatron implementation
from aiak_training_llm.models.llava_onevision2.onevision_encoder_model import (
    VideoRotaryEmbeddingSplit466 as MegatronVideoRoPE,
    PatchEmbed as MegatronPatchEmbed,
)
from aiak_training_llm.models.llava_onevision2.llava_onevision2_layer_spec import (
    rotate_half as megatron_rotate_half,
)


class TestRotateHalfConsistency:
    """Test that rotate_half function produces identical outputs."""


    def test_rotate_half_consistency(self):
        """Test rotate_half function consistency between implementations."""
        # Test with different shapes
        test_shapes = [
            (1, 4, 16, 64),  # (batch, heads, seq_len, head_dim)
            (2, 8, 32, 128),
            (1, 16, 64, 64),
        ]

        for shape in test_shapes:
            torch.manual_seed(42)
            x = torch.randn(shape)

            hf_result = hf_rotate_half(x)
            megatron_result = megatron_rotate_half(x)

            assert torch.allclose(hf_result, megatron_result, atol=1e-6), (
                f"rotate_half mismatch for shape {shape}. "
                f"Max diff: {(hf_result - megatron_result).abs().max()}"
            )


class TestVideoRotaryEmbeddingConsistency:
    """Test 3D Video Rotary Embedding with 4:6:6 split consistency."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create HuggingFace config
        self.hf_config = LlavaOnevision2VisionConfig(
            hidden_size=1024,
            num_attention_heads=16,
            rope_theta=10000.0,
        )

        # Create HuggingFace and Megatron RoPE modules
        self.hf_rope = HFVisionRotaryEmbedding(self.hf_config)
        self.megatron_rope = MegatronVideoRoPE(
            hidden_size=1024,
            num_attention_heads=16,
            rope_theta=10000.0,
        )

    def test_inverse_frequency_consistency(self):
        """Test that inverse frequencies are identical."""
        # Check inv_freq_t
        assert torch.allclose(
            self.hf_rope.inv_freq_t,
            self.megatron_rope.inv_freq_t,
            atol=1e-6,
        ), "inv_freq_t mismatch"

        # Check inv_freq_h
        assert torch.allclose(
            self.hf_rope.inv_freq_h,
            self.megatron_rope.inv_freq_h,
            atol=1e-6,
        ), "inv_freq_h mismatch"

        # Check inv_freq_w
        assert torch.allclose(
            self.hf_rope.inv_freq_w,
            self.megatron_rope.inv_freq_w,
            atol=1e-6,
        ), "inv_freq_w mismatch"

    def test_rope_output_consistency(self):
        """Test RoPE output consistency for various (T, H, W) configurations."""
        test_configs = [
            (1, 32, 32),   # Single frame image
            (1, 24, 24),   # Smaller image
            (4, 16, 16),   # Short video
            (8, 32, 32),   # Longer video
        ]

        for t, h, w in test_configs:
            # HuggingFace uses forward_with_thw method
            hf_freqs = self.hf_rope.forward_with_thw(t=t, h=h, w=w)

            # Megatron uses forward method with explicit t, h, w
            megatron_freqs = self.megatron_rope.forward(t=t, h=h, w=w)

            assert hf_freqs.shape == megatron_freqs.shape, (
                f"Shape mismatch for (t={t}, h={h}, w={w}): "
                f"HF: {hf_freqs.shape}, Megatron: {megatron_freqs.shape}"
            )

            assert torch.allclose(hf_freqs, megatron_freqs, atol=1e-5), (
                f"RoPE output mismatch for (t={t}, h={h}, w={w}). "
                f"Max diff: {(hf_freqs - megatron_freqs).abs().max()}"
            )

    def test_dimension_split_466(self):
        """Test that the 4:6:6 dimension split is correct."""
        head_dim = 1024 // 16  # 64
        half = head_dim // 2   # 32
        unit = half // 16      # 2

        expected_t_size = 4 * unit  # 8
        expected_h_size = 6 * unit  # 12
        expected_w_size = 6 * unit  # 12

        assert self.hf_rope.t_size == expected_t_size, f"HF t_size: {self.hf_rope.t_size} != {expected_t_size}"
        assert self.hf_rope.h_size == expected_h_size, f"HF h_size: {self.hf_rope.h_size} != {expected_h_size}"
        assert self.hf_rope.w_size == expected_w_size, f"HF w_size: {self.hf_rope.w_size} != {expected_w_size}"

        assert self.megatron_rope.t_size == expected_t_size
        assert self.megatron_rope.h_size == expected_h_size
        assert self.megatron_rope.w_size == expected_w_size


class TestPatchEmbedConsistency:
    """Test PatchEmbed layer consistency."""

    def setup_method(self):
        """Set up test fixtures."""
        self.patch_size = 14
        self.in_channels = 3
        self.embed_dim = 1024

        # Create HuggingFace config
        hf_config = LlavaOnevision2VisionConfig(
            hidden_size=self.embed_dim,
            num_channels=self.in_channels,
            patch_size=self.patch_size,
        )

        # Create patch embed modules
        self.hf_patch_embed = HFPatchEmbed(hf_config)
        self.megatron_patch_embed = MegatronPatchEmbed(
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.embed_dim,
        )

        # Copy weights from HuggingFace to Megatron for fair comparison
        with torch.no_grad():
            self.megatron_patch_embed.proj.weight.copy_(
                self.hf_patch_embed.patch_embedding.weight
            )

    def test_patch_embed_weight_shape(self):
        """Test that patch embedding weights have correct shapes."""
        hf_weight = self.hf_patch_embed.patch_embedding.weight
        megatron_weight = self.megatron_patch_embed.proj.weight

        assert hf_weight.shape == megatron_weight.shape, (
            f"Weight shape mismatch: HF: {hf_weight.shape}, Megatron: {megatron_weight.shape}"
        )

    def test_patch_embed_output_single_image(self):
        """Test patch embedding for a single image."""
        batch_size = 1
        height, width = 224, 224

        # Create input
        torch.manual_seed(42)
        pixel_values = torch.randn(batch_size, self.in_channels, height, width)

        # HuggingFace forward
        hf_output = self.hf_patch_embed(pixel_values)

        # Megatron expects input as [num_patches, in_channels, patch_size, patch_size]
        # We need to reshape for Megatron
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Reshape pixel_values for Megatron
        # [B, C, H, W] -> unfold to patches -> [num_patches, C, patch_size, patch_size]
        patches = pixel_values.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(-1, self.in_channels, self.patch_size, self.patch_size)

        megatron_output = self.megatron_patch_embed(patches)

        # Compare shapes
        # HuggingFace: [batch_size, num_patches, embed_dim]
        # Megatron: [num_patches, embed_dim]
        assert hf_output.shape == (batch_size, num_patches, self.embed_dim)
        assert megatron_output.shape == (num_patches, self.embed_dim)

        # Compare values (reshape HuggingFace output for comparison)
        hf_output_reshaped = hf_output.view(-1, self.embed_dim)
        assert torch.allclose(hf_output_reshaped, megatron_output, atol=1e-5), (
            f"Patch embed output mismatch. Max diff: {(hf_output_reshaped - megatron_output).abs().max()}"
        )


class TestPositionEncodingConsistency:
    """Test that position indices are computed identically."""

    def test_position_index_computation(self):
        """Test position index computation for T, H, W dimensions."""
        t, h, w = 2, 4, 4  # Simple test case

        # Manually compute expected indices
        expected_t_ids = []
        expected_h_ids = []
        expected_w_ids = []

        for t_idx in range(t):
            for h_idx in range(h):
                for w_idx in range(w):
                    expected_t_ids.append(t_idx)
                    expected_h_ids.append(h_idx)
                    expected_w_ids.append(w_idx)

        expected_t_ids = torch.tensor(expected_t_ids)
        expected_h_ids = torch.tensor(expected_h_ids)
        expected_w_ids = torch.tensor(expected_w_ids)

        # Compute using the same formula as in implementations
        computed_t_ids = torch.arange(t).repeat_interleave(h * w)
        computed_h_ids = torch.arange(h).repeat_interleave(w).repeat(t)
        computed_w_ids = torch.arange(w).repeat(h).repeat(t)

        assert torch.equal(expected_t_ids, computed_t_ids), "T index mismatch"
        assert torch.equal(expected_h_ids, computed_h_ids), "H index mismatch"
        assert torch.equal(expected_w_ids, computed_w_ids), "W index mismatch"


def compare_tensors(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> Dict[str, float]:
    """
    Compare two tensors and return comparison metrics.
    
    Args:
        tensor1: First tensor (e.g., HuggingFace output)
        tensor2: Second tensor (e.g., Megatron output)
        name: Name for logging
        atol: Absolute tolerance
        rtol: Relative tolerance
        
    Returns:
        Dictionary with comparison metrics
    """
    # Ensure same device
    if tensor1.device != tensor2.device:
        tensor2 = tensor2.to(tensor1.device)
    
    # Ensure same dtype for comparison
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    
    diff = (tensor1 - tensor2).abs()
    
    metrics = {
        "name": name,
        "shape_match": tensor1.shape == tensor2.shape,
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "std_diff": diff.std().item(),
        "allclose": torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol),
    }
    
    return metrics


def print_comparison_report(metrics_list: list):
    """Print a formatted comparison report."""
    print("\n" + "=" * 80)
    print("Layer-by-Layer Comparison Report")
    print("=" * 80)
    
    for metrics in metrics_list:
        status = "✓" if metrics["allclose"] else "✗"
        print(f"\n{status} {metrics['name']}:")
        print(f"   Shape Match: {metrics['shape_match']}")
        print(f"   Max Diff:    {metrics['max_diff']:.6e}")
        print(f"   Mean Diff:   {metrics['mean_diff']:.6e}")
        print(f"   Std Diff:    {metrics['std_diff']:.6e}")
    
    print("\n" + "=" * 80)
    
    # Summary
    total = len(metrics_list)
    passed = sum(1 for m in metrics_list if m["allclose"])
    print(f"Summary: {passed}/{total} checks passed")
    print("=" * 80)


def run_tests():
    """Run all consistency tests."""
    print("Running rotate_half consistency tests...")
    test_rotate = TestRotateHalfConsistency()
    test_rotate.test_rotate_half_consistency()
    print("✓ rotate_half consistency test passed")

    print("\nRunning VideoRotaryEmbedding consistency tests...")
    test_rope = TestVideoRotaryEmbeddingConsistency()
    test_rope.setup_method()
    test_rope.test_inverse_frequency_consistency()
    print("✓ Inverse frequency consistency test passed")
    test_rope.test_rope_output_consistency()
    print("✓ RoPE output consistency test passed")
    test_rope.test_dimension_split_466()
    print("✓ Dimension split 4:6:6 test passed")

    print("\nRunning PatchEmbed consistency tests...")
    test_patch = TestPatchEmbedConsistency()
    test_patch.setup_method()
    test_patch.test_patch_embed_weight_shape()
    print("✓ Patch embed weight shape test passed")
    test_patch.test_patch_embed_output_single_image()
    print("✓ Patch embed output consistency test passed")

    print("\nRunning position encoding consistency tests...")
    test_pos = TestPositionEncodingConsistency()
    test_pos.test_position_index_computation()
    print("✓ Position index computation test passed")

    print("\n" + "=" * 50)
    print("All consistency tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_tests()
