from collections.abc import Callable

import pytest
import torch
from torch._dynamo import mark_dynamic
from torch._dynamo.exc import BackendCompilerFailed
from torch.ops import aten

from torch_max_backend.testing import (
    Conf,
    check_functions_are_equivalent,
    check_outputs,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adaptive_avg_pool2d_backward_basic(device: str, dtype: torch.dtype):
    """Test _adaptive_avg_pool2d_backward basic functionality"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 8, 8
    output_height, output_width = 4, 4

    input_tensor = torch.randn(
        batch_size, channels, height, width, device=device, dtype=dtype
    )
    grad_output = torch.randn(
        batch_size, channels, output_height, output_width, device=device, dtype=dtype
    )

    check_functions_are_equivalent(fn, device, [grad_output, input_tensor])


@pytest.mark.parametrize(
    "input_size,output_size",
    [((8, 8), (4, 4)), ((10, 10), (5, 5)), ((7, 7), (3, 3)), ((16, 16), (8, 8))],
)
def test_adaptive_avg_pool2d_backward_different_sizes(
    device: str, input_size: tuple, output_size: tuple
):
    """Test _adaptive_avg_pool2d_backward with different input and output sizes"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    batch_size, channels = 2, 3
    input_height, input_width = input_size
    output_height, output_width = output_size

    input_tensor = torch.randn(
        batch_size, channels, input_height, input_width, device=device
    )
    grad_output = torch.randn(
        batch_size, channels, output_height, output_width, device=device
    )

    check_functions_are_equivalent(fn, device, [grad_output, input_tensor])


def test_adaptive_avg_pool2d_backward_3d_input(device: str):
    """Test _adaptive_avg_pool2d_backward with 3D input (no batch dimension)"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    channels, height, width = 3, 8, 8
    output_height, output_width = 4, 4

    input_tensor = torch.randn(channels, height, width, device=device)
    grad_output = torch.randn(channels, output_height, output_width, device=device)

    check_functions_are_equivalent(fn, device, [grad_output, input_tensor])


@pytest.mark.parametrize("channels", [1, 3, 16, 64])
def test_adaptive_avg_pool2d_backward_different_channels(device: str, channels: int):
    """Test _adaptive_avg_pool2d_backward with different numbers of channels"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    batch_size, height, width = 2, 8, 8
    output_height, output_width = 4, 4

    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    grad_output = torch.randn(
        batch_size, channels, output_height, output_width, device=device
    )

    check_functions_are_equivalent(fn, device, [grad_output, input_tensor])


def test_adaptive_avg_pool2d_backward_non_uniform_pooling(device: str):
    """Test _adaptive_avg_pool2d_backward with non-uniform pooling regions"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    # Input size 9x9 to output size 4x4 creates non-uniform pooling regions
    batch_size, channels = 2, 3
    input_tensor = torch.randn(batch_size, channels, 9, 9, device=device)
    grad_output = torch.randn(batch_size, channels, 4, 4, device=device)

    check_functions_are_equivalent(fn, device, [grad_output, input_tensor])


def test_adaptive_avg_pool2d_backward_output_size_one(device: str):
    """Test _adaptive_avg_pool2d_backward with output size (1, 1)"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    batch_size, channels, height, width = 2, 3, 8, 8

    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    grad_output = torch.randn(batch_size, channels, 1, 1, device=device)

    check_functions_are_equivalent(fn, device, [grad_output, input_tensor])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_adaptive_avg_pool2d_backward_half_precision(device: str, dtype: torch.dtype):
    """Test _adaptive_avg_pool2d_backward with half precision types"""

    def fn(grad_output, input_tensor):
        return aten._adaptive_avg_pool2d_backward.default(grad_output, input_tensor)

    batch_size, channels, height, width = 2, 3, 8, 8
    output_height, output_width = 4, 4

    input_tensor = torch.randn(
        batch_size, channels, height, width, device=device, dtype=dtype
    )
    grad_output = torch.randn(
        batch_size, channels, output_height, output_width, device=device, dtype=dtype
    )

    # Half precision may have lower accuracy
    check_functions_are_equivalent(
        fn, device, [grad_output, input_tensor], atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_basic(cuda_device: str, dtype: torch.dtype):
    """Test _scaled_dot_product_flash_attention basic functionality"""

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=False, return_debug_mask=False
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_functions_are_equivalent(fn, cuda_device, [q, k, v], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_with_causal(cuda_device: str, dtype: str):
    """Test _scaled_dot_product_flash_attention with causal masking"""

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q, k, v, dropout_p=0.0, is_causal=True, return_debug_mask=False
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 1, 2, 4, 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_functions_are_equivalent(fn, cuda_device, [q, k, v], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scaled_dot_product_flash_attention_with_scale(cuda_device: str, dtype):
    """Test _scaled_dot_product_flash_attention with custom scale"""

    def fn(q, k, v):
        return torch.ops.aten._scaled_dot_product_flash_attention(
            q,
            k,
            v,
            dropout_p=0.0,
            is_causal=False,
            return_debug_mask=False,
            scale=0.125,
        )[0]  # For the moment we support only training

    batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 8
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=dtype)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_functions_are_equivalent(fn, cuda_device, [q, k, v], atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_batch_norm_legit_no_training_basic(device: str, dtype: torch.dtype):
    """Test basic batch normalization inference with different dtypes"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        outputs = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return outputs[0]

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(
        batch_size, channels, height, width, dtype=dtype, device=device
    )
    weight = torch.randn(channels, dtype=dtype, device=device)
    bias = torch.randn(channels, dtype=dtype, device=device)
    running_mean = torch.randn(channels, dtype=dtype, device=device)
    running_var = torch.abs(torch.randn(channels, dtype=dtype, device=device)) + 1e-5

    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


@pytest.mark.parametrize("channels", [1, 4, 16])
def test_native_batch_norm_legit_no_training_different_channels(
    device: str, channels: int
):
    """Test batch norm with different numbers of channels"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create test tensors with varying channel dimensions
    batch_size, height, width = 2, 8, 8
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_native_batch_norm_legit_no_training_none_weight_bias(device: str):
    """Test batch norm with None weight and bias"""

    def fn(input_tensor, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, None, None, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, running_mean, running_var]
    )


@pytest.mark.parametrize("eps", [1e-5, 1e-3])
def test_native_batch_norm_legit_no_training_different_eps(device: str, eps: float):
    """Test batch norm with different epsilon values"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, eps
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create test tensors
    batch_size, channels, height, width = 2, 3, 4, 4
    input_tensor = torch.randn(batch_size, channels, height, width, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + eps * 10

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


def test_native_batch_norm_legit_no_training_2d_input(device: str):
    """Test batch norm with 2D input (N, C)"""

    def fn(input_tensor, weight, bias, running_mean, running_var):
        output = aten._native_batch_norm_legit_no_training.default(
            input_tensor, weight, bias, running_mean, running_var, 0.1, 1e-5
        )
        # We don't support returning the saved mean and variance yet.
        # It's not sure we'll ever support returning those, notably because of
        # https://github.com/pytorch/pytorch/issues/85960
        return output[0]

    # Create 2D test tensors (batch_size, channels)
    batch_size, channels = 10, 5
    input_tensor = torch.randn(batch_size, channels, device=device)
    weight = torch.randn(channels, device=device)
    bias = torch.randn(channels, device=device)
    running_mean = torch.randn(channels, device=device)
    running_var = torch.abs(torch.randn(channels, device=device)) + 1e-5

    # Test that compilation works and outputs match
    check_functions_are_equivalent(
        fn, device, [input_tensor, weight, bias, running_mean, running_var]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_acos_basic(device: str, dtype: torch.dtype):
    """Test aten.acos basic functionality with values in valid domain [-1, 1]"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.acos(x)

    # Test with values in valid domain [-1, 1]
    x = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_acos_special_values(device: str, dtype: torch.dtype):
    """Test aten.acos with special mathematical values"""
    if device == "cuda" and dtype == torch.float64:
        pytest.xfail("Bug: could not find LLVM intrinsic: 'llvm.nvvm.sqrt.approx.d'")

    def fn(x):
        return aten.acos(x)

    # Test known mathematical values
    # acos(1.0) = 0.0
    # acos(0.0) = π/2 ≈ 1.5708
    # acos(-1.0) = π ≈ 3.1416
    x = torch.tensor([1.0, 0.0, -1.0], dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_acos_2d_tensor(device: str):
    """Test aten.acos with 2D tensor"""

    def fn(x):
        return aten.acos(x)

    x = torch.tensor(
        [[-1.0, -0.5], [0.0, 0.5], [0.8, 1.0]], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


def test_aten_acos_3d_tensor(device: str):
    """Test aten.acos with 3D tensor"""

    def fn(x):
        return aten.acos(x)

    # Random values in [-1, 1] range
    x = torch.rand(2, 3, 4, dtype=torch.float32, device=device) * 2 - 1
    check_functions_are_equivalent(fn, device, [x])


def test_aten_acos_edge_domain_values(device: str):
    """Test aten.acos with values near domain boundaries"""

    def fn(x):
        return aten.acos(x)

    # Test values very close to -1 and 1
    x = torch.tensor([-0.999, -0.99, 0.99, 0.999], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_acos_single_element(device: str):
    """Test aten.acos with single element tensor"""

    def fn(x):
        return aten.acos(x)

    x = torch.tensor([0.5], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_amax_all_dims(device: str, dtype: torch.dtype):
    """Test aten_amax with default empty dim list (reduces over all dimensions)"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.amax(x)

    # Test with different shapes
    x = torch.randn(3, 4, 5, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])

    # Test with 1D tensor
    x1d = torch.randn(10, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x1d])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_not(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_not(x)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_bitwise_not_bool(device: str):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_not(x)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_and_scalar(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_and(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_and_scalar_bool(device: str, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_and(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_and(device: str, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_bool(device: str):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (5,), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting_ones(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (3, 1, 5), dtype=torch.int32)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting_ones_pad(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (8, 3, 1, 5), dtype=torch.int32)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_and_broadcasting_ones_pad_dynamic_dim(device: str):
    def fn(x, y):
        return aten.bitwise_and(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 100, (8, 3, 1, 5), dtype=torch.int32, device=device)
    mark_dynamic(x, 0)
    mark_dynamic(x, 1)
    y = torch.randint(0, 100, (1, 4, 5), dtype=torch.int32, device=device)
    mark_dynamic(y, 1)

    check_functions_are_equivalent(fn, None, [x, y])


# Tests for bitwise_or operations
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_or_scalar(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_or(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_or_scalar_bool(device: str, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_or(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_or(device: str, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_or_bool(device: str):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_or_broadcasting(device: str):
    def fn(x, y):
        return aten.bitwise_or(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


# Tests for bitwise_xor operations
@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_xor_scalar(device: str, dtype: torch.dtype):
    def fn(x):
        return aten.bitwise_xor(x, 6)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("bool_value", [True, False])
def test_aten_bitwise_xor_scalar_bool(device: str, bool_value: bool):
    dtype = torch.bool

    def fn(x):
        return aten.bitwise_xor(x, bool_value)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
)
def test_aten_bitwise_xor(device: str, dtype: torch.dtype):
    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors
    x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_xor_bool(device: str):
    dtype = torch.bool

    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors
    x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_bitwise_xor_broadcasting(device: str):
    def fn(x, y):
        return aten.bitwise_xor(x, y)

    # Create test tensors with broadcasting shapes
    x = torch.randint(0, 10, (3, 4, 5), dtype=torch.int32)
    y = torch.randint(0, 10, (4, 5), dtype=torch.int32)

    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_add.Scalar - adds scalar to each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_add.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_list(device: str, dtype: torch.dtype):
    """Test _foreach_add.List - adds corresponding tensors with alpha scaling"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_add.List(self_tensors, other_tensors, alpha=1.0)

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("alpha", [1.0, 2.0, -0.5])
def test_foreach_add_list_alpha(device: str, alpha: float):
    """Test _foreach_add.List with different alpha values"""

    def fn(x1, y1, x2, y2):
        self_tensors = [x1, y1]
        other_tensors = [x2, y2]
        return aten._foreach_add.List(self_tensors, other_tensors, alpha=alpha)

    x1 = torch.randn(3, 4, device=device)
    y1 = torch.randn(2, 5, device=device)
    x2 = torch.randn(3, 4, device=device)
    y2 = torch.randn(2, 5, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, x2, y2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_add.ScalarList - adds corresponding scalar to each tensor"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [1.5, -2.0, 3.5]
        return aten._foreach_add.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_add_tensor(device: str, dtype: torch.dtype):
    """Test _foreach_add.Tensor - broadcasts single 0-d tensor to all tensors in list"""

    def fn(x, y, z, other):
        tensors = [x, y, z]
        return aten._foreach_add.Tensor(tensors, other, alpha=1.0)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)
    other = torch.tensor(2.5, dtype=dtype, device=device)  # 0-d tensor

    check_functions_are_equivalent(fn, device, [x, y, z, other])


@pytest.mark.parametrize("alpha", [1.0, 2.0, -0.5])
def test_foreach_add_tensor_alpha(device: str, alpha: float):
    """Test _foreach_add.Tensor with different alpha values"""

    def fn(x, y, other):
        tensors = [x, y]
        return aten._foreach_add.Tensor(tensors, other, alpha=alpha)

    x = torch.randn(3, 4, device=device)
    y = torch.randn(2, 5, device=device)
    other = torch.tensor(1.5, device=device)  # 0-d tensor

    check_functions_are_equivalent(fn, device, [x, y, other])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sub_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_sub.Scalar - subtracts scalar from each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_sub.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sub_list(device: str, dtype: torch.dtype):
    """Test _foreach_sub.List - subtracts corresponding tensors with alpha scaling"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_sub.List(self_tensors, other_tensors, alpha=1.0)

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("alpha", [1.0, 2.0, -0.5])
def test_foreach_sub_list_alpha(device: str, alpha: float):
    """Test _foreach_sub.List with different alpha values"""

    def fn(x1, y1, x2, y2):
        self_tensors = [x1, y1]
        other_tensors = [x2, y2]
        return aten._foreach_sub.List(self_tensors, other_tensors, alpha=alpha)

    x1 = torch.randn(3, 4, device=device)
    y1 = torch.randn(2, 5, device=device)
    x2 = torch.randn(3, 4, device=device)
    y2 = torch.randn(2, 5, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, x2, y2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sub_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_sub.ScalarList - subtracts corresponding scalar from each tensor"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [1.5, -2.0, 3.5]
        return aten._foreach_sub.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_mul.Scalar - multiplies each tensor in list by scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_mul.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_list(device: str, dtype: torch.dtype):
    """Test _foreach_mul.List - multiplies corresponding tensors"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_mul.List(self_tensors, other_tensors)

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_mul.ScalarList - multiplies each tensor by corresponding scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [1.5, -2.0, 3.5]
        return aten._foreach_mul.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_mul_tensor(device: str, dtype: torch.dtype):
    """Test _foreach_mul.Tensor - broadcasts single 0-d tensor to all tensors in list"""

    def fn(x, y, z, other):
        tensors = [x, y, z]
        return aten._foreach_mul.Tensor(tensors, other)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)
    other = torch.tensor(2.5, dtype=dtype, device=device)  # 0-d tensor

    check_functions_are_equivalent(fn, device, [x, y, z, other])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_pow.Scalar - raises each tensor in list to scalar power"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_pow.Scalar(tensors, 2.0)

    x = torch.randn(3, 4, dtype=dtype, device=device).abs() + 0.1
    y = torch.randn(2, 5, dtype=dtype, device=device).abs() + 0.1
    z = torch.randn(4, dtype=dtype, device=device).abs() + 0.1

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_list(device: str, dtype: torch.dtype):
    """Test _foreach_pow.List - raises corresponding tensors to powers"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        exponent_tensors = [x2, y2, z2]
        return aten._foreach_pow.List(self_tensors, exponent_tensors)

    x1 = torch.randn(3, 4, dtype=dtype, device=device).abs() + 0.1
    y1 = torch.randn(2, 5, dtype=dtype, device=device).abs() + 0.1
    z1 = torch.randn(4, dtype=dtype, device=device).abs() + 0.1
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_pow.ScalarList - raises each tensor to corresponding scalar power"""

    def fn(x, y, z):
        tensors = [x, y, z]
        exponents = [2.0, 3.0, 0.5]
        return aten._foreach_pow.ScalarList(tensors, exponents)

    x = torch.randn(3, 4, dtype=dtype, device=device).abs() + 0.1
    y = torch.randn(2, 5, dtype=dtype, device=device).abs() + 0.1
    z = torch.randn(4, dtype=dtype, device=device).abs() + 0.1

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_pow_scalarandtensor(device: str, dtype: torch.dtype):
    """Test _foreach_pow.ScalarAndTensor - raises scalar to tensor powers"""

    def fn(x, y, z):
        exponent_tensors = [x, y, z]
        return aten._foreach_pow.ScalarAndTensor(2.0, exponent_tensors)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_div.Scalar - divides each tensor in list by scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_div.Scalar(tensors, 2.5)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_list(device: str, dtype: torch.dtype):
    """Test _foreach_div.List - divides corresponding tensors"""

    def fn(x1, y1, z1, x2, y2, z2):
        self_tensors = [x1, y1, z1]
        other_tensors = [x2, y2, z2]
        return aten._foreach_div.List(self_tensors, other_tensors)

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device) + 0.1
    y2 = torch.randn(2, 5, dtype=dtype, device=device) + 0.1
    z2 = torch.randn(4, dtype=dtype, device=device) + 0.1

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_div.ScalarList - divides each tensor by corresponding scalar"""

    def fn(x, y, z):
        tensors = [x, y, z]
        scalars = [2.0, 3.0, 1.5]
        return aten._foreach_div.ScalarList(tensors, scalars)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_div_tensor(device: str, dtype: torch.dtype):
    """Test _foreach_div.Tensor - broadcasts single 0-d tensor to all tensors in list"""

    def fn(x, y, z, other):
        tensors = [x, y, z]
        return aten._foreach_div.Tensor(tensors, other)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)
    other = torch.tensor(2.5, dtype=dtype, device=device)  # 0-d tensor

    check_functions_are_equivalent(fn, device, [x, y, z, other])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_sqrt(device: str, dtype: torch.dtype):
    """Test _foreach_sqrt - computes square root of each tensor in list"""
    # xfail for float64 on CUDA due to current MAX limitation with sqrt intrinsic
    if device == "cuda" and dtype == torch.float64:
        pytest.xfail(
            "float64 sqrt on CUDA currently fails in MAX (llvm.nvvm.sqrt.approx.d intrinsic issue)"
        )

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_sqrt(tensors)

    x = torch.randn(3, 4, dtype=dtype, device=device).abs() + 0.1
    y = torch.randn(2, 5, dtype=dtype, device=device).abs() + 0.1
    z = torch.randn(4, dtype=dtype, device=device).abs() + 0.1

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_neg(device: str, dtype: torch.dtype):
    """Test _foreach_neg - computes negation of each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_neg(tensors)

    x = torch.randn(3, 4, dtype=dtype, device=device)
    y = torch.randn(2, 5, dtype=dtype, device=device)
    z = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_reciprocal(device: str, dtype: torch.dtype):
    """Test _foreach_reciprocal - computes reciprocal (1/x) of each tensor in list"""

    def fn(x, y, z):
        tensors = [x, y, z]
        return aten._foreach_reciprocal(tensors)

    # Add small offset to avoid division by zero
    x = torch.randn(3, 4, dtype=dtype, device=device) + 0.5
    y = torch.randn(2, 5, dtype=dtype, device=device) + 0.5
    z = torch.randn(4, dtype=dtype, device=device) + 0.5

    check_functions_are_equivalent(fn, device, [x, y, z])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcmul_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_addcmul.Scalar - adds element-wise product scaled by scalar"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        return aten._foreach_addcmul.Scalar(
            self_tensors, tensor1_list, tensor2_list, 2.0
        )

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)
    x3 = torch.randn(3, 4, dtype=dtype, device=device)
    y3 = torch.randn(2, 5, dtype=dtype, device=device)
    z3 = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcmul_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_addcmul.ScalarList - adds element-wise products scaled by corresponding scalars"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        scalars = [1.0, 2.0, 0.5]
        return aten._foreach_addcmul.ScalarList(
            self_tensors, tensor1_list, tensor2_list, scalars
        )

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)
    x3 = torch.randn(3, 4, dtype=dtype, device=device)
    y3 = torch.randn(2, 5, dtype=dtype, device=device)
    z3 = torch.randn(4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


# NOTE: _foreach_addcmul.Tensor is NOT tested
# The .Tensor variant requires a 1-D CPU tensor with concrete values to extract scalars,
# which is incompatible with torch.compile's meta tensor tracing.
# See: https://github.com/pytorch/pytorch/issues/139795


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcdiv_scalar(device: str, dtype: torch.dtype):
    """Test _foreach_addcdiv.Scalar - adds element-wise quotient scaled by scalar"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        return aten._foreach_addcdiv.Scalar(
            self_tensors, tensor1_list, tensor2_list, value=2.0
        )

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)
    x3 = torch.randn(3, 4, dtype=dtype, device=device) + 0.5
    y3 = torch.randn(2, 5, dtype=dtype, device=device) + 0.5
    z3 = torch.randn(4, dtype=dtype, device=device) + 0.5

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_foreach_addcdiv_scalarlist(device: str, dtype: torch.dtype):
    """Test _foreach_addcdiv.ScalarList - adds element-wise quotient scaled by scalar list"""

    def fn(x1, y1, z1, x2, y2, z2, x3, y3, z3):
        self_tensors = [x1, y1, z1]
        tensor1_list = [x2, y2, z2]
        tensor2_list = [x3, y3, z3]
        scalars = [1.0, 2.0, 0.5]
        return aten._foreach_addcdiv.ScalarList(
            self_tensors, tensor1_list, tensor2_list, scalars
        )

    x1 = torch.randn(3, 4, dtype=dtype, device=device)
    y1 = torch.randn(2, 5, dtype=dtype, device=device)
    z1 = torch.randn(4, dtype=dtype, device=device)
    x2 = torch.randn(3, 4, dtype=dtype, device=device)
    y2 = torch.randn(2, 5, dtype=dtype, device=device)
    z2 = torch.randn(4, dtype=dtype, device=device)
    x3 = torch.randn(3, 4, dtype=dtype, device=device) + 0.5
    y3 = torch.randn(2, 5, dtype=dtype, device=device) + 0.5
    z3 = torch.randn(4, dtype=dtype, device=device) + 0.5

    check_functions_are_equivalent(fn, device, [x1, y1, z1, x2, y2, z2, x3, y3, z3])


# NOTE: _foreach_addcdiv.Tensor is NOT tested
# The .Tensor variant requires a 1-D CPU tensor with concrete values to extract scalars,
# which is incompatible with torch.compile's meta tensor tracing.
# See: https://github.com/pytorch/pytorch/issues/139795


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_ceil_basic(device: str, dtype: torch.dtype):
    """Test aten.ceil basic functionality with floating point numbers"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.ceil(x)

    # Test with positive and negative floating point values
    x = torch.tensor(
        [-2.7, -1.5, -1.0, -0.3, 0.0, 0.3, 1.0, 1.5, 2.7], dtype=dtype, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_aten_ceil_integer_types(device: str, dtype: torch.dtype):
    """Test aten.ceil with integer types (should return copy)"""

    def fn(x):
        return aten.ceil(x)

    # Integer types should return a copy with no change
    x = torch.tensor([-5, -1, 0, 1, 5], dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_2d_tensor(device: str):
    """Test aten.ceil with 2D tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.tensor(
        [[-2.7, -1.3], [0.5, 1.8], [2.1, 3.9]], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_3d_tensor(device: str):
    """Test aten.ceil with 3D tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_edge_cases(device: str):
    """Test aten.ceil with edge cases"""

    def fn(x):
        return aten.ceil(x)

    # Test with already integer values, zero, and boundary cases
    x = torch.tensor(
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_large_values(device: str):
    """Test aten.ceil with large floating point values"""

    def fn(x):
        return aten.ceil(x)

    # Test with large positive and negative values
    x = torch.tensor(
        [-1000.1, -100.9, 100.1, 1000.9], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_small_fractional_values(device: str):
    """Test aten.ceil with small fractional values"""

    def fn(x):
        return aten.ceil(x)

    # Test with small positive and negative fractional values
    x = torch.tensor(
        [-0.001, -0.5, -0.999, 0.001, 0.5, 0.999], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_single_element(device: str):
    """Test aten.ceil with single element tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.tensor([2.3], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_ceil_scalar_tensor(device: str):
    """Test aten.ceil with scalar tensor"""

    def fn(x):
        return aten.ceil(x)

    x = torch.tensor(2.7, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_aten_gelu_backward_basic(conf: Conf, dtype: torch.dtype, approximate: str):
    """Test aten.gelu_backward with different approximations"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate=approximate)

    # Test with varied input values covering negative, zero, and positive ranges
    x = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0], dtype=dtype)
    grad_output = torch.ones_like(x)

    # bfloat16 requires higher tolerance due to lower precision
    if dtype == torch.bfloat16:
        check_outputs(fn, conf, [grad_output, x], atol=1e-2, rtol=5e-2)
    else:
        check_outputs(fn, conf, [grad_output, x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_gelu_backward_2d_tensor(conf: Conf, dtype: torch.dtype):
    """Test aten.gelu_backward with 2D tensor"""
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.randn(3, 4, dtype=dtype)
    grad_output = torch.randn(3, 4, dtype=dtype)

    # bfloat16 requires higher tolerance due to lower precision
    if dtype == torch.bfloat16:
        check_outputs(fn, conf, [grad_output, x], atol=1e-2, rtol=5e-2)
    else:
        check_outputs(fn, conf, [grad_output, x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_gelu_backward_3d_tensor(conf: Conf, dtype: torch.dtype):
    """Test aten.gelu_backward with 3D tensor"""
    if conf.device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.randn(2, 3, 4, dtype=dtype)
    grad_output = torch.randn(2, 3, 4, dtype=dtype)

    # bfloat16 requires higher tolerance due to lower precision
    if dtype == torch.bfloat16:
        check_outputs(fn, conf, [grad_output, x], atol=1e-2, rtol=5e-2)
    else:
        check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_tanh_approx(conf: Conf):
    """Test aten.gelu_backward with tanh approximation"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="tanh")

    x = torch.randn(10, dtype=torch.float32)
    grad_output = torch.ones_like(x)
    check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_edge_values(conf: Conf):
    """Test aten.gelu_backward with edge case values"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    # Test with large values, small values, and exact zeros
    x = torch.tensor([-10.0, -5.0, -1e-6, 0.0, 1e-6, 5.0, 10.0], dtype=torch.float32)
    grad_output = torch.ones_like(x)
    check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_different_grad_outputs(conf: Conf):
    """Test aten.gelu_backward with non-uniform grad_output"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.randn(8, dtype=torch.float32)
    # Test with varied gradient values
    grad_output = torch.tensor(
        [0.1, 0.5, 1.0, 2.0, -0.5, -1.0, 0.0, 3.0], dtype=torch.float32
    )
    check_outputs(fn, conf, [grad_output, x])


def test_aten_gelu_backward_scalar_tensor(conf: Conf):
    """Test aten.gelu_backward with scalar tensor"""

    def fn(grad_output, x):
        return aten.gelu_backward(grad_output, x, approximate="none")

    x = torch.tensor(1.5, dtype=torch.float32)
    grad_output = torch.tensor(1.0, dtype=torch.float32)
    check_outputs(fn, conf, [grad_output, x])


TRIGON_FUNCTIONS = [aten.asinh, aten.cosh, aten.sinh, aten.tanh]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_basic(device: str, fn: Callable, dtype: torch.dtype):
    """Test trigonometric functions basic functionality with floating point numbers"""

    if device == "cuda" and dtype == torch.float64 and fn == aten.asinh:
        pytest.xfail("could not find LLVM intrinsic: 'llvm.nvvm.sqrt.approx.d'")

    # Test with positive, negative, and zero values
    # cosh(0) = 1, cosh is even function: cosh(-x) = cosh(x)
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_2d_tensor(device: str, fn: Callable):
    """Test trigonometric functions with 2D tensor"""

    x = torch.tensor(
        [[-1.5, -0.5], [0.0, 1.0], [1.5, 2.5]], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_3d_tensor(device: str, fn: Callable):
    """Test trigonometric functions with 3D tensor"""

    x = torch.randn(2, 3, 4, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_large_values(device: str, fn: Callable):
    """Test trigonometric functions with large values (may approach infinity)"""

    # large values will produce large results due to exponential growth
    x = torch.tensor([-5.0, -3.0, 3.0, 5.0], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_small_values(device: str, fn: Callable):
    """Test trigonometric functions with small values near zero"""

    # for small x, cosh(x) ≈ 1 + x²/2
    x = torch.tensor(
        [-0.1, -0.01, -0.001, 0.001, 0.01, 0.1], dtype=torch.float32, device=device
    )
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_single_element(device: str, fn: Callable):
    """Test trigonometric functions with single element tensor"""

    x = torch.tensor([1.5], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("fn", TRIGON_FUNCTIONS)
def test_aten_trigon_scalar_tensor(device: str, fn: Callable):
    """Test trigonometric functions with scalar tensor"""

    x = torch.tensor(1.0, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_select_scatter_basic_2d(device: str, dtype: torch.dtype):
    """Test aten.select_scatter basic functionality with 2D tensors"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=0, index=1)

    # Replace row 1 with new values
    self = torch.zeros(3, 4, dtype=dtype, device=device)
    src = torch.ones(4, dtype=dtype, device=device) * 5

    check_functions_are_equivalent(fn, device, [self, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_aten_select_scatter_dim1(device: str, dtype: torch.dtype):
    """Test aten.select_scatter along dimension 1"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=1, index=2)

    # Replace column 2 with new values
    self = torch.zeros(3, 4, dtype=dtype, device=device)
    src = torch.ones(3, dtype=dtype, device=device) * 7

    check_functions_are_equivalent(fn, device, [self, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_select_scatter_3d_tensor(device: str, dtype: torch.dtype):
    """Test aten.select_scatter with 3D tensor"""
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(self, src):
        return aten.select_scatter(self, src, dim=1, index=2)

    # Replace middle slice along dimension 1
    self = torch.zeros(2, 4, 3, dtype=dtype, device=device)
    src = torch.randn(2, 3, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [self, src])


def test_aten_select_scatter_negative_dim(device: str):
    """Test aten.select_scatter with negative dimension"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=-1, index=1)

    # Negative dimension indexing (dim=-1 is last dimension)
    self = torch.zeros(3, 4, dtype=torch.float32, device=device)
    src = torch.ones(3, dtype=torch.float32, device=device) * 2

    check_functions_are_equivalent(fn, device, [self, src])


def test_aten_select_scatter_negative_index(device: str):
    """Test aten.select_scatter with negative index"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=0, index=-1)

    # Negative index (index=-1 is last index)
    self = torch.zeros(3, 4, dtype=torch.float32, device=device)
    src = torch.ones(4, dtype=torch.float32, device=device) * 3

    check_functions_are_equivalent(fn, device, [self, src])


def test_aten_select_scatter_scalar_src(device: str):
    """Test aten.select_scatter with scalar src (1D self)"""

    def fn(self, src):
        return aten.select_scatter(self, src, dim=0, index=1)

    # When self is 1D, src is a scalar (0D tensor)
    self = torch.zeros(3, dtype=torch.float32, device=device)
    src = torch.tensor(5.0, dtype=torch.float32, device=device)

    check_functions_are_equivalent(fn, device, [self, src])


@pytest.mark.parametrize("repeats", [1, 2, 3, 5])
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_aten_repeat_interleave_basic(device: str, repeats: int, dim: int):
    """Test aten.repeat_interleave with basic parameters"""

    def fn(x):
        return aten.repeat_interleave(x, repeats, dim)

    x = torch.randn(3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_repeat_interleave_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.repeat_interleave with different data types"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 4), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (3, 4), dtype=dtype, device=device)
    else:
        x = torch.randn(3, 4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_1d(device: str):
    """Test aten.repeat_interleave with 1D tensor"""

    def fn(x):
        return aten.repeat_interleave(x, 3, 0)

    x = torch.randn(5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_3d(device: str):
    """Test aten.repeat_interleave with 3D tensor"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 1)

    x = torch.randn(2, 3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(1, 5), (5, 1), (1, 1)])
def test_aten_repeat_interleave_edge_cases(device: str, shape: tuple):
    """Test aten.repeat_interleave with edge case shapes"""

    def fn(x):
        return aten.repeat_interleave(x, 2, 0)

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_repeat_interleave_large_repeats(device: str):
    """Test aten.repeat_interleave with large repeat count"""

    def fn(x):
        return aten.repeat_interleave(x, 10, 0)

    x = torch.randn(2, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_aten_scatter_src_basic_2d(device: str, dtype: torch.dtype):
    """Test aten.scatter.src basic functionality with 2D tensors"""

    def fn(self, index, src):
        return aten.scatter.src(self, dim=1, index=index, src=src)

    # Basic 2D scatter along dim=1
    self = torch.zeros(3, 5, dtype=dtype, device=device)
    src = torch.ones(3, 2, dtype=dtype, device=device)
    index = torch.tensor([[0, 2], [1, 4], [3, 0]], dtype=torch.long, device=device)

    check_functions_are_equivalent(fn, device, [self, index, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.int64])
def test_aten_scatter_src_dim0(device: str, dtype: torch.dtype):
    """Test aten.scatter.src along dimension 0"""

    def fn(self, index, src):
        return aten.scatter.src(self, dim=0, index=index, src=src)

    # Scatter along dim=0
    self = torch.zeros(4, 4, dtype=dtype, device=device)
    src = torch.ones(2, 4, dtype=dtype, device=device) * 5
    index = torch.tensor([[1, 0, 2, 3], [2, 1, 3, 0]], dtype=torch.long, device=device)

    check_functions_are_equivalent(fn, device, [self, index, src])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_scatter_src_3d_tensor(device: str, dtype: torch.dtype):
    """Test aten.scatter.src with 3D tensor"""
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(self, index, src):
        return aten.scatter.src(self, dim=1, index=index, src=src)

    # 3D scatter along middle dimension
    self = torch.zeros(2, 4, 3, dtype=dtype, device=device)
    src = torch.randn(2, 2, 3, dtype=dtype, device=device)
    index = torch.tensor(
        [[[0, 2, 1], [3, 1, 2]], [[2, 0, 3], [1, 3, 0]]],
        dtype=torch.long,
        device=device,
    )

    check_functions_are_equivalent(fn, device, [self, index, src])


def test_aten_scatter_src_negative_dim(device: str):
    """Test aten.scatter.src with negative dimension"""

    def fn(self, index, src):
        return aten.scatter.src(self, dim=-1, index=index, src=src)

    # Negative dimension indexing (dim=-1 is last dimension)
    self = torch.zeros(3, 4, dtype=torch.float32, device=device)
    src = torch.ones(3, 2, dtype=torch.float32, device=device) * 2
    index = torch.tensor([[0, 3], [1, 2], [2, 1]], dtype=torch.long, device=device)

    check_functions_are_equivalent(fn, device, [self, index, src])


def test_aten_scatter_value_basic(device: str):
    """Test aten.scatter.value with scalar value - basic functionality"""

    def fn(x, index):
        return aten.scatter.value(x, 0, index, 1.0)

    # Create base tensor of zeros
    x = torch.zeros(3, 5, device=device)
    # Indices where we want to write along dimension 0
    index = torch.tensor(
        [[0, 1, 2, 0, 0], [2, 0, 0, 1, 2]], dtype=torch.long, device=device
    )
    check_functions_are_equivalent(fn, device, [x, index])


def test_aten_scatter_value_diagonal(device: str):
    """Test aten.scatter.value with scalar value - create diagonal pattern"""

    def fn(x, index):
        return aten.scatter.value(x, 0, index, 5.0)

    x = torch.zeros(3, 3, device=device)
    index = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    check_functions_are_equivalent(fn, device, [x, index])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_aten_scatter_value_dtypes(device: str, dtype: torch.dtype):
    """Test aten.scatter.value with scalar value - different dtypes"""

    def fn(x, index):
        if dtype in [torch.int32, torch.int64]:
            return aten.scatter.value(x, 0, index, 7)
        else:
            return aten.scatter.value(x, 0, index, 3.5)

    if dtype in [torch.int32, torch.int64]:
        x = torch.zeros(4, 4, dtype=dtype, device=device)
    else:
        x = torch.zeros(4, 4, dtype=dtype, device=device)

    index = torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)
    check_functions_are_equivalent(fn, device, [x, index])


def test_aten_scatter_value_dim1(device: str):
    """Test aten.scatter.value with scalar value along dimension 1"""

    def fn(x, index):
        return aten.scatter.value(x, 1, index, 2.5)

    x = torch.zeros(3, 5, device=device)
    index = torch.tensor(
        [[0, 2, 4], [1, 3, 4], [0, 1, 2]], dtype=torch.long, device=device
    )
    check_functions_are_equivalent(fn, device, [x, index])


def test_aten_scatter_value_3d(device: str):
    """Test aten.scatter.value with scalar value on 3D tensor"""

    def fn(x, index):
        return aten.scatter.value(x, 1, index, 9.0)

    x = torch.zeros(2, 3, 4, device=device)
    index = torch.tensor(
        [[[0, 1, 2, 0], [1, 2, 0, 1]], [[2, 0, 1, 2], [0, 1, 2, 1]]],
        dtype=torch.long,
        device=device,
    )
    check_functions_are_equivalent(fn, device, [x, index])


def test_aten_split_with_sizes_basic(device: str):
    """Test aten.split_with_sizes with basic splits"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 3, 5], dim=0)

    x = torch.randn(10, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_aten_split_with_sizes_different_dims(device: str, dim: int):
    """Test aten.split_with_sizes on different dimensions"""

    def fn(x):
        # Adjust split sizes based on dimension
        if dim == 0:
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)  # sum=4, dim size=4
        elif dim == 1:
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)  # sum=4, dim size=4
        else:  # dim == 2
            return aten.split_with_sizes(x, [2, 2, 1], dim=dim)  # sum=5, dim size=5

    x = torch.randn(4, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [-1, -2, -3])
def test_aten_split_with_sizes_negative_dims(device: str, dim: int):
    """Test aten.split_with_sizes with negative dimension indices"""

    def fn(x):
        # Adjust split sizes based on dimension
        if dim == -1:  # last dim, size=5
            return aten.split_with_sizes(x, [2, 2, 1], dim=dim)
        elif dim == -2:  # middle dim, size=4
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)
        else:  # dim == -3, first dim, size=4
            return aten.split_with_sizes(x, [1, 2, 1], dim=dim)

    x = torch.randn(4, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_uneven(device: str):
    """Test aten.split_with_sizes with uneven splits"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 3, 2, 4], dim=1)

    x = torch.randn(3, 10, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_single_split(device: str):
    """Test aten.split_with_sizes with single split (entire tensor)"""

    def fn(x):
        return aten.split_with_sizes(x, [5], dim=0)

    x = torch.randn(5, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_split_with_sizes_dtypes(device: str, dtype: torch.dtype):
    """Test aten.split_with_sizes with different data types"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 2, 2], dim=0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (6, 4), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (6, 4), dtype=dtype, device=device)
    else:
        x = torch.randn(6, 4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_1d(device: str):
    """Test aten.split_with_sizes with 1D tensor"""

    def fn(x):
        return aten.split_with_sizes(x, [3, 3, 4], dim=0)

    x = torch.randn(10, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_3d(device: str):
    """Test aten.split_with_sizes with 3D tensor"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 1, 2], dim=2)

    x = torch.randn(2, 3, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_many_splits(device: str):
    """Test aten.split_with_sizes with many small splits"""

    def fn(x):
        return aten.split_with_sizes(x, [1, 1, 1, 1, 1, 1, 1, 1], dim=0)

    x = torch.randn(8, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("split_sizes", [[2, 0, 3], [0, 5, 0], [0, 0, 5]])
def test_aten_split_with_sizes_zero_size(device: str, split_sizes: list[int]):
    """Test aten.split_with_sizes with zero-sized splits"""

    def fn(x):
        return aten.split_with_sizes(x, split_sizes, dim=0)

    x = torch.randn(5, 3, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_split_with_sizes_exact_split(device: str):
    """Test aten.split_with_sizes where sizes exactly match dimension"""

    def fn(x):
        return aten.split_with_sizes(x, [2, 2, 2, 2, 2], dim=1)

    x = torch.randn(3, 10, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_single_dim(device: str):
    """Test aten.squeeze with single dimension"""

    def fn(x):
        return aten.squeeze(x, 1)

    x = torch.randn(3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_aten_squeeze_different_dims(device: str, dim: int):
    """Test aten.squeeze on different dimensions"""

    def fn(x):
        return aten.squeeze(x, dim)

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_negative_dim(device: str):
    """Test aten.squeeze with negative dimension"""

    def fn(x):
        return aten.squeeze(x, -2)

    x = torch.randn(3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_multiple_dims(device: str):
    """Test aten.squeeze with multiple dimensions"""

    def fn(x):
        return aten.squeeze(x, [0, 2])

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_no_change(device: str):
    """Test aten.squeeze when dimension is not size 1"""

    def fn(x):
        return aten.squeeze(x, 1)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_aten_squeeze_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.squeeze with different data types"""

    def fn(x):
        return aten.squeeze(x, 1)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (3, 1, 5), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (3, 1, 5), dtype=dtype, device=device)
    else:
        x = torch.randn(3, 1, 5, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_all_ones(device: str):
    """Test aten.squeeze with tensor of all size-1 dimensions"""

    def fn(x):
        return aten.squeeze(x, [0, 1, 2, 3])

    x = torch.randn(1, 1, 1, 1, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_2d(device: str):
    """Test aten.squeeze with 2D tensor"""

    def fn(x):
        return aten.squeeze(x, 0)

    x = torch.randn(1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_5d(device: str):
    """Test aten.squeeze with 5D tensor"""

    def fn(x):
        return aten.squeeze(x, [1, 3])

    x = torch.randn(2, 1, 3, 1, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_squeeze_empty_dims(device: str):
    """Test aten.squeeze with empty dimensions list"""

    def fn(x):
        return aten.squeeze(x, [])

    x = torch.randn(1, 3, 1, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(1,), (1, 1), (1, 1, 1)])
def test_aten_squeeze_edge_cases(device: str, shape: tuple):
    """Test aten.squeeze with edge case shapes"""

    def fn(x):
        return aten.squeeze(x, list(range(len(shape))))

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_basic(device: str):
    """Test aten.triu with default diagonal=0"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(5, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
def test_aten_triu_different_diagonals(device: str, diagonal: int):
    """Test aten.triu with different diagonal values"""

    def fn(x):
        return aten.triu(x, diagonal=diagonal)

    x = torch.randn(6, 6, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("shape", [(3, 5), (5, 3), (7, 7)])
def test_aten_triu_rectangular(device: str, shape: tuple):
    """Test aten.triu with rectangular matrices"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(*shape, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.bool]
)
def test_aten_triu_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.triu with different data types"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    if dtype == torch.bool:
        x = torch.randint(0, 2, (4, 4), dtype=dtype, device=device)
    elif dtype == torch.int32:
        x = torch.randint(0, 10, (4, 4), dtype=dtype, device=device)
    else:
        x = torch.randn(4, 4, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_3d(device: str):
    """Test aten.triu with 3D tensor (batch of matrices)"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(3, 4, 4, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("diagonal", [-1, 0, 1])
def test_aten_triu_3d_different_diagonals(device: str, diagonal: int):
    """Test aten.triu with 3D tensor and different diagonals"""

    def fn(x):
        return aten.triu(x, diagonal=diagonal)

    x = torch.randn(2, 5, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_large_diagonal(device: str):
    """Test aten.triu with diagonal larger than matrix size"""

    def fn(x):
        return aten.triu(x, diagonal=10)

    x = torch.randn(5, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_negative_large_diagonal(device: str):
    """Test aten.triu with large negative diagonal"""

    def fn(x):
        return aten.triu(x, diagonal=-10)

    x = torch.randn(5, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_small_matrix(device: str):
    """Test aten.triu with small matrices"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(2, 2, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_single_element(device: str):
    """Test aten.triu with 1x1 matrix"""

    def fn(x):
        return aten.triu(x, diagonal=0)

    x = torch.randn(1, 1, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("diagonal", [10, -10])
def test_aten_triu_dynamic_dimensions_large_diagonal(device: str, diagonal: int):
    """Test aten.triu with dynamic dimensions and large diagonal"""

    def fn(x):
        return aten.triu(x, diagonal=diagonal)

    x = torch.randn(5, 7, device=device)
    # Mark both dimensions as dynamic
    mark_dynamic(x, 0)
    mark_dynamic(x, 1)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_triu_dynamic_batch_dimension(device: str):
    """Test aten.triu with dynamic batch dimension"""

    def fn(x):
        return aten.triu(x, diagonal=1)

    x = torch.randn(3, 4, 4, device=device)
    # Mark only the batch dimension as dynamic
    mark_dynamic(x, 0)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_logical_and_bool_tensors(device: str):
    """Test aten.logical_and with boolean tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([True, False, True, False], device=device)
    y = torch.tensor([True, True, False, False], device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_numeric_tensors(device: str):
    """Test aten.logical_and with numeric tensors (converted to bool)"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([1, 0, 2, -1], dtype=torch.float32, device=device)
    y = torch.tensor([3, 0, 0, 4], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize(
    "dtype", [torch.int32, torch.int64, torch.float32, torch.float64]
)
def test_aten_logical_and_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten.logical_and with different numeric data types"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([1, 0, 2, -1], dtype=dtype, device=device)
    y = torch.tensor([3, 0, 0, 4], dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_2d_tensors(device: str):
    """Test aten.logical_and with 2D tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    y = torch.randint(0, 2, (3, 4), dtype=torch.bool, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_broadcasting(device: str):
    """Test aten.logical_and with broadcasting"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 3, (3, 4), dtype=torch.int32, device=device)
    y = torch.randint(0, 3, (4,), dtype=torch.int32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_mixed_types(device: str):
    """Test aten.logical_and with mixed boolean and numeric types"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([True, False, True, False], device=device)
    y = torch.tensor([1, 0, 2, 0], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_zeros_and_ones(device: str):
    """Test aten.logical_and with patterns of zeros and ones"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([0, 1, 0, 1], dtype=torch.float32, device=device)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_all_false(device: str):
    """Test aten.logical_and with all false values"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.zeros(4, dtype=torch.float32, device=device)
    y = torch.zeros(4, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_all_true(device: str):
    """Test aten.logical_and with all true values"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.ones(4, dtype=torch.float32, device=device)
    y = torch.ones(4, dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_negative_values(device: str):
    """Test aten.logical_and with negative values (should be treated as true)"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor([-1, -2, 0, 3], dtype=torch.float32, device=device)
    y = torch.tensor([4, 0, -5, 6], dtype=torch.float32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_3d_tensors(device: str):
    """Test aten.logical_and with 3D tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    y = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


def test_aten_logical_and_scalar_like(device: str):
    """Test aten.logical_and with scalar-like tensors"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x = torch.tensor(True, device=device)
    y = torch.tensor(False, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize(
    "shape_pair", [((3, 1), (3, 4)), ((1, 4), (3, 4)), ((3, 1, 1), (3, 4, 5))]
)
def test_aten_logical_and_broadcasting_shapes(device: str, shape_pair: tuple):
    """Test aten.logical_and with various broadcasting shapes"""

    def fn(x, y):
        return aten.logical_and(x, y)

    x_shape, y_shape = shape_pair
    x = torch.randint(0, 2, x_shape, dtype=torch.int32, device=device)
    y = torch.randint(0, 2, y_shape, dtype=torch.int32, device=device)
    check_functions_are_equivalent(fn, device, [x, y])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amax_single_dim(device: str, dim: int, keepdim: bool):
    """Test aten_amax with single dimension"""

    def fn(x):
        return aten.amax(x, dim=[dim], keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amax_multiple_dims(device: str, dims: list[int], keepdim: bool):
    """Test aten_amax with multiple dimensions"""

    def fn(x):
        return aten.amax(x, dim=dims, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_max_no_dim(device: str):
    """Test aten_max without dimension (returns single value)"""

    def fn(x):
        return aten.max(x)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_max_with_dim(device: str, dim: int, keepdim: bool):
    """Test aten_max with dimension (returns values and indices tuple)"""

    def fn(x):
        return aten.max(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_aten_max_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten_max with different data types"""

    def fn(x):
        return aten.max(x, dim=1, keepdim=False)

    if dtype.is_floating_point:
        x = torch.randn(3, 4, dtype=dtype, device=device)
    else:
        x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_aten_amin_all_dims(device: str, dtype: torch.dtype):
    """Test aten_amin with default empty dim list (reduces over all dimensions)"""
    # Skip float16 on CPU as MAX doesn't support f16 on CPU
    if device == "cpu" and dtype == torch.float16:
        pytest.skip("float16 not supported on CPU in MAX")

    def fn(x):
        return aten.amin(x)

    # Test with different shapes
    x = torch.randn(3, 4, 5, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x])

    # Test with 1D tensor
    x1d = torch.randn(10, dtype=dtype, device=device)
    check_functions_are_equivalent(fn, device, [x1d])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amin_single_dim(device: str, dim: int, keepdim: bool):
    """Test aten_amin with single dimension"""

    def fn(x):
        return aten.amin(x, dim=[dim], keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dims", [[0, 1], [1, 2], [0, 2]])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_amin_multiple_dims(device: str, dims: list[int], keepdim: bool):
    """Test aten_amin with multiple dimensions"""

    def fn(x):
        return aten.amin(x, dim=dims, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


def test_aten_min_no_dim(device: str):
    """Test aten_min without dimension (returns single value)"""

    def fn(x):
        return aten.min(x)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("keepdim", [True, False])
def test_aten_min_with_dim(device: str, dim: int, keepdim: bool):
    """Test aten_min with dimension (returns values and indices tuple)"""

    def fn(x):
        return aten.min(x, dim=dim, keepdim=keepdim)

    x = torch.randn(3, 4, 5, device=device)
    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_aten_min_different_dtypes(device: str, dtype: torch.dtype):
    """Test aten_min with different data types"""

    def fn(x):
        return aten.min(x, dim=1, keepdim=False)

    if dtype.is_floating_point:
        x = torch.randn(3, 4, dtype=dtype, device=device)
    else:
        x = torch.randint(-10, 10, (3, 4), dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("value", [-1.5, 42])
def test_fill_scalar_basic(device: str, dtype: torch.dtype, shape: tuple, value: float):
    """Test basic fill.Scalar functionality with different dtypes, shapes, and values"""

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Create input tensor
    x = torch.randn(shape, dtype=dtype, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("shape", [(2, 3), (1, 4, 4)])
@pytest.mark.parametrize("value", [-5, 42])
def test_fill_scalar_integer_dtypes(
    device: str, dtype: torch.dtype, shape: tuple, value: int
):
    """Test fill.Scalar functionality with integer dtypes"""

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Create input tensor with integer values
    x = torch.zeros(shape, dtype=dtype, device=device) + 1

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("value", [-5, 100])
def test_fill_scalar_integer_values(device: str, value: int):
    """Test fill.Scalar with integer values"""

    def fn(x):
        return aten.fill.Scalar(x, value)

    # Test with float tensor
    x = torch.randn(3, 4, device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_fill_scalar_single_element(device: str):
    """Test fill.Scalar with single element tensor"""

    def fn(x):
        return torch.ops.aten.fill.Scalar(x, 7.5)

    # Single element tensor
    x = torch.tensor([1.0], device=device)

    check_functions_are_equivalent(fn, device, [x])


def test_fill_scalar_zero_dim(device: str):
    """Test fill.Scalar with single element tensor"""

    def fn(x):
        return torch.ops.aten.fill.Scalar(x, 7.5)

    # Single element tensor
    x = torch.tensor(1.0, device=device)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.xfail(reason="Fixme, currently off to support eager mode")
def test_max_pool2d_error_message_not_supported_output(device: str):
    def fn(x):
        return aten.max_pool2d_with_indices(x, kernel_size=2, stride=2)

    # Test different sizes
    batch_size, channels = 1, 2
    x = torch.randn(batch_size, channels, 16, 16)
    with pytest.raises(
        BackendCompilerFailed,
        match="The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet.",
    ):
        check_functions_are_equivalent(fn, device, [x])


@pytest.mark.xfail(reason="Fixme, currently off to support eager mode")
def test_max_pool2d_error_message_not_supported_in_graph(device: str):
    def fn(x):
        return aten.max_pool2d_with_indices(x, kernel_size=2, stride=2)[1] * 2

    # Test different sizes
    batch_size, channels = 1, 2
    x = torch.randn(batch_size, channels, 16, 16)
    with pytest.raises(
        BackendCompilerFailed,
        match="The implementation of aten.max_pool2d_with_indices doesn't support returning indices yet.",
    ):
        check_functions_are_equivalent(fn, device, [x])


# aten._log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def test_log_softmax_basic(conf: Conf):
    """Test _log_softmax basic functionality."""

    def fn(x):
        return aten._log_softmax(x, -1, False)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_log_softmax_numerical_stability(conf: Conf):
    """Test _log_softmax with large values to verify numerical stability."""

    def fn(x):
        return aten._log_softmax(x, -1, False)

    # Create tensor with large values that could cause overflow without max subtraction
    x = torch.randn(2, 3, dtype=torch.float32) * 100
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_log_softmax_half_to_float_true(conf: Conf, dim: int):
    """Test _log_softmax with half_to_float=True.

    When half_to_float=True:
    - Input must be float16
    - Computation is done in float32
    - Output is float32 (not converted back to float16)
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required for half_to_float=True tests"
            " as the cpu does not have a reference implementation."
        )

    def fn(x):
        initial_device = x.device
        if x.device.type == "cpu" and not torch.compiler.is_compiling():
            # We're in the reference eager cpu execution, which doesn't work on
            # cpu. We move the computation to cuda for reference.
            x = x.to("cuda")

        output = aten._log_softmax(x, dim, True)
        return output.to(initial_device)

    # half_to_float=True requires float16 input
    x = torch.randn(3, 4, 5, dtype=torch.float16)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("dim", [-1, 0])
def test_log_softmax_half_to_float_false(conf: Conf, dtype: torch.dtype, dim: int):
    """Test _log_softmax with half_to_float=False.

    When half_to_float=False:
    - Input can be any dtype
    - Output dtype matches input dtype
    - For float16 inputs, computation happens in float32 but result is converted back
    """

    def fn(x):
        return aten._log_softmax(x, dim, False)

    x = torch.randn(3, 4, 5, dtype=dtype)
    check_outputs(fn, conf, [x], atol=1e-3, rtol=1e-2)
