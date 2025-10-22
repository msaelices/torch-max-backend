---
name: mojo-kernels
description: Guide for writing efficient GPU kernels in Mojo using the MAX framework, including patterns for element-wise operations, reductions, shared memory usage, and PyTorch backend integration
---

# Mojo GPU Kernel Development Skill

This skill provides comprehensive guidance for writing GPU kernels in Mojo using Modular's MAX framework, with specific focus on integration with the PyTorch backend.

## Using This Skill

Reference the skill in your prompts:

```
Use the mojo-kernels skill to help me implement a custom activation function.
```

```
I need to write a GPU kernel for matrix transpose. Use the mojo-kernels skill.
```

The skill provides:
- **Kernel patterns**: Element-wise, reductions, shared memory operations
- **Code templates**: Ready-to-use templates in `assets/` directory
- **Detailed references**: In-depth guides in `references/` directory
- **MAX operations catalog**: Complete reference of available MAX operations

## Core Concepts

### MAX GPU Kernel Architecture

Mojo GPU kernels are functions that execute on GPU devices via the MAX framework. Unlike CUDA/HIP kernels, Mojo kernels:
- Use regular function syntax (no `__global__` decorator)
- Leverage compile-time parameters for type safety and specialization
- Employ explicit address space annotations for memory types
- Integrate with DeviceContext for memory management and kernel launches
- **Use LayoutTensor for tensor operations** - This is the preferred high-level abstraction in MAX
- Use UnsafePointer for low-level kernel implementations when needed

### Working with Tensors

**Important**: When implementing ATen operations, prioritize using `LayoutTensor` over raw pointers:

- **LayoutTensor**: High-level tensor abstraction with layout information (preferred for most operations)
- **UnsafePointer**: Low-level pointer for custom kernels (use when LayoutTensor doesn't fit)
- **DeviceBuffer**: For managing GPU memory allocations
- **NDBuffer**: Deprecated - do not use

Example with LayoutTensor:
```mojo
fn operation[dtype: DType](
    input: LayoutTensor[dtype],
    ctx: DeviceContext
) -> LayoutTensor[dtype]:
    # LayoutTensor provides shape, strides, and layout information
    # Use MAX operations that accept LayoutTensor when possible
    return max_ops.operation(input)
```

### Development Workflow

When implementing a new ATen operation for the PyTorch backend:

1. **Research the operation**: Explore PyTorch source (`../pytorch`) to understand the ATen function signature and semantics
2. **Write tests first**: Add parametrized unit tests in `test_aten_functions.py`
3. **Find MAX equivalents**: Search `../modular/max/kernels` for similar operations or composable primitives
4. **Implement the kernel**: Add the operation to `aten_functions.py` with proper type hints
5. **Verify and iterate**: Run tests, fix type errors using beartype feedback, verify with multiple data types
6. **Lint before committing**: Run `uvx pre-commit run --all-files`

## Essential Imports

### Core GPU Primitives

```mojo
from gpu import (
    barrier,           # Thread synchronization
    block_dim,         # Block dimensions
    block_idx,         # Block index
    global_idx,        # Global thread index
    grid_dim,          # Grid dimensions
    lane_id,           # Warp lane ID
    thread_idx,        # Thread index within block
    WARP_SIZE,         # Warp size constant (32 for NVIDIA)
    syncwarp,          # Warp-level sync
)
from gpu.host import DeviceContext, DeviceBuffer, FuncAttribute
from gpu.memory import AddressSpace
import gpu.warp as warp
```

### Memory and Layout

```mojo
from layout import Layout, LayoutTensor, RuntimeLayout
from memory import stack_allocation
```

### Algorithms and Reductions

```mojo
from algorithm import vectorize, sync_parallelize
from algorithm.functional import elementwise
from algorithm._gpu.reduction import reduce_launch, block_reduce, row_reduce
```

### Math and Utilities

```mojo
from math import align_down, ceildiv, exp, log, rsqrt, sqrt, tanh
from sys import align_of, simd_width_of, size_of
```

## Kernel Patterns

### Pattern 1: Simple Element-wise Operation

**Use case**: Operations that process each element independently (activations, arithmetic)

**Structure**:
```mojo
fn elementwise_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    len: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    output[tid] = operation(input[tid])
```

**Launch configuration**:
```mojo
var block_dim = 256
var grid_dim = ceildiv(length, block_dim)
ctx.enqueue_function_checked[kernel, kernel](
    output_device,
    input_device,
    length,
    grid_dim=grid_dim,
    block_dim=block_dim,
)
```

**Reference**: See `references/elementwise_patterns.md`

### Pattern 2: Warp-level Reduction

**Use case**: Fast reductions within a warp (32 threads on NVIDIA)

**Structure**:
```mojo
fn warp_reduction_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= UInt(size):
        return

    # Warp primitives: sum, max, min, broadcast, shuffle
    var result = warp.sum(input[tid])
    if lane_id() == 0:
        output[tid // UInt(WARP_SIZE)] = result
```

**Available warp operations**:
- `warp.sum()`: Sum reduction across warp
- `warp.max()`, `warp.min()`: Min/max reduction
- `warp.broadcast()`: Broadcast from one lane
- `warp.shuffle()`: Exchange data between lanes
- `warp.lane_group_sum[num_lanes]()`: Partial warp reduction

**Reference**: See `references/reduction_patterns.md`

### Pattern 3: Block-level Reduction with Shared Memory

**Use case**: Reductions across entire thread block (requires shared memory)

**Structure**:
```mojo
@always_inline
fn block_reduce[dtype: DType, max_warps_per_block: Int](
    val: Scalar[dtype]
) -> Scalar[dtype]:
    # Shared memory for warp results
    var warp_results = stack_allocation[
        max_warps_per_block, dtype,
        address_space = AddressSpace.SHARED
    ]()
    var broadcast = stack_allocation[
        1, dtype, address_space = AddressSpace.SHARED
    ]()

    # Reduce within warp
    var warp_sum = warp.sum(val)
    var warp_idx = thread_idx.x // UInt(WARP_SIZE)
    var lane_idx = lane_id()

    # First thread of each warp stores result
    if lane_idx == 0:
        warp_results[warp_idx] = warp_sum
    barrier()

    # Final reduction across warps
    if warp_idx == 0 and lane_idx < UInt(max_warps_per_block):
        var block_sum = warp.lane_group_sum[num_lanes=max_warps_per_block](
            warp_results[lane_idx]
        )
        if lane_idx == 0:
            broadcast[0] = block_sum
    barrier()

    return broadcast[0]
```

**Key points**:
- Use `stack_allocation` with `AddressSpace.SHARED` for shared memory
- Call `barrier()` after writes before reads
- Two-level reduction: warp-level then across warps
- Size shared memory for maximum warps per block

**Reference**: See `references/reduction_patterns.md`

### Pattern 4: Shared Memory Stencil Operations

**Use case**: Operations requiring neighbor access (convolution, pooling, stencil)

**Structure**:
```mojo
fn stencil_kernel(
    output: UnsafePointer[Float32],
    input: UnsafePointer[Float32],
    size: Int,
):
    alias BLOCK_SIZE = 256
    var tid = global_idx.x
    var local_idx = thread_idx.x + 1  # Offset for halo

    # Allocate shared memory with halo region
    var shared = stack_allocation[
        BLOCK_SIZE + 2,  # +2 for left/right neighbors
        DType.float32,
        address_space = AddressSpace.SHARED
    ]()

    # Load data with boundary handling
    if tid < UInt(size):
        shared[local_idx] = input[tid]

    # Load halo elements (first thread loads left, last loads right)
    if thread_idx.x == 0:
        shared[0] = (tid > 0).select(input[tid - 1], 0.0)
    if thread_idx.x == BLOCK_SIZE - 1:
        shared[local_idx + 1] = (tid < UInt(size - 1)).select(
            input[tid + 1], 0.0
        )

    barrier()

    # Compute using neighbors
    if tid < UInt(size):
        output[tid] = (
            shared[local_idx - 1] +
            shared[local_idx] +
            shared[local_idx + 1]
        ) / 3.0
```

**Key considerations**:
- Allocate shared memory for block + halo regions
- Handle boundary conditions (clamp, zero-pad, wrap)
- Synchronize after loading before computation
- Coalescence: consecutive threads load consecutive addresses

**Reference**: See `references/shared_memory_patterns.md`

## Type Handling

### SIMD Types and Vectorization

```mojo
# SIMD vectors for parallelism
var vec = SIMD[DType.float32, 8]()  # 8-wide float32 vector

# Horizontal reductions
var sum = vec.reduce_add()
var max_val = vec.reduce_max()
var min_val = vec.reduce_min()

# Type casting
var casted = vec.cast[DType.float64]()
```

### Compile-time Parameters

```mojo
fn generic_kernel[
    dtype: DType,           # Data type parameter
    simd_width: Int,        # Vectorization width
](input: UnsafePointer[Scalar[dtype]], size: Int):
    @parameter
    @always_inline
    fn inner_op(x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
        return x * x

    # Use parameters for specialization...
```

### Type Hints for ATen Operations

Finding correct type hints:
1. Add intentionally wrong type hint (e.g., `datetime.timezone`)
2. Run unit test calling the function
3. Beartype error reveals actual type being passed
4. Replace hint with correct type
5. Verify with full test suite to ensure type isn't too narrow

**Common types**:
- `Tensor`: PyTorch tensor
- `Optional[Tensor]`: Nullable tensor
- `IntArrayRef`: Sequence of integers (shape, stride)
- `Scalar`: Single value (int, float)
- `bool`: Boolean flag
- `Optional[int]`: Nullable integer

## Device Context Management

### Basic Pattern

```mojo
def main():
    with DeviceContext() as ctx:
        # Allocate device buffers
        var input_device = ctx.enqueue_create_buffer[DType.float32](size)
        var output_device = ctx.enqueue_create_buffer[DType.float32](size)

        # Copy data to device
        ctx.enqueue_copy(input_device, input_host)

        # Launch kernel
        ctx.enqueue_function_checked[kernel, kernel](
            output_device,
            input_device,
            size,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )

        # Copy results back
        ctx.enqueue_copy(output_host, output_device)

        # Wait for completion
        ctx.synchronize()
```

### Buffer Mapping for Initialization

```mojo
with DeviceContext() as ctx:
    var buffer = ctx.enqueue_create_buffer[DType.float32](size)

    # Map to host for initialization
    with buffer.map_to_host() as host_ptr:
        for i in range(size):
            host_ptr[i] = i

    # Launch kernel...

    # Map to host for verification
    with buffer.map_to_host() as host_ptr:
        for i in range(size):
            assert host_ptr[i] == expected[i]
```

## Activation Functions

Common activation patterns from MAX framework:

```mojo
@always_inline
fn relu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    return max(x, 0)

@always_inline
fn gelu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    alias inv_SQRT_2 = 0.70710678118654752440
    var val_half = 0.5 * x
    var erf_res = math.erf(x * inv_SQRT_2)
    return val_half.fma(erf_res, val_half)

@always_inline
fn leaky_relu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width],
    negative_slope: Scalar[dtype]
) -> SIMD[dtype, simd_width]:
    return x.ge(0).select(x, negative_slope * x)

@always_inline
fn silu[dtype: DType, simd_width: Int](
    x: SIMD[dtype, simd_width]
) -> SIMD[dtype, simd_width]:
    # SiLU/Swish: x * sigmoid(x)
    return x * (1.0 / (1.0 + exp(-x)))
```

**Reference**: See `assets/activation_templates.mojo`

## Performance Considerations

### Occupancy Optimization

- **Block size**: Multiple of warp size (32), typically 128-256 threads
- **Grid size**: Enough blocks to saturate GPU (`ceildiv(N, block_size)`)
- **Registers**: Minimize local variables to increase occupancy
- **Shared memory**: Balance usage vs. occupancy

### Memory Coalescing

- **Global memory**: Consecutive threads access consecutive addresses
- **Alignment**: Align data structures to 128-byte boundaries when possible
- **Strided access**: Avoid if possible; use shared memory to restructure

### Warp Divergence

- **Minimize branches**: Use select() instead of if/else when possible
- **Uniform control flow**: Keep threads in a warp on same execution path
- **Early exit**: Place bounds checks early to minimize wasted work

### Shared Memory Bank Conflicts

- **32 banks**: Pad arrays to avoid stride-32 access patterns
- **Broadcast**: Reading same address from multiple threads is conflict-free

## Common Patterns from MAX

The MAX framework provides high-level patterns that generate optimized kernels:

### Element-wise Operations

```mojo
@always_inline
@parameter
fn operation[simd_width: Int, rank: Int](idx: IndexList[rank]):
    output_tensor.store(
        idx,
        func(input_tensor.load[width=simd_width](idx))
    )

elementwise[operation, pack_size, target="gpu"](dims, ctx)
```

### Reductions

```mojo
@parameter
fn reduce_fn[dtype: DType, width: Int, reduction_idx: Int](
    lhs: SIMD[dtype, width],
    rhs: SIMD[dtype, width]
) -> SIMD[dtype, width]:
    return lhs + rhs

reduce_launch[num_reductions, input_fn, output_fn, reduce_fn, rank, dtype](
    shape, axis, init_value, ctx
)
```

## Available MAX Operations

The MAX framework provides implementations for:

- **Activations**: relu, gelu, elu, leaky_relu, tanh, sigmoid, silu
- **Normalization**: layer_norm, batch_norm, softmax, log_softmax
- **Reductions**: sum, max, min, mean, variance
- **Linear Algebra**: matmul, bmm, gemv, grouped_matmul
- **Convolution**: conv2d, conv_transpose
- **Pooling**: max_pool, avg_pool
- **Attention**: flash_attention, multi-head attention
- **Utilities**: broadcast, concat, split, pad, reshape, transpose

**Reference**: See `references/max_operations.md` for detailed API

## Debugging and Profiling

### Environment Variables

```bash
# Enable profiling
TORCH_MAX_BACKEND_PROFILE=1 uv run pytest test_file.py

# Enable verbose output (print graph structures)
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest test_file.py
```

### Kernel Debugging

- **Print from kernel**: Not directly supported; use output buffers for debugging values
- **Bounds checking**: Always include `if tid >= UInt(size): return`
- **Synchronization**: Verify barrier() calls match across all threads
- **Type mismatches**: Use beartype's error messages to identify type issues

### Testing Strategy

```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(10,), (100,), (1000,), (10, 10)])
def test_operation(dtype, shape):
    # Test multiple data types and shapes
    input_tensor = torch.randn(shape, dtype=dtype)
    # ...
```

## Best Practices

1. **Prioritize LayoutTensor**: Use `LayoutTensor[dtype]` for tensor operations instead of raw pointers or deprecated NDBuffer
2. **Always bounds-check**: Start kernels with `if tid >= UInt(size): return`
3. **Use compile-time parameters**: `@parameter` for dtype, simd_width enables specialization
4. **Prefer warp primitives**: Use `warp.sum()` over manual reduction when possible
5. **Synchronize correctly**: Call `barrier()` after shared memory writes before reads
6. **Vectorize on CPU**: Use `vectorize[]` for CPU-like patterns
7. **Check alignment**: Use `align_of[]` for proper memory alignment
8. **Manage context lifetime**: Always use `with DeviceContext() as ctx:`
9. **Type hint accurately**: Use beartype feedback to find correct types
10. **Test thoroughly**: Parametrize tests across dtypes and shapes
11. **Lint before commit**: Run `uvx pre-commit run --all-files`

## Common Pitfalls

- **Using deprecated NDBuffer**: Use LayoutTensor instead for tensor operations
- **Missing synchronization**: Forgetting `barrier()` after shared memory writes
- **Incorrect grid/block sizing**: Not using `ceildiv()` for grid dimension
- **Type mismatches**: Not casting between accumulation type and output type
- **Bank conflicts**: Accessing shared memory with stride-32 patterns
- **Warp divergence**: Using if/else instead of select() for simple conditions
- **Uninitialized shared memory**: Not writing to all shared memory before reading
- **Incorrect halo handling**: Off-by-one errors in stencil boundary conditions

## Quick Reference

### Template Selection
- Element-wise: `assets/elementwise_template.mojo`
- Reduction: `assets/reduction_template.mojo`
- Activations: `assets/activation_templates.mojo`

### Reference Documentation
- Element-wise patterns: `references/elementwise_patterns.md`
- Reduction patterns: `references/reduction_patterns.md`
- Shared memory patterns: `references/shared_memory_patterns.md`
- MAX operations: `references/max_operations.md`

### Common Calculations
- Grid dimension: `ceildiv(total_elements, block_size)`
- Thread index: `global_idx.x`
- Warp index: `thread_idx.x // UInt(WARP_SIZE)`
- Lane index: `lane_id()`

### Synchronization
- Block-level: `barrier()`
- Warp-level: Implicit in warp primitives (`warp.sum()`, etc.)

### Pattern Selection Guide

**Use element-wise when**: Each output depends only on corresponding input(s), no thread communication needed

**Use warp reduction when**: Reducing ≤32 elements, need very fast reduction

**Use block reduction when**: Reducing ≤1024 elements, can use shared memory

**Use shared memory when**: Need neighbor access (stencils), data reuse opportunities

For detailed examples and complete implementations, consult the reference files and template assets.
