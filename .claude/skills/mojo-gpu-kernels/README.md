# Mojo GPU Kernels Skill

A comprehensive skill for writing efficient GPU kernels in Mojo using Modular's MAX framework, with specific focus on PyTorch backend integration.

## Overview

This skill provides:
- **Patterns**: Element-wise, reductions, shared memory operations
- **Templates**: Ready-to-use code templates for common kernel types
- **References**: Detailed documentation on kernel development
- **MAX Integration**: Guide to available MAX operations and how to use them

## Skill Structure

```
mojo-gpu-kernels/
├── SKILL.md                          # Main skill file with comprehensive guidance
├── references/                       # Detailed pattern documentation
│   ├── elementwise_patterns.md       # Element-wise operation patterns
│   ├── reduction_patterns.md         # Reduction patterns (warp, block, global)
│   ├── shared_memory_patterns.md     # Shared memory usage patterns
│   └── max_operations.md             # MAX framework operations catalog
└── assets/                           # Code templates
    ├── elementwise_template.mojo     # Element-wise kernel templates
    ├── reduction_template.mojo       # Reduction kernel templates
    └── activation_templates.mojo     # Neural network activation functions
```

## Using the Skill

### In Claude Code

Simply reference the skill by name in your requests:

```
Use the mojo-gpu-kernels skill to help me implement a custom activation function.
```

or

```
I need to write a GPU kernel for matrix transpose. Can you use the mojo-gpu-kernels
skill to show me the best approach?
```

### Key Capabilities

The skill helps with:

1. **Implementing ATen Operations**
   - Following test-driven development workflow
   - Finding MAX equivalents for PyTorch operations
   - Proper type hints and error handling

2. **Writing Custom Kernels**
   - Element-wise operations (activations, math functions)
   - Reductions (sum, max, min, mean, variance)
   - Shared memory operations (transpose, stencils, convolutions)

3. **Optimization**
   - Choosing the right kernel pattern
   - Memory coalescing and bank conflict avoidance
   - Occupancy optimization
   - Using warp primitives effectively

4. **Integration with PyTorch Backend**
   - Device context management
   - Buffer allocation and copying
   - Kernel launch configuration
   - Testing strategies

## Quick Start Examples

### Example 1: Simple Element-wise Operation

```mojo
from gpu import global_idx

fn square_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    output[tid] = input[tid] * input[tid]
```

### Example 2: Warp Reduction

```mojo
import gpu.warp as warp
from gpu import global_idx, lane_id, WARP_SIZE

fn warp_sum_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    size: Int,
):
    var tid = global_idx.x
    if tid >= UInt(size):
        return

    var value = input[tid]
    var warp_sum = warp.sum(value)

    if lane_id() == 0:
        output[tid // UInt(WARP_SIZE)] = warp_sum
```

### Example 3: Using MAX Operations

```mojo
from max.nn import relu, gelu

# Use built-in MAX operations when available
fn activation_kernel[dtype: DType](
    output: UnsafePointer[Scalar[dtype]],
    input: UnsafePointer[Scalar[dtype]],
    len: Int,
):
    var tid = global_idx.x
    if tid >= UInt(len):
        return

    var x = input[tid]
    output[tid] = gelu[dtype, 1](x)
```

## Reference Documentation

### Pattern Selection Guide

**Choose element-wise pattern when:**
- Each output depends only on corresponding input(s)
- No communication between threads needed
- Examples: activations, arithmetic operations, broadcasting

**Choose warp reduction when:**
- Reducing up to 32 elements
- Need very fast reduction
- Examples: small tensor reductions, per-warp statistics

**Choose block reduction when:**
- Reducing up to 1024 elements (typical block size)
- Need shared memory for efficiency
- Examples: row-wise reductions, block-level statistics

**Choose multi-block reduction when:**
- Reducing large arrays (millions of elements)
- Need multiple kernel launches
- Examples: full tensor sum/mean/max

**Choose shared memory pattern when:**
- Need neighbor access (stencils)
- Data reuse opportunities
- Avoiding redundant global memory loads
- Examples: convolution, pooling, transpose

### Performance Guidelines

1. **Memory Bandwidth**
   - Element-wise ops are memory-bound
   - Fuse operations to increase arithmetic intensity
   - Ensure coalesced memory access

2. **Occupancy**
   - Block size: 128-256 threads typical
   - Balance shared memory usage vs. active blocks
   - Minimize register usage

3. **Synchronization**
   - Use warp primitives (no sync needed within warp)
   - Minimize barrier() calls
   - Avoid divergent control flow before barriers

4. **Shared Memory**
   - Pad arrays to avoid bank conflicts
   - 48-96 KB available per block
   - Use for data reuse and inter-thread communication

## Integration with torch-max-backend

This skill is specifically designed for the torch-max-backend project workflow:

### Step 1: Research
```
Ask subagent to explore ../pytorch for ATen operation signature and semantics
```

### Step 2: Write Tests
```python
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_my_operation(dtype):
    # Test implementation
    pass
```

### Step 3: Find MAX Equivalents
```
Ask subagent to search ../modular/max/kernels for similar operations
```

### Step 4: Implement
```mojo
# Add to aten_functions.py with proper type hints
fn aten_my_operation(...) -> Tensor:
    # Implementation using MAX operations or custom kernel
    pass
```

### Step 5: Verify
```bash
uv run pytest tests/test_aten_functions.py -k "my_operation" -v
uvx pre-commit run --all-files
```

## Common Patterns Quick Reference

### Imports
```mojo
from gpu import global_idx, block_idx, thread_idx, barrier, lane_id, WARP_SIZE
from gpu.host import DeviceContext
from gpu.memory import AddressSpace
import gpu.warp as warp
from memory import stack_allocation
from math import ceildiv
```

### Launch Configuration
```mojo
var block_dim = 256
var grid_dim = ceildiv(total_elements, block_dim)

ctx.enqueue_function_checked[kernel, kernel](
    output, input, size,
    grid_dim=grid_dim,
    block_dim=block_dim,
)
```

### Shared Memory Allocation
```mojo
var shared = stack_allocation[
    SIZE, dtype,
    address_space = AddressSpace.SHARED
]()
```

### Synchronization
```mojo
barrier()  # Block-level synchronization
# Warp primitives have implicit synchronization
```

## Resources

### Within the Skill
- `SKILL.md`: Main skill document with comprehensive guidance
- `references/`: Pattern documentation and best practices
- `assets/`: Ready-to-use code templates

### External References
- MAX kernels source: `../modular/max/kernels/src/`
- MAX examples: `../modular/max/examples/`
- PyTorch source: `../pytorch/`
- Project documentation: `CLAUDE.md`

## Tips for Using This Skill

1. **Start with templates**: Use the asset templates as starting points
2. **Reference the patterns**: Check reference docs for detailed explanations
3. **Search MAX operations**: Many operations already exist in MAX
4. **Test thoroughly**: Use parametrized tests with multiple dtypes and shapes
5. **Profile**: Use environment variables to enable profiling and verbose output
6. **Iterate**: Start simple, verify correctness, then optimize

## Skill Maintenance

To update this skill:

1. Add new patterns to appropriate reference files
2. Create templates for common new kernel types
3. Update MAX operations catalog as framework evolves
4. Add examples from successful implementations

## License

This skill is part of the torch-max-backend project.
