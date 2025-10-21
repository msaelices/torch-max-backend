---
name: aten-ops
description: Guide for implementing and reviewing PyTorch ATen operations in the MAX backend, including test-driven development workflow, MAX operation discovery, type hints, and best practices
---

# ATen Operations Implementation Skill

This skill provides comprehensive guidance for implementing PyTorch ATen operations using Modular's MAX framework in the torch-max-backend project.

## Using This Skill

Reference the skill in your prompts:

```
Use the aten-ops skill to help me implement aten::relu
```

```
What pattern should I use for a reduction operation? Use the aten-ops skill.
```

The skill provides:
- **8-step TDD workflow**: Complete test-driven development process
- **Implementation patterns**: Templates for all operation types in `assets/`
- **Testing strategies**: Comprehensive testing guide in `references/`
- **Type hints guidance**: Beartype method for finding correct types

## Overview

ATen (A Tensor Library) is PyTorch's foundational tensor operation library. This skill guides you through the complete workflow of adding support for ATen operations in the MAX backend using test-driven development.

## Core Workflow: The 8-Step Process

### Step 1: Research the ATen Operation

**Goal**: Understand the operation's signature, semantics, and expected behavior.

**How**:
```
Ask a subagent to explore the PyTorch codebase ../pytorch and look for the signature
and the meaning of inputs and outputs of this aten function and to give you a full report.
```

**What to find**:
- Function signature with all parameters
- Input tensor shapes and types
- Output tensor shape and type
- Special behaviors (broadcasting, in-place, optional parameters)
- Edge cases and boundary conditions
- Backward/gradient computation if applicable

**Where to look**:
- `../pytorch/aten/src/ATen/native/` - Native implementations
- `../pytorch/aten/src/ATen/native/native_functions.yaml` - Operation declarations
- `../pytorch/torch/` - Python API documentation

**Example research output**:
```
aten::relu(Tensor self) -> Tensor
- Applies ReLU activation: max(0, x) element-wise
- Input: Any shape tensor
- Output: Same shape as input, same dtype
- Supports: float32, float16, bfloat16
- In-place variant: relu_
```

### Step 2: Write Unit Tests

**Goal**: Define expected behavior through tests before implementation.

**Location**: `tests/test_aten_functions.py`

**Placement**: Insert tests in the middle of the file to avoid merge conflicts

**Pattern**:
```python
@pytest.mark.parametrize("dtype", [
    torch.float32,
    torch.float16,
    torch.bfloat16,
])
@pytest.mark.parametrize("shape", [
    (10,),           # 1D
    (100,),          # Larger 1D
    (10, 20),        # 2D
    (5, 10, 15),     # 3D
])
def test_aten_operation_name(dtype, shape):
    """Test aten::operation_name implementation."""
    # Create input tensors
    input_tensor = torch.randn(shape, dtype=dtype)

    # Reference implementation (PyTorch)
    expected = torch.operation_name(input_tensor)

    # Compile with MAX backend
    @torch.compile(backend=max_backend)
    def compiled_fn(x):
        return torch.operation_name(x)

    result = compiled_fn(input_tensor)

    # Verify results
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-7)
```

**Best practices**:
- Test multiple data types (float32, float16, bfloat16)
- Test various shapes (1D, 2D, 3D, large tensors)
- Test edge cases (empty tensors, size-1 dimensions, broadcasting)
- Test with different devices if applicable (CPU, CUDA)
- Use descriptive test names

### Step 3: Run Tests (Expect Failure)

**Command**:
```bash
uv run pytest tests/test_aten_functions.py::test_aten_operation_name -v
```

**Expected output**:
```
ValueError: Unsupported operation: aten.operation_name
```

This confirms the operation isn't implemented yet.

### Step 4: Locate or Add Operation Signature

**Location**: `torch_max_backend/aten_functions.py`

**Important**: File is sorted alphabetically - maintain this order!

**Format**:
```python
# operation_name(Tensor self, Scalar? alpha=None) -> Tensor
@map_to(aten.operation_name)
def aten_operation_name(
    self: TensorValue,
    alpha: float | None = None,
) -> TensorValue:
    # Implementation goes here
    pass
```

**If signature comment doesn't exist**:
1. Find it in PyTorch source or from Step 1 research
2. Add it as a comment above the function
3. Keep alphabetical order

**Signature comment format**:
```python
# operation_name(Tensor input, int dim, bool keepdim=False) -> Tensor
```

### Step 5: Research MAX Implementation

**Goal**: Find MAX operations that can implement the ATen operation.

**How**:
```
Ask a subagent to look into the directory ../modular/max to find if functions exist
in MAX to do something similar (sometimes they have direct equivalents) or can be
composed to re-implement the op. The subagent must give you a full report of useful
functions for your task and descriptions of inputs and outputs.
```

**Where to look**:
1. **MAX Graph Ops**: `../modular/max/graph/ops/` - High-level Python ops
2. **MAX Kernels**: `../modular/max/kernels/src/` - Lower-level implementations
3. **Examples**: `../modular/max/examples/` - Usage patterns
4. **Models**: Pre-built models showing operation usage

**Or use the mojo-gpu-kernels skill**:
```
Use the mojo-gpu-kernels skill to find MAX operations for implementing [operation_name]
```

**Decision tree**:

- **Direct equivalent exists**: Use it directly
  ```python
  return max_ops.relu(self)
  ```

- **Can compose operations**: Combine MAX ops
  ```python
  # log_softmax = log(softmax(x))
  return max_ops.log(max_ops.softmax(self, axis=dim))
  ```

- **"backward" in name**: Implement in Mojo, place in `mojo_kernels/`
  ```python
  # Use custom Mojo kernel
  return custom_mojo_kernel(self, ...)
  ```

- **No MAX alternative**: Port from PyTorch C++ implementation
  ```
  Use mojo-gpu-kernels skill to write custom Mojo kernel based on PyTorch C++ code
  ```

### Step 6: Implement the Operation

**Location**: `torch_max_backend/aten_functions.py`, directly below the signature comment

**Pattern**:
```python
# operation_name(Tensor self, int dim=-1) -> Tensor
@map_to(aten.operation_name)
def aten_operation_name(
    self: TensorValue,
    dim: int = -1,
) -> TensorValue:
    """
    Brief description of what this operation does.

    Maps to max_ops.equivalent_operation or custom implementation.
    """
    # Handle default/negative dimensions if needed
    if dim < 0:
        dim = len(self.shape) + dim

    # Call MAX operation
    return max_ops.operation(self, axis=dim)
```

**Type hints**:
- `TensorValue`: Input/output tensors
- `int`, `float`, `bool`: Scalar values
- `list[int]`: Integer sequences (shapes, dimensions)
- `int | None`, `float | None`: Optional scalar values
- `list[TensorValue]`: List of tensors

**Common patterns**:

**Element-wise operation**:
```python
@map_to(aten.relu)
def aten_relu(self: TensorValue) -> TensorValue:
    return max_ops.relu(self)
```

**Operation with dimension parameter**:
```python
@map_to(aten.softmax)
def aten_softmax(self: TensorValue, dim: int, dtype: int | None = None) -> TensorValue:
    result = max_ops.softmax(self, axis=dim)
    # Handle dtype if needed
    return result
```

**Operation with optional parameters**:
```python
@map_to(aten.add)
def aten_add(self: TensorValue, other: TensorValue, alpha: float = 1.0) -> TensorValue:
    if alpha != 1.0:
        other = max_ops.mul(other, alpha)
    return max_ops.add(self, other)
```

**List of tensors**:
```python
@map_to(aten.cat)
def aten_cat(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    return max_ops.concat(tensors, axis=dim)
```

### Step 7: Verify Implementation

**Run your specific tests**:
```bash
uv run pytest tests/test_aten_functions.py::test_aten_operation_name -v
```

**Test with different data types**:
```bash
uv run pytest tests/test_aten_functions.py::test_aten_operation_name -v -k "float32"
uv run pytest tests/test_aten_functions.py::test_aten_operation_name -v -k "float16"
```

**What to verify**:
- ✅ All tests pass
- ✅ Correct output values
- ✅ Correct output shapes
- ✅ Works with all tested dtypes
- ✅ Edge cases handled properly

**If tests fail**:
1. Check MAX operation signature matches usage
2. Verify dimension/axis parameter mapping
3. Check dtype handling
4. Add debug prints to understand intermediate values
5. Compare with PyTorch reference implementation

### Step 8: Run Linter

**Command**:
```bash
uvx pre-commit run --all-files
```

**What it checks**:
- Code formatting (Ruff)
- Import ordering
- Type hint correctness
- Line length
- Trailing whitespace

**Important**: Do NOT run the full test suite (`uv run pytest`) as it takes too long.

## Type Hints Discovery Process

Finding correct type hints can be challenging. Here's the systematic approach:

### The Beartype Method

1. **Add wrong type hint**:
   ```python
   @map_to(aten.operation)
   def aten_operation(self: datetime.timezone) -> TensorValue:  # Intentionally wrong
       return max_ops.operation(self)
   ```

2. **Run a unit test**:
   ```bash
   uv run pytest tests/test_aten_functions.py::test_aten_operation -v
   ```

3. **Beartype error reveals correct type**:
   ```
   beartype.roar.BeartypeCallHintParamViolation: @beartyped aten_operation()
   parameter self=<TensorValue ...> violates type hint <class 'datetime.timezone'>,
   as <TensorValue ...> not instance of <class 'datetime.timezone'>.
   ```

4. **Replace with correct type**:
   ```python
   def aten_operation(self: TensorValue) -> TensorValue:
       return max_ops.operation(self)
   ```

5. **Verify**:
   ```bash
   uv run pytest tests/test_aten_functions.py::test_aten_operation -v
   ```

6. **Run full suite to ensure type isn't too narrow**:
   ```bash
   uv run pytest tests/test_aten_functions.py -v
   ```

### Common Type Hints

```python
# Tensors
TensorValue              # Single tensor
list[TensorValue]        # List of tensors
TensorValue | None       # Optional tensor

# Scalars
int                      # Integer scalar
float                    # Float scalar
bool                     # Boolean
int | None               # Optional integer
float | None             # Optional float

# Sequences
list[int]                # Shape, stride, dimensions
tuple[int, ...]          # Variable-length int tuple

# Enumerations (if needed)
str                      # For mode strings like "bilinear", "nearest"
```

## Operation Categories

### Element-wise Operations

**Characteristics**:
- Process each element independently
- Output shape matches input shape
- Usually have direct MAX equivalents

**Examples**: relu, sigmoid, tanh, abs, neg, exp, log

**Pattern**:
```python
@map_to(aten.relu)
def aten_relu(self: TensorValue) -> TensorValue:
    return max_ops.relu(self)
```

### Reduction Operations

**Characteristics**:
- Reduce one or more dimensions
- Output shape smaller than input (unless keepdim=True)
- May return tuple (value, indices)

**Examples**: sum, mean, max, min, any, all

**Pattern**:
```python
@map_to(aten.sum)
def aten_sum(
    self: TensorValue,
    dim: int | None = None,
    keepdim: bool = False,
) -> TensorValue:
    if dim is None:
        # Reduce all dimensions
        return max_ops.sum(self)
    return max_ops.sum(self, axis=dim, keepdim=keepdim)
```

### Shape Operations

**Characteristics**:
- Change tensor shape or layout
- Usually don't modify data

**Examples**: reshape, view, transpose, permute, squeeze, unsqueeze

**Pattern**:
```python
@map_to(aten.transpose)
def aten_transpose(self: TensorValue, dim0: int, dim1: int) -> TensorValue:
    return max_ops.transpose(self, dim0, dim1)
```

### Linear Algebra Operations

**Characteristics**:
- Matrix or tensor operations
- May involve batching

**Examples**: matmul, bmm, addmm, mm

**Pattern**:
```python
@map_to(aten.matmul)
def aten_matmul(self: TensorValue, other: TensorValue) -> TensorValue:
    return max_ops.matmul(self, other)
```

### Tensor Creation/Manipulation

**Characteristics**:
- Create new tensors or combine existing ones
- May involve broadcasting

**Examples**: cat, stack, split, chunk

**Pattern**:
```python
@map_to(aten.cat)
def aten_cat(tensors: list[TensorValue], dim: int = 0) -> TensorValue:
    return max_ops.concat(tensors, axis=dim)
```

## Common Patterns

### Handling Negative Dimensions

```python
def aten_operation(self: TensorValue, dim: int) -> TensorValue:
    # Normalize negative dimension
    if dim < 0:
        dim = len(self.shape) + dim
    return max_ops.operation(self, axis=dim)
```

### Handling Optional Parameters

```python
def aten_operation(
    self: TensorValue,
    alpha: float | None = None,
) -> TensorValue:
    if alpha is None:
        alpha = 1.0
    return max_ops.operation(self, alpha=alpha)
```

### Dimension Mapping (dim vs axis)

PyTorch uses `dim`, MAX often uses `axis`:
```python
def aten_softmax(self: TensorValue, dim: int) -> TensorValue:
    return max_ops.softmax(self, axis=dim)  # Note: axis not dim
```

### Broadcasting Operations

```python
def aten_add(self: TensorValue, other: TensorValue, alpha: float = 1.0) -> TensorValue:
    # Handle alpha scaling
    if alpha != 1.0:
        other = max_ops.mul(other, alpha)
    # MAX handles broadcasting automatically
    return max_ops.add(self, other)
```

### In-place Operations

```python
# ATen has in-place variants with trailing underscore
# relu_ is in-place version of relu

@map_to(aten.relu_)
def aten_relu_(self: TensorValue) -> TensorValue:
    # For graph compilation, in-place can often be same as out-of-place
    # The graph compiler may optimize this
    return max_ops.relu(self)
```

## Debugging

### Environment Variables

```bash
# Show compilation and execution timing
TORCH_MAX_BACKEND_PROFILE=1 uv run pytest tests/test_aten_functions.py::test_op -v

# Show PyTorch FX graphs and detailed info
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_op -v

# Disable type checking (for debugging type errors)
TORCH_MAX_BACKEND_BEARTYPE=0 uv run pytest tests/test_aten_functions.py::test_op -v
```

### Common Issues

**Type mismatch errors**:
- Use beartype method to find correct types
- Check if parameter should be optional (`int | None`)

**Shape errors**:
- Print intermediate shapes for debugging
- Verify dimension/axis parameter handling
- Check if broadcasting is needed

**MAX operation not found**:
- Operation may have different name in MAX
- May need to compose multiple operations
- May need custom Mojo kernel

**Test numerical differences**:
- Adjust tolerance: `rtol=1e-4, atol=1e-6` for float16
- Some operations have acceptable numerical differences
- Check if operation is numerically stable

## Best Practices

1. **Always use test-driven development**: Write tests before implementation
2. **Use parametrized tests**: Test multiple dtypes and shapes
3. **Follow alphabetical order**: Maintain sorted order in `aten_functions.py`
4. **Add signature comments**: Include ATen signature as comment
5. **Use descriptive names**: Follow `aten_operation_name` convention
6. **Check for MAX equivalents first**: Don't reinvent the wheel
7. **Handle edge cases**: Test empty tensors, size-1 dims, negative dims
8. **Type hints accurately**: Use beartype method to find correct types
9. **Run linter before commit**: `uvx pre-commit run --all-files`
10. **Don't run full test suite during development**: Too slow, run specific tests

## Integration with mojo-gpu-kernels Skill

When implementing operations requiring custom kernels:

```
Use the mojo-gpu-kernels skill to help me implement a custom kernel for [operation]
```

The mojo-gpu-kernels skill provides:
- Kernel patterns for element-wise, reductions, shared memory
- Code templates for common kernel types
- Performance optimization guidance
- MAX operations catalog

## Quick Reference

**Research ATen op**:
```
Ask subagent to explore ../pytorch for aten::operation_name signature and semantics
```

**Find MAX equivalent**:
```
Ask subagent to search ../modular/max for operations similar to [operation_name]
```
Or:
```
Use mojo-gpu-kernels skill to find MAX operations for [operation_name]
```

**Test single operation**:
```bash
uv run pytest tests/test_aten_functions.py::test_aten_op_name -v
```

**Run linter**:
```bash
uvx pre-commit run --all-files
```

**Debug with environment variables**:
```bash
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_op -v
```

## Checklist

Before considering an ATen operation complete:

- [ ] Researched ATen operation signature and semantics
- [ ] Written parametrized unit tests (multiple dtypes, shapes)
- [ ] Tests initially fail with "Unsupported operation" error
- [ ] Found operation location in `aten_functions.py` (alphabetical)
- [ ] Added signature comment if missing
- [ ] Researched MAX equivalent operations
- [ ] Implemented operation with correct type hints
- [ ] All unit tests pass
- [ ] Tested with float32, float16, bfloat16 (if applicable)
- [ ] Tested edge cases (empty tensors, broadcasting, etc.)
- [ ] Linter passes (`uvx pre-commit run --all-files`)
- [ ] Code is in alphabetical order in `aten_functions.py`

## Resources

### Within This Skill
- **Implementation templates**: `assets/operation_template.py` - 12+ implementation patterns
- **Test templates**: `assets/test_template.py` - 15+ test patterns
- **Testing guide**: `references/testing_guide.md` - Comprehensive testing strategies
- **Common patterns**: `references/common_patterns.md` - All operation categories with examples

### Commands Reference

**Testing**:
```bash
uv run pytest tests/test_aten_functions.py::test_aten_op -v  # Run specific test
uv run pytest tests/test_aten_functions.py::test_aten_op -v -k "float32"  # Filter by dtype
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_aten_op -v  # Debug
```

**Linting**:
```bash
uvx pre-commit run --all-files  # Always run before committing
```

### Integration with mojo-gpu-kernels Skill

For custom kernel implementation:
```
Use the mojo-gpu-kernels skill to find MAX operations for [operation]
```
```
Use the mojo-gpu-kernels skill to write a custom kernel for [operation]
```

For detailed workflow documentation and examples, see the reference files in this skill.
