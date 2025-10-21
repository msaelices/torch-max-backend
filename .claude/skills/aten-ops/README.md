# ATen Operations Implementation Skill

Comprehensive skill for implementing and reviewing PyTorch ATen operations in the MAX backend.

## Overview

This skill guides you through the complete workflow of adding support for PyTorch ATen (A Tensor Library) operations using Modular's MAX framework, following test-driven development practices.

## Skill Structure

```
aten-ops/
├── SKILL.md                          # Main skill file with complete workflow
├── README.md                         # This file
├── references/                       # Detailed documentation
│   ├── testing_guide.md             # Comprehensive testing guide
│   └── common_patterns.md           # Reusable implementation patterns
└── assets/                           # Code templates
    ├── operation_template.py        # Operation implementation templates
    └── test_template.py             # Test templates
```

## Quick Start

### Using the Skill

In Claude Code, reference the skill in your prompts:

```
Use the aten-ops skill to help me implement aten::relu
```

```
I need to add support for aten::softmax. Use the aten-ops skill to guide me through the process.
```

### The 8-Step Workflow

1. **Research**: Understand the ATen operation signature and semantics
2. **Write Tests**: Create parametrized unit tests (TDD)
3. **Run Tests**: Verify tests fail with "Unsupported operation"
4. **Locate Signature**: Find or add operation signature in `aten_functions.py`
5. **Research MAX**: Find MAX operations to implement the ATen op
6. **Implement**: Write the operation using MAX ops
7. **Verify**: Run tests and ensure they pass
8. **Lint**: Run linter before committing

## Key Features

### Comprehensive Workflow

- **Step-by-step guidance** through test-driven development
- **Integration with research agents** for PyTorch and MAX exploration
- **Type hint discovery** using beartype method
- **Best practices** for each operation category

### Operation Patterns

Complete patterns for:
- Element-wise operations (relu, sigmoid, tanh)
- Reductions (sum, mean, max, min)
- Shape manipulation (view, transpose, squeeze)
- Linear algebra (matmul, bmm)
- Tensor combination (cat, stack, split)

### Testing Strategies

- Parametrized testing for multiple dtypes and shapes
- Edge case coverage
- Broadcasting tests
- Numerical stability tests
- Tolerance guidelines by dtype

### Code Templates

Ready-to-use templates in `assets/`:
- 12+ operation implementation patterns
- 15+ test patterns
- Helper functions
- Type hint examples

## Common Use Cases

### Implementing a New Operation

```
Use the aten-ops skill to implement aten::log_softmax

The skill will:
1. Guide you through researching the operation
2. Help you write parametrized tests
3. Show you how to find MAX equivalents
4. Provide implementation pattern
5. Guide verification process
```

### Reviewing an Implementation

```
Use the aten-ops skill to review this aten::relu implementation

The skill will check:
- Correct signature and type hints
- Proper handling of edge cases
- Test coverage
- Following best practices
```

### Finding Implementation Patterns

```
What pattern should I use for implementing a reduction operation? Use the aten-ops skill.

The skill provides:
- Multiple reduction patterns
- Examples with code
- Type hints
- Testing strategies
```

## Integration with Other Skills

### mojo-gpu-kernels Skill

For operations requiring custom kernels:

```
Use the aten-ops skill to implement aten::custom_backward, and use the
mojo-gpu-kernels skill for writing the custom Mojo kernel.
```

The aten-ops skill references mojo-gpu-kernels for:
- Finding MAX operations catalog
- Custom kernel implementation
- Performance optimization

## Operation Categories

### Element-wise
Operations that process each element independently.
**Examples**: relu, sigmoid, abs, exp, log

### Reductions
Operations that reduce tensor dimensions.
**Examples**: sum, mean, max, min, any, all

### Shape Manipulation
Operations that change tensor shape or layout.
**Examples**: view, reshape, transpose, squeeze

### Linear Algebra
Matrix and tensor mathematical operations.
**Examples**: matmul, bmm, addmm, mv

### Tensor Combination
Operations that create or combine tensors.
**Examples**: cat, stack, split, chunk

## Testing Guidelines

### Data Types to Test

```python
@pytest.mark.parametrize("dtype", [
    torch.float32,    # Always test
    torch.float16,    # Half precision
    torch.bfloat16,   # ML-optimized precision
])
```

### Shapes to Test

```python
@pytest.mark.parametrize("shape", [
    (10,),           # 1D
    (10, 20),        # 2D
    (5, 10, 15),     # 3D
    (1,),            # Edge: single element
    (0,),            # Edge: empty
])
```

### Tolerance by Dtype

- **float32**: `rtol=1e-5, atol=1e-7`
- **float16**: `rtol=1e-3, atol=1e-5`
- **bfloat16**: `rtol=1e-2, atol=1e-4`

## Type Hints Reference

Common types used in implementations:

```python
TensorValue              # Single tensor
list[TensorValue]        # List of tensors
int, float, bool         # Scalar values
int | None               # Optional integer
TensorValue | None       # Optional tensor
list[int]                # Integer sequences
```

## Commands Reference

### Testing

```bash
# Run specific test
uv run pytest tests/test_aten_functions.py::test_aten_relu -v

# Test with specific dtype
uv run pytest tests/test_aten_functions.py::test_aten_relu -v -k "float32"

# Run with verbose output
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_aten_relu -v
```

### Linting

```bash
# Run linter (always before committing)
uvx pre-commit run --all-files
```

### Debugging

```bash
# Show FX graphs
TORCH_MAX_BACKEND_VERBOSE=1 uv run pytest tests/test_aten_functions.py::test_op -v

# Show timing
TORCH_MAX_BACKEND_PROFILE=1 uv run pytest tests/test_aten_functions.py::test_op -v

# Disable type checking
TORCH_MAX_BACKEND_BEARTYPE=0 uv run pytest tests/test_aten_functions.py::test_op -v
```

## Best Practices

1. ✅ **Write tests first** - TDD ensures correctness
2. ✅ **Parametrize extensively** - Test multiple dtypes and shapes
3. ✅ **Follow alphabetical order** - Keep `aten_functions.py` sorted
4. ✅ **Add signature comments** - Include ATen signature
5. ✅ **Handle negative dimensions** - Normalize to positive
6. ✅ **Check for MAX equivalents** - Don't reinvent the wheel
7. ✅ **Use accurate type hints** - Use beartype method
8. ✅ **Test edge cases** - Empty tensors, size-1 dims
9. ✅ **Run linter** - Before committing
10. ✅ **Don't run full test suite** - Too slow during development

## Quick Reference

### Research ATen Operation
```
Ask subagent to explore ../pytorch for aten::operation signature and semantics
```

### Find MAX Equivalent
```
Ask subagent to search ../modular/max for operations similar to [operation]
```
Or:
```
Use mojo-gpu-kernels skill to find MAX operations for [operation]
```

### Verify Implementation
```bash
uv run pytest tests/test_aten_functions.py::test_aten_op -v
uvx pre-commit run --all-files
```

## Implementation Checklist

Before considering an operation complete:

- [ ] Researched ATen operation
- [ ] Written parametrized tests
- [ ] Tests fail initially
- [ ] Found/added signature in `aten_functions.py`
- [ ] Researched MAX equivalents
- [ ] Implemented with correct type hints
- [ ] All tests pass
- [ ] Tested multiple dtypes
- [ ] Tested edge cases
- [ ] Linter passes
- [ ] Alphabetical order maintained

## Resources

### Within the Skill
- `SKILL.md` - Complete workflow with all 8 steps
- `references/testing_guide.md` - Comprehensive testing guide
- `references/common_patterns.md` - Implementation patterns
- `assets/operation_template.py` - Implementation templates
- `assets/test_template.py` - Test templates

### External Resources
- PyTorch source: `../pytorch/aten/src/ATen/native/`
- MAX operations: `../modular/max/`
- Project docs: `CLAUDE.md`
- Contributing guide: `docs/CONTRIBUTING.md`

## Examples

### Simple Element-wise

```python
# relu(Tensor self) -> Tensor
@map_to(aten.relu)
def aten_relu(self: TensorValue) -> TensorValue:
    return max_ops.relu(self)
```

### Reduction with Dimension

```python
# sum(Tensor self, int dim, bool keepdim=False) -> Tensor
@map_to(aten.sum)
def aten_sum(self: TensorValue, dim: int, keepdim: bool = False) -> TensorValue:
    if dim < 0:
        dim = len(self.shape) + dim
    return max_ops.sum(self, axis=dim, keepdim=keepdim)
```

### Composed Operation

```python
# log_softmax(Tensor self, int dim) -> Tensor
@map_to(aten.log_softmax)
def aten_log_softmax(self: TensorValue, dim: int) -> TensorValue:
    if dim < 0:
        dim = len(self.shape) + dim
    return max_ops.log_softmax(self, axis=dim)
```

## Getting Help

If you're unsure about implementation:

```
How do I implement aten::operation_name? Use the aten-ops skill.
```

The skill will provide:
- Appropriate pattern for the operation type
- Code template
- Testing strategy
- Type hints
- Common pitfalls

## Maintenance

To keep the skill up to date:
- Add new patterns as you discover them
- Update MAX operations catalog references
- Add examples from successful implementations
- Document common issues and solutions

---

This skill is part of the torch-max-backend project and works in conjunction with the mojo-gpu-kernels skill for complete ATen operation support.
