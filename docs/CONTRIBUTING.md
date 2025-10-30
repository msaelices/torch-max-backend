# Contributing to Torch MAX Backend

Thank you for your interest in contributing to the Torch MAX Backend project! This guide will help you get started with development and understand our development workflow.

## Prerequisites

- **uv**: Package manager for dependency management
- **Git**: For version control

## Getting Started

### 1. Fork and Clone the Repository

1. Fork the repository on GitHub
2. Clone your fork locally:

```bash
git clone git@github.com:youraccount/torch-max-backend.git
cd torch-max-backend/
```

### 2. Install Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
uvx pre-commit install
```

### 3. Verify Setup

Run the test suite to ensure everything is working:

```bash
# Run all tests with parallel execution
uv run pytest -n 15

# Run a test file
uv run pytest tests/test_compiler.py
```

## Project Architecture

The project supports PyTorch operations in **two execution modes**:

1. **Graph Mode**: Via `torch.compile(backend=max_backend)` - compiles FX graphs to MAX
2. **Eager Mode**: Via `torch.device("max_device")` - executes operations immediately on MAX

When implementing operations, you must:
- Implement in `torch_max_backend/aten_functions.py` (works for both modes)
- Register in `torch_max_backend/max_device/max_device_aten_ops.py` (enables eager mode)

## Development Workflow

### Code Quality Standards

We maintain high code quality using several tools:

- **Ruff**: Linting and formatting (>=0.12.7)
- **Beartype**: Runtime type checking
- **Pre-commit**: Automated code quality checks

## Adding Support for New Operations

We use **test-driven development** to add support for new PyTorch operations. Follow these steps:

### Step 1: Research the Operation

Explore the [PyTorch ATen native codebase](https://github.com/pytorch/pytorch/tree/main/aten/src/ATen/native) to understand:
- Function signature
- Input and output meanings
- Expected behavior

### Step 2: Write Unit Tests

1. Add unit tests in `tests/test_aten_functions.py`
2. Place tests in the middle of the file to avoid merge conflicts
3. Use `pytest.mark.parametrize` to test various data types

### Step 3: Run Tests (Should Fail)

```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op
```

You should see an error indicating the operation is not supported.

### Step 4: Find Operation Signature

1. Locate the operation signature in `torch_max_backend/aten_functions.py`
2. If not present, add it following alphabetical order
3. The signature should be in a comment above the implementation

### Step 5: Research MAX Implementation

Check the [Max Graph Ops](https://docs.modular.com/max/api/python/graph/ops/) available looking for:

- Equivalent MAX function
- Composable functions to implement the operation
- Examples in existing MAX models

If there is "backward" in the name, it's likely that you'll have to implement it in Mojo and put it in the "mojo_kernels" directory. If not, it's likely MAX already have something that can do it in.

When there is no MAX alternative, the best alternative would be to migrate to Mojo a C++ function, by looking for the signature in the Pytorch codebase.

### Step 6: Implement the Operation

Write the ATen operation implementation in `torch_max_backend/aten_functions.py` using MAX functions.

**Important**: Implementation must support **both** graph mode (`TensorValue`) and eager mode (`MaxEagerTensor`). Use `MaxTensor` type hint for dual-mode support.

Example:
```python
# aten::cat(Tensor[] tensors, int dim=0) -> Tensor
def aten_cat(tensors: list[MaxTensor], dim: int = 0) -> MaxTensor:
    return max_ops.concat(tensors, axis=dim)
```

### Step 7: Register for Eager Mode

Add registration in `torch_max_backend/max_device/max_device_aten_ops.py`:

```python
register_aten_op("aten::cat")(
    wrap_for_max_device(aten_functions.aten_cat)
)
```

Place in alphabetical order. The wrapper handles tensor conversion automatically.

### Step 8: Verify Implementation

```bash
# Run your specific tests
uv run pytest tests/test_aten_functions.py::test_your_new_op

# Run full test suite
uv run pytest -n 15

# Run code quality checks
uvx pre-commit run --all-files
```

## Type Hints

Finding correct type hints can be challenging. Here's our approach:

1. Add an obviously wrong type hint (e.g., `datetime.timezone`)
2. Run a unit test that calls the function
3. Beartype will throw an error with the correct type name
4. Replace with the correct type from Beartype's error message
5. Verify with unit tests and full test suite

## Debugging Tools

### Environment Variables

- `TORCH_MAX_BACKEND_PROFILE=1`: Enable timing profiling
- `TORCH_MAX_BACKEND_VERBOSE=1`: Show graph structures
- `TORCH_MAX_BACKEND_BEARTYPE=0`: Disable type checking (for debugging)

Values accepted: "1", "true", "yes" (case-insensitive)

### Example Usage

```bash
# Debug compilation with verbose output
TORCH_MAX_BACKEND_VERBOSE=1 uv run python your_script.py

# Profile performance
TORCH_MAX_BACKEND_PROFILE=1 uv run python your_script.py
```

## Testing Strategy

### Test Files

- **`test_aten_functions.py`**: Individual ATen operations (both graph and eager modes)
- **`test_compiler.py`**: Torch.compile backend integration
- **`test_max_device.py`**: MAX device eager execution
- **`test_high_level_ops.py`**: High-level composed operations

### Test Coverage Areas

- **ATen Operations**: Graph mode (`torch.compile`) and eager mode (`max_device`)
- **Device Support**: CPU and max_device (with GPU if available)
- **Compilation**: `torch.compile(backend=max_backend)` integration
- **Error Handling**: Unsupported operations

### Test Fixtures

- `tensor_shapes`: Common tensor shapes for testing
- `devices`: Available devices from MAX accelerator detection

## Keeping Your Fork Updated

To stay synchronized with the upstream repository:

```bash
# Add upstream remote (one-time setup)
git remote add upstream https://github.com/gabrieldemarmiesse/torch-max-backend.git

# Fetch latest changes
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Push updated branch
git push --force-with-lease origin your-branch-name
```

Happy contributing! 🚀
