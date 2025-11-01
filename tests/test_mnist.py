"""
Unit tests for MNIST SimpleNet model.

This module tests the MNIST model implementation from demo_scripts/mnist.py,
including forward pass, backward pass, training, evaluation, and compilation.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from demo_scripts.mnist import SimpleNet
from torch_max_backend import max_backend
from torch_max_backend.testing import (
    Conf,
    check_functions_are_equivalent,
    check_outputs,
)


def test_mnist_simplenet_forward(device: str):
    """Test SimpleNet forward pass produces correct output shape."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)

    output = model(x)

    assert output.shape == (8, 10), f"Expected shape (8, 10), got {output.shape}"
    assert output.device.type == torch.device(device).type


def test_mnist_simplenet_compilation(device: str):
    """Test SimpleNet compiles successfully with max_backend."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)

    check_functions_are_equivalent(model, device, [x])


def test_mnist_no_graph_breaks(device: str):
    """Ensure SimpleNet compiles without graph breaks (fullgraph=True works)."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)

    explanation = torch._dynamo.explain(model)(x)

    assert explanation.graph_break_count == 0, (
        f"Expected 0 graph breaks, got {explanation.graph_break_count}"
    )
    assert explanation.graph_count == 1, (
        f"Expected 1 graph, got {explanation.graph_count}"
    )


@pytest.mark.skip(reason="Uses unsupported operations in compiled mode")
def test_mnist_training_step(device: str):
    """Test complete MNIST training iteration (forward + backward + optimizer step)."""

    class SimpleNet(torch.nn.Module):
        """Simple feedforward neural network for MNIST classification."""

        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(784, 128)
            self.fc2 = torch.nn.Linear(128, 64)
            self.fc3 = torch.nn.Linear(64, 10)

        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return loss

    x = torch.randn(8, 1, 28, 28).to(device)
    y = torch.randint(0, 10, (8,)).to(device)

    # Reset weights before each test
    model.fc1.weight.data.fill_(0.01)
    model.fc1.bias.data.fill_(0.01)
    model.fc2.weight.data.fill_(0.01)
    model.fc2.bias.data.fill_(0.01)
    model.fc3.weight.data.fill_(0.01)
    model.fc3.bias.data.fill_(0.01)

    loss_not_compiled = train_step(x, y).cpu().detach().numpy()
    weight_fc1_not_compiled = model.fc1.weight.data.cpu().numpy()
    weight_fc2_not_compiled = model.fc2.weight.data.cpu().numpy()
    weight_fc3_not_compiled = model.fc3.weight.data.cpu().numpy()

    # Reset and compile
    model.fc1.weight.data.fill_(0.01)
    model.fc1.bias.data.fill_(0.01)
    model.fc2.weight.data.fill_(0.01)
    model.fc2.bias.data.fill_(0.01)
    model.fc3.weight.data.fill_(0.01)
    model.fc3.bias.data.fill_(0.01)

    loss_compiled = (
        torch.compile(backend=max_backend)(train_step)(x, y).cpu().detach().numpy()
    )
    weight_fc1_compiled = model.fc1.weight.data.cpu().numpy()
    weight_fc2_compiled = model.fc2.weight.data.cpu().numpy()
    weight_fc3_compiled = model.fc3.weight.data.cpu().numpy()

    # Compare loss and weights
    np.testing.assert_allclose(loss_not_compiled, loss_compiled, rtol=5e-2, atol=5e-3)
    np.testing.assert_allclose(
        weight_fc1_not_compiled, weight_fc1_compiled, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        weight_fc2_not_compiled, weight_fc2_compiled, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        weight_fc3_not_compiled, weight_fc3_compiled, rtol=5e-2, atol=5e-3
    )


def test_mnist_eval_mode(device: str):
    """Test evaluation mode with torch.no_grad() context."""
    model = SimpleNet().to(device)
    model.eval()

    x = torch.randn(16, 1, 28, 28).to(device)

    with torch.no_grad():
        output = model(x)

    assert output.shape == (16, 10)
    assert output.requires_grad is False


@pytest.mark.skip(reason="CrossEntropyLoss uses unsupported operations")
def test_mnist_cross_entropy_loss(conf: Conf):
    """Test CrossEntropyLoss used in MNIST training."""

    def fn(predictions, targets):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(predictions, targets)

    predictions = torch.randn(8, 10)
    targets = torch.randint(0, 10, (8,))
    check_outputs(fn, conf, [predictions, targets])


@pytest.mark.skip(reason="argmax and eq operations not fully supported")
def test_mnist_accuracy_computation(conf: Conf):
    """Test accuracy computation operations (argmax and eq)."""

    def fn(output, target):
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum()
        return correct

    output = torch.randn(8, 10)
    target = torch.randint(0, 10, (8,))
    check_outputs(fn, conf, [output, target])


@pytest.mark.skip(reason="view operation not fully supported in all contexts")
def test_mnist_flatten_operation(conf: Conf):
    """Test MNIST-style flatten operation (view/reshape)."""

    def fn(x):
        return x.view(-1, 784)

    x = torch.randn(8, 1, 28, 28)
    check_outputs(fn, conf, [x])


def test_mnist_batch_handling():
    """Test SimpleNet handles different batch sizes correctly."""
    model = SimpleNet()

    batch_sizes = [1, 4, 8, 16, 32, 64]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1, 28, 28)
        output = model(x)
        assert output.shape == (batch_size, 10), (
            f"Batch size {batch_size}: expected shape ({batch_size}, 10), "
            f"got {output.shape}"
        )


@pytest.mark.skip(reason="ReLU in isolation not fully supported in all contexts")
def test_mnist_relu_activation(conf: Conf):
    """Test ReLU activation function used in MNIST model."""

    def fn(x):
        return F.relu(x)

    x = torch.randn(128)
    check_outputs(fn, conf, [x])


@pytest.mark.skip(reason="Uses unsupported operations")
def test_mnist_compiled_vs_eager(device: str):
    """Compare compiled vs non-compiled SimpleNet outputs."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)

    # Non-compiled output
    model.eval()
    with torch.no_grad():
        output_eager = model(x)

    # Compiled output
    compiled_model = torch.compile(model, backend=max_backend)
    with torch.no_grad():
        output_compiled = compiled_model(x)

    # Verify outputs match
    torch.testing.assert_close(output_eager, output_compiled, rtol=5e-2, atol=5e-3)


@pytest.mark.skip(
    reason="Linear layer in isolation not fully supported in all contexts"
)
def test_mnist_linear_layers(conf: Conf):
    """Test linear layers used in MNIST architecture."""

    def fn(x):
        linear = torch.nn.Linear(784, 128)
        return linear(x)

    x = torch.randn(8, 784)
    check_outputs(fn, conf, [x])


@pytest.mark.skip(reason="Uses unsupported operations")
def test_mnist_multi_layer_forward(conf: Conf):
    """Test multi-layer forward pass (simulating MNIST architecture)."""

    def fn(x):
        fc1 = torch.nn.Linear(784, 128)
        fc2 = torch.nn.Linear(128, 64)
        fc3 = torch.nn.Linear(64, 10)

        x = x.view(-1, 784)
        x = F.relu(fc1(x))
        x = F.relu(fc2(x))
        x = fc3(x)
        return x

    x = torch.randn(4, 1, 28, 28)
    check_outputs(fn, conf, [x])


@pytest.mark.skip(reason="argmax not fully supported in all contexts")
def test_mnist_prediction_extraction(conf: Conf):
    """Test prediction extraction using argmax (as used in eval)."""

    def fn(output):
        return output.argmax(dim=1, keepdim=True)

    output = torch.randn(8, 10)
    check_outputs(fn, conf, [output])


@pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
def test_mnist_forward_different_batch_sizes(device: str, batch_size: int):
    """Test forward pass with various batch sizes."""
    model = SimpleNet().to(device)
    x = torch.randn(batch_size, 1, 28, 28).to(device)

    output = model(x)

    assert output.shape == (batch_size, 10)


def test_mnist_loss_backward(device: str):
    """Test that loss.backward() works correctly."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)
    y = torch.randint(0, 10, (8,)).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    # Verify gradients are computed
    for param in model.parameters():
        assert param.grad is not None, "Gradients were not computed"
        assert not torch.all(param.grad == 0), "All gradients are zero"


def test_mnist_optimizer_updates_weights(device: str):
    """Test that SGD optimizer updates model weights."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)
    y = torch.randint(0, 10, (8,)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # Store initial weights
    initial_fc1_weight = model.fc1.weight.data.clone()

    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    # Verify weights changed
    assert not torch.equal(initial_fc1_weight, model.fc1.weight.data), (
        "Weights were not updated by optimizer"
    )


def test_mnist_zero_grad(device: str):
    """Test that optimizer.zero_grad() clears gradients."""
    model = SimpleNet().to(device)
    x = torch.randn(8, 1, 28, 28).to(device)
    y = torch.randint(0, 10, (8,)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # First forward-backward pass
    output = model(x)
    loss = criterion(output, y)
    loss.backward()

    # Clear gradients
    optimizer.zero_grad()

    # Verify all gradients are zero or None
    for param in model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0), "Gradients were not zeroed"
