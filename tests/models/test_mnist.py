"""
Comprehensive unit tests for MNIST SimpleNet model.

This module provides thorough testing of the MNIST model implementation,
including forward/backward passes, compilation, training steps, and
integration with the MAX backend.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from torch_max_backend import max_backend
from torch_max_backend.testing import check_functions_are_equivalent


class SimpleNet(nn.Module):
    """Simple feedforward neural network for MNIST classification."""

    def __init__(self):
        super().__init__()
        # Input: 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)

        # Hidden layers with RELU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        return x


def train_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """Train for one epoch.

    Args:
        model: The neural network model
        device: Device to run training on (cpu/cuda)
        train_loader: DataLoader for training data
        optimizer: Optimizer for updating weights
        criterion: Loss function
        epoch: Current epoch number

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def evaluate(model, device, test_loader, criterion):
    """Evaluate the model on test data.

    Args:
        model: The neural network model
        device: Device to run evaluation on (cpu/cuda)
        test_loader: DataLoader for test data
        criterion: Loss function

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


class TestMNISTForwardPass:
    """Tests for MNIST model forward pass."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64])
    def test_forward_output_shape(self, batch_size: int, device: str):
        """Test forward pass produces correct output shape for various batch sizes."""
        model = SimpleNet().to(device)
        x = torch.randn(batch_size, 1, 28, 28).to(device)

        output = model(x)

        assert output.shape == (batch_size, 10), (
            f"Expected shape ({batch_size}, 10), got {output.shape}"
        )
        assert output.device.type == torch.device(device).type

    def test_forward_output_type(self, device: str):
        """Test that forward pass returns correct tensor type."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)

        output = model(x)

        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32

    def test_forward_deterministic(self, device: str):
        """Test that forward pass is deterministic for same input."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(8, 1, 28, 28).to(device)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_forward_different_inputs_different_outputs(self, device: str):
        """Test that different inputs produce different outputs."""
        model = SimpleNet().to(device)
        model.eval()

        x1 = torch.randn(8, 1, 28, 28).to(device)
        x2 = torch.randn(8, 1, 28, 28).to(device)

        with torch.no_grad():
            output1 = model(x1)
            output2 = model(x2)

        # Outputs should be different
        assert not torch.allclose(output1, output2)

    def test_forward_with_zero_input(self, device: str):
        """Test forward pass with zero input."""
        model = SimpleNet().to(device)
        x = torch.zeros(8, 1, 28, 28).to(device)

        output = model(x)

        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_ones_input(self, device: str):
        """Test forward pass with ones input."""
        model = SimpleNet().to(device)
        x = torch.ones(8, 1, 28, 28).to(device)

        output = model(x)

        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMNISTBackwardPass:
    """Tests for MNIST model backward pass and gradients."""

    def test_backward_computes_gradients(self, device: str):
        """Test that backward pass computes gradients."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        criterion = nn.CrossEntropyLoss()
        model.train()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Verify gradients are computed for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient not computed for {name}"
            assert not torch.all(param.grad == 0), f"All gradients are zero for {name}"

    def test_gradient_flow_through_layers(self, device: str):
        """Test that gradients flow through all layers."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        criterion = nn.CrossEntropyLoss()
        model.train()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Check gradients exist for each layer
        assert model.fc1.weight.grad is not None
        assert model.fc1.bias.grad is not None
        assert model.fc2.weight.grad is not None
        assert model.fc2.bias.grad is not None
        assert model.fc3.weight.grad is not None
        assert model.fc3.bias.grad is not None

    def test_zero_grad_clears_gradients(self, device: str):
        """Test that optimizer.zero_grad() clears gradients."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

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


class TestMNISTCompilation:
    """Tests for MNIST model compilation with MAX backend."""

    def test_model_compiles_successfully(self, device: str):
        """Test that SimpleNet compiles successfully with max_backend."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)

        check_functions_are_equivalent(model, device, [x])

    def test_compiled_no_graph_breaks(self, device: str):
        """Test that SimpleNet compiles without graph breaks (fullgraph=True)."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)

        explanation = torch._dynamo.explain(model)(x)

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}"
        )
        assert explanation.graph_count == 1, (
            f"Expected 1 graph, got {explanation.graph_count}"
        )

    def test_compiled_model_forward(self, device: str):
        """Test that compiled model produces same output as eager."""
        model = SimpleNet().to(device)
        model.eval()
        x = torch.randn(8, 1, 28, 28).to(device)

        # Eager execution
        with torch.no_grad():
            output_eager = model(x)

        # Compiled execution
        compiled_model = torch.compile(model, backend=max_backend, fullgraph=True)
        with torch.no_grad():
            output_compiled = compiled_model(x)

        # Verify outputs match
        torch.testing.assert_close(output_eager, output_compiled, rtol=1e-4, atol=1e-4)

    def test_compiled_model_multiple_calls(self, device: str):
        """Test that compiled model can be called multiple times."""
        model = SimpleNet().to(device)
        compiled_model = torch.compile(model, backend=max_backend, fullgraph=True)

        x1 = torch.randn(8, 1, 28, 28).to(device)
        x2 = torch.randn(4, 1, 28, 28).to(device)

        # First call
        output1 = compiled_model(x1)
        assert output1.shape == (8, 10)

        # Second call with different batch size
        output2 = compiled_model(x2)
        assert output2.shape == (4, 10)


class TestMNISTTraining:
    """Tests for MNIST model training functionality."""

    def test_optimizer_updates_weights(self, device: str):
        """Test that SGD optimizer updates model weights."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Store initial weights
        initial_fc1_weight = model.fc1.weight.data.clone()
        initial_fc2_weight = model.fc2.weight.data.clone()
        initial_fc3_weight = model.fc3.weight.data.clone()

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Verify weights changed
        assert not torch.equal(initial_fc1_weight, model.fc1.weight.data), (
            "FC1 weights were not updated"
        )
        assert not torch.equal(initial_fc2_weight, model.fc2.weight.data), (
            "FC2 weights were not updated"
        )
        assert not torch.equal(initial_fc3_weight, model.fc3.weight.data), (
            "FC3 weights were not updated"
        )

    def test_loss_decreases_with_training(self, device: str):
        """Test that loss decreases over multiple training steps."""
        model = SimpleNet().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # Fixed input/output for reproducibility
        torch.manual_seed(42)
        x = torch.randn(32, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (32,)).to(device)

        losses = []
        for _ in range(5):
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_model_overfits_single_batch(self, device: str):
        """Test that model can overfit a single batch (sanity check)."""
        model = SimpleNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Single batch
        torch.manual_seed(42)
        x = torch.randn(16, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (16,)).to(device)

        # Train for many iterations
        for _ in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Check final accuracy is high
        model.eval()
        with torch.no_grad():
            output = model(x)
            predictions = output.argmax(dim=1)
            accuracy = (predictions == y).float().mean()

        # Should achieve high accuracy on single batch
        assert accuracy > 0.8, (
            f"Failed to overfit single batch: accuracy={accuracy:.2f}"
        )


class TestMNISTEvaluation:
    """Tests for MNIST model evaluation functionality."""

    def test_eval_mode_no_grad(self, device: str):
        """Test evaluation mode with torch.no_grad() context."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(16, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (16, 10)
        assert output.requires_grad is False

    def test_eval_mode_consistent_predictions(self, device: str):
        """Test that predictions are consistent in eval mode."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(16, 1, 28, 28).to(device)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_predictions_in_valid_range(self, device: str):
        """Test that model predictions are valid logits."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(16, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Check that each sample has 10 class scores
        assert output.shape[1] == 10


class TestMNISTInputVariations:
    """Tests for MNIST model with various input variations."""

    def test_normalized_input(self, device: str):
        """Test with normalized MNIST input (mean=0.1307, std=0.3081)."""
        model = SimpleNet().to(device)

        # Simulate normalized MNIST data
        x = torch.randn(8, 1, 28, 28).to(device) * 0.3081 + 0.1307

        output = model(x)
        assert output.shape == (8, 10)

    def test_unnormalized_input(self, device: str):
        """Test with unnormalized input [0, 1] range."""
        model = SimpleNet().to(device)

        # Simulate unnormalized MNIST data [0, 1]
        x = torch.rand(8, 1, 28, 28).to(device)

        output = model(x)
        assert output.shape == (8, 10)

    def test_negative_input(self, device: str):
        """Test model handles negative input values."""
        model = SimpleNet().to(device)

        x = torch.randn(8, 1, 28, 28).to(device) - 2.0

        output = model(x)
        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()

    def test_large_input_values(self, device: str):
        """Test model handles large input values."""
        model = SimpleNet().to(device)

        x = torch.randn(8, 1, 28, 28).to(device) * 10.0

        output = model(x)
        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()


class TestMNISTEdgeCases:
    """Tests for MNIST model edge cases."""

    def test_single_sample_inference(self, device: str):
        """Test inference on a single sample."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(1, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 10)

    def test_large_batch_inference(self, device: str):
        """Test inference on a large batch."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(128, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (128, 10)

    def test_model_state_dict_save_load(self, device: str):
        """Test saving and loading model state dict."""
        model1 = SimpleNet().to(device)
        model2 = SimpleNet().to(device)

        # Set random weights for model1
        torch.manual_seed(42)
        for param in model1.parameters():
            param.data.normal_()

        # Save and load state dict
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Verify weights are identical
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_model_dtype_consistency(self, device: str):
        """Test that model maintains dtype consistency."""
        model = SimpleNet().to(device)

        # All parameters should be float32
        for param in model.parameters():
            assert param.dtype == torch.float32


class TestMNISTRealData:
    """Tests using real MNIST dataset."""

    @pytest.fixture(scope="class")
    def mnist_data_loaders(self):
        """Fixture to load MNIST dataset for testing."""
        data_dir = Path.home() / ".pytorch" / "datasets" / "mnist"

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

        # Load training dataset (use subset for faster testing)
        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform
        )

        # Load test dataset
        test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

        # Use small batch size and subset for testing
        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=True, num_workers=0
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1000, shuffle=False, num_workers=0
        )

        return train_loader, test_loader

    def test_train_one_epoch_with_real_data(self, device: str, mnist_data_loaders):
        """Test training for one epoch with real MNIST data."""
        train_loader, test_loader = mnist_data_loaders

        model = SimpleNet().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch=1
        )

        # Verify metrics are reasonable
        assert train_loss > 0, "Training loss should be positive"
        assert 0 <= train_acc <= 100, f"Training accuracy {train_acc} out of range"

        # After one epoch, we should have some reasonable accuracy
        assert train_acc > 10, f"Training accuracy too low: {train_acc:.2f}%"

    def test_evaluate_with_real_data(self, device: str, mnist_data_loaders):
        """Test evaluation on real MNIST data."""
        train_loader, test_loader = mnist_data_loaders

        model = SimpleNet().to(device)
        criterion = nn.CrossEntropyLoss()

        # Evaluate untrained model
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        # Verify metrics are reasonable
        assert test_loss > 0, "Test loss should be positive"
        assert 0 <= test_acc <= 100, f"Test accuracy {test_acc} out of range"

        # Untrained model should have ~10% accuracy (random guessing)
        assert 5 <= test_acc <= 20, (
            f"Untrained model accuracy {test_acc:.2f}% unexpected"
        )

    def test_training_improves_accuracy(self, device: str, mnist_data_loaders):
        """Test that training improves accuracy on real data."""
        train_loader, test_loader = mnist_data_loaders

        model = SimpleNet().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Initial evaluation
        initial_loss, initial_acc = evaluate(model, device, test_loader, criterion)

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, device, train_loader, optimizer, criterion, epoch=1
        )

        # Evaluate after training
        final_loss, final_acc = evaluate(model, device, test_loader, criterion)

        # Accuracy should improve after training
        assert final_acc > initial_acc, (
            f"Accuracy did not improve: {initial_acc:.2f}% -> {final_acc:.2f}%"
        )

        # Loss should decrease
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )

    def test_compiled_model_train_epoch(self, device: str, mnist_data_loaders):
        """Test training with compiled model on real data."""
        train_loader, test_loader = mnist_data_loaders

        # Regular model
        model_eager = SimpleNet().to(device)
        optimizer_eager = torch.optim.SGD(
            model_eager.parameters(), lr=0.01, momentum=0.9
        )
        criterion = nn.CrossEntropyLoss()

        # Compiled model
        model_compiled = SimpleNet().to(device)
        model_compiled = torch.compile(model_compiled, backend=max_backend)
        optimizer_compiled = torch.optim.SGD(
            model_compiled.parameters(), lr=0.01, momentum=0.9
        )

        # Train both models for one epoch on a small subset
        # Use only first 5 batches for speed
        train_subset = []
        for i, batch in enumerate(train_loader):
            if i >= 5:
                break
            train_subset.append(batch)

        # Train eager model
        model_eager.train()
        for data, target in train_subset:
            data, target = data.to(device), target.to(device)
            optimizer_eager.zero_grad()
            output = model_eager(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_eager.step()

        # Train compiled model
        model_compiled.train()
        for data, target in train_subset:
            data, target = data.to(device), target.to(device)
            optimizer_compiled.zero_grad()
            output = model_compiled(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer_compiled.step()

        # Both models should produce valid outputs
        test_data, _ = next(iter(test_loader))
        test_data = test_data.to(device)

        model_eager.eval()
        model_compiled.eval()
        with torch.no_grad():
            output_eager = model_eager(test_data)
            output_compiled = model_compiled(test_data)

        assert output_eager.shape == output_compiled.shape
        assert not torch.isnan(output_eager).any()
        assert not torch.isnan(output_compiled).any()
