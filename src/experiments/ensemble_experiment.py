from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.models.base import ModelInterface, TorchModel
from src.models.ensemble import Ensemble, EnsembleMethod, evaluate_ensemble


def train_model(
    model: TorchModel, train_loader: DataLoader, test_loader: DataLoader, epochs: int = 10
):
    if not model.is_trainable():
        acc = eval_classification(model, test_loader)["accuracy"] * 100
        return {"train_acc": [acc], "test_acc": [acc]}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {"train_acc": [], "test_acc": []}

    for epoch in range(epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        history["train_acc"].append(train_acc)

        # Testing
        acc = eval_classification(model, test_loader)["accuracy"] * 100
        history["test_acc"].append(acc)

        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% Test Acc: {acc:.2f}%")

    return history


def eval_classification(model: TorchModel, loader: DataLoader) -> dict[str, float]:
    """Evaluate classification accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return {"accuracy": correct / total}


class VerticalBiasedMLP(TorchModel):
    """MLP with filters biased towards vertical features"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Reshape each image into 28 rows of 28 pixels
        # Apply different weights to each row position
        self.row_weights = nn.Parameter(torch.randn(28) * 0.1)
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 10)
        )

    def forward(self, x):
        # x shape: [batch, 1, 28, 28]
        batch_size = x.shape[0]
        # Add row-wise importance
        x = x.view(batch_size, 28, 28) * self.row_weights.view(1, 28, 1)
        x = x.flatten(1)
        return self.fc(x)

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        return eval_classification(self, data)


class HorizontalBiasedMLP(TorchModel):
    """MLP with filters biased towards horizontal features"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # Apply different weights to each column position
        self.col_weights = nn.Parameter(torch.randn(28) * 0.1)
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32, 10)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # Add column-wise importance
        x = x.view(batch_size, 28, 28) * self.col_weights.view(1, 1, 28)
        x = x.flatten(1)
        return self.fc(x)

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        return eval_classification(self, data)


class LocalPatternCNN(TorchModel):
    """CNN focused on local patterns with small receptive field"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),  # Small kernel for local patterns
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        return self.fc(x)

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        return eval_classification(self, data)


class RandomModel(ModelInterface[Tensor, Tensor, DataLoader]):
    """Completely dumb model that just outputs random predictions."""

    def __init__(self):
        self.num_classes = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        return torch.randn(batch_size, self.num_classes).to(self.device)

    def predict(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def parameters(self):
        return []

    def train(self, mode=True):
        return self

    def eval(self):
        return self

    def fit(self, train_data: DataLoader, val_data: DataLoader | None = None) -> None:
        pass

    def evaluate(self, data: DataLoader) -> dict[str, float]:
        return eval_classification(self, data)

    def save(self, path: Path) -> None:
        pass

    @classmethod
    def load(cls, path: Path) -> "RandomModel":
        return cls()

    def is_trainable(self) -> bool:
        """Whether this model can be trained."""
        return False


def get_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    val_loader = DataLoader(test_dataset, batch_size=1000)

    return train_loader, test_loader, val_loader


def experiment():
    train_loader, test_loader, val_loader = get_data()

    models = [VerticalBiasedMLP(), HorizontalBiasedMLP(), LocalPatternCNN(), RandomModel()]
    histories = []

    print("Training individual models:")
    for i, model in enumerate(models, 1):
        print(f"\nTraining Model {i}:")
        history = train_model(model, train_loader, test_loader, epochs=1)
        histories.append(history)

    # 1. Simple Mean Ensemble
    mean_ensemble = Ensemble(models=models, method=EnsembleMethod.MEAN)
    mean_acc = evaluate_ensemble(mean_ensemble, test_loader)

    # 2. Majority Voting Ensemble
    vote_ensemble = Ensemble(models=models, method=EnsembleMethod.VOTE)
    vote_acc = evaluate_ensemble(vote_ensemble, test_loader)

    # 3. Weighted Ensemble with temperature-scaled softmax calibration
    weighted_ensemble = Ensemble(models=models, method=EnsembleMethod.WEIGHTED)
    weighted_ensemble.calibrate_weights(val_loader, method="softmax", temperature=0.5)
    weighted_acc = evaluate_ensemble(weighted_ensemble, test_loader)

    # 4. Stacking Ensemble
    stacking_ensemble = Ensemble(models=models, method=EnsembleMethod.STACKING)
    stacking_ensemble.fit_stacking(train_loader, val_loader)
    stacking_acc = evaluate_ensemble(stacking_ensemble, test_loader)

    print("\nIndividual Model Performances:")
    for i, model in enumerate(models, 1):
        acc = eval_classification(model, test_loader)["accuracy"] * 100
        print(f"Model {i} ({type(model).__name__}): {acc:.2f}%")

    print("\nEnsemble Performances:")
    print(f"Mean Ensemble: {mean_acc['accuracy']*100:.2f}%")
    print(f"Vote Ensemble: {vote_acc['accuracy']*100:.2f}%")
    print(f"Weighted Ensemble: {weighted_acc['accuracy']*100:.2f}%")
    print(f"Stacking Ensemble: {stacking_acc['accuracy']*100:.2f}%")

    print("\nCalibrated Weights (Weighted Ensemble):")
    for i, (model, weight) in enumerate(zip(models, weighted_ensemble.weights), 1):
        print(f"Model {i} ({type(model).__name__}): {weight:.3f}")

    print("\nTrying different temperatures for weighted ensemble:")
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0]
    for temp in temperatures:
        weighted_ensemble = Ensemble(models=models, method=EnsembleMethod.WEIGHTED)
        weighted_ensemble.calibrate_weights(val_loader, method="softmax", temperature=temp)
        acc = evaluate_ensemble(weighted_ensemble, test_loader)["accuracy"] * 100
        print(f"\nTemperature {temp}:")
        print(f"Accuracy: {acc:.2f}%")
        print("Weights:", [f"{w:.3f}" for w in weighted_ensemble.weights])


if __name__ == "__main__":
    experiment()
