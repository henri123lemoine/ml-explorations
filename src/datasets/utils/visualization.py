from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def plot_confusion_matrix(
    confusion_matrix: np.ndarray, class_names: list[str] = ["bicycle", "non_bicycle"]
):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def plot_misclassified_examples(
    examples: list[dict[str, Any]],
    class_names: list[str] = ["bicycle", "non_bicycle"],
    num_examples: int = 10,
):
    """Plot grid of misclassified examples"""
    if not examples:
        print("No misclassified examples found!")
        return

    n = min(num_examples, len(examples))
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(20, 8))
    axes = np.atleast_1d(axes.ravel())

    for idx, example in enumerate(examples[:n]):
        if example["image"] is None:
            continue

        # Denormalize image for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = example["image"] * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(
            f"True: {class_names[example['true_label']]}\n"
            f"Pred: {class_names[example['predicted_label']]}\n"
            f"Conf: {example['confidence']:.2f}"
        )

    plt.tight_layout()
    plt.show()
