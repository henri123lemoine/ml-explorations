from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.config import DatasetConfig, PretrainedConfig
from src.datasets.image.bicycle import create_dataloaders
from src.models.image.base import PretrainedImageClassifier
from src.settings import MODELS_PATH
from src.train import train_model, validate_model

if __name__ == "__main__":
    model_config = PretrainedConfig(
        model_name="google/mobilenet_v2_1.0_224",
        model_class=AutoModelForImageClassification,
        processor_class=AutoImageProcessor,
        num_labels=2,
        learning_rate=2e-5,
        freeze_backbone=True,
        backbone_attr="mobilenet",
        classifier_attr="classifier",
    )
    model = PretrainedImageClassifier(model_config)
    model.save(MODELS_PATH / "mobilenet_v2_1.0_224_bicycle.pt")

    # Print model parameter status
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")

    dataset_config = DatasetConfig(max_images=1000, neg_ratio=1.0, batch_size=32)

    train_loader, val_loader = create_dataloaders(processor=model.processor, config=dataset_config)

    # First validate without any training
    print("\nValidating initial model performance...")
    initial_metrics = validate_model(model, val_loader, model.config.device)
    print(f"Initial validation accuracy: {initial_metrics['accuracy']:.4f}")

    # Train with frozen backbone first
    print("\nTraining with frozen backbone...")
    trained_model = train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
    )
    trained_model.save(MODELS_PATH / "mobilenet_v2_1.0_224_bicycle.pt")
