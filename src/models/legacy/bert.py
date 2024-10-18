import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
)

from src.datasets.retrieval import DataPoint
from src.models.base import Model

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# TODO: Add softmax classifier


class BERT(Model):
    def __init__(self, model_name, num_labels, batch_size, device=DEVICE):
        super().__init__()
        if model_name == "bert-base-uncased":
            self.model = BertForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                output_attentions=True,
                output_hidden_states=False,
            )
        elif model_name == "Vasanth/bert-base-uncased-finetuned-emotion":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else:
            raise ValueError(f'The model "{model_name}" is not supported.')

        self.model_name = model_name
        self.device = device
        self.model.to(self.device)
        self.batch_size = batch_size

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels
        )

    def fit(self, train_dataloader, optimizer: Optimizer, scheduler: LRScheduler, epochs: int):
        losses = []
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()
                outputs = self.forward(b_input_ids, b_input_mask, b_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                losses.append(loss.item())

                logger.info(f"Epoch {epoch} - Step {step} - Loss {loss.item()}")
            if scheduler:
                scheduler.step(total_loss)

        return losses

    def predict(self, dataloader):
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs.logits
                predictions.append(logits)

        return predictions

    def evaluate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == b_labels).sum().item()
                total += b_labels.size(0)

        accuracy = correct / total
        return {"accuracy": accuracy}

    def save(self, save_path: Path | str):
        # TODO:
        # if file_name is None:
        #     file_name = self.__class__.__name__
        # file_path = f"{dir_path}/{file_name}_weights.{ext}"
        torch.save(self.model.state_dict(), save_path)

    @classmethod
    def load(
        cls,
        load_path: Path | str,
        model_name: str,
        num_labels: int,
        batch_size: int = 64,
        device=DEVICE,
    ):
        # TODO:
        # if file_name is None:
        #     file_name = cls.__name__
        # file_path = f"{dir_path}/{file_name}_weights.{ext}"

        model = cls(model_name, num_labels, batch_size, device)
        model.model.load_state_dict(torch.load(load_path))
        # model.model.to(device)
        return model


class DefaultBERT(BERT):
    """
    Standard pretrained BERT.
    """

    def __init__(self, batch_size, device=DEVICE):
        super().__init__("bert-base-uncased", num_labels=6, batch_size=batch_size, device=device)


class PrefinetunedBERT(BERT):
    """
    BERT finetuned on emotion data from Huggingface.
    """

    def __init__(self, batch_size, device=DEVICE):
        super().__init__(
            "Vasanth/bert-base-uncased-finetuned-emotion",
            num_labels=6,
            batch_size=batch_size,
            device=device,
        )


class InitBERT(BERT):
    """
    Standard pre-trained BERT but with weights and biases re-initalized.
    """

    def __init__(self, batch_size, device=DEVICE):
        super().__init__("bert-base-uncased", num_labels=6, batch_size=batch_size, device=device)

        self.model.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class FrozenBERT(BERT):
    """
    Standard pretrained BERT, but with all layers except the n last ones frozen.
    """

    def __init__(self, batch_size, num_layers_to_finetune=1, device=DEVICE):
        super().__init__("bert-base-uncased", num_labels=6, batch_size=batch_size, device=device)

        # Freeze all the layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the last N layers
        for layer in self.model.bert.encoder.layer[-num_layers_to_finetune:]:
            for param in layer.parameters():
                param.requires_grad = True


# -------------------------------- PREPROCESSING -------------------------------- #


class BERTDataProcessor:
    def __init__(
        self,
        data: tuple[list[DataPoint], list[DataPoint], list[DataPoint]],
        model_name: BERT = None,
        batch_size: int = 64,
        max_length: int = 64,
    ):
        """
        Initializes the BERTDataProcessor with the dataset, model name, maximum sequence length, and batch size.

        Parameters:
        - data: list of lists containing DataPoint objects for each dataset partition (train, validation, test).
        - model_name: The name of the BERT model to be used for tokenization.
        - max_length: The maximum length of the sequences after tokenization.
        - batch_size: The size of the batches for the DataLoader.
        """
        self.data = data
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = self.get_tokenizer(model_name=model_name)
        self.model_name = model_name

    @staticmethod
    def get_tokenizer(model_name):
        if model_name is None:
            return None
        elif model_name == PrefinetunedBERT:
            return AutoTokenizer.from_pretrained("Vasanth/bert-base-uncased-finetuned-emotion")
        elif model_name == BERT:
            return torch.hub.load(
                "huggingface/pytorch-transformers", "tokenizer", "bert-base-uncased"
            )
        else:
            return torch.hub.load(
                "huggingface/pytorch-transformers", "tokenizer", "bert-base-uncased"
            )

    def preprocess(self):
        """
        Processes the data by tokenizing and creating DataLoader objects for each dataset partition.

        Returns:
        - A list of DataLoader objects for each dataset partition.
        """
        return [self._create_dataloader(partition) for partition in self.data]

    def _create_dataloader(self, data_partition: list[DataPoint]):
        """
        Tokenizes a data partition and creates a DataLoader.

        Parameters:
        - data_partition: A list of DataPoint objects to be tokenized and loaded into a DataLoader.

        Returns:
        - A DataLoader for the processed data partition.
        """
        inputs = self._tokenize_data(data_partition)
        dataset = TensorDataset(*inputs)
        sampler = (
            RandomSampler(dataset) if self.model_name == "train" else SequentialSampler(dataset)
        )
        return DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

    def _tokenize_data(self, data_partition: list[DataPoint]):
        """
        Tokenizes the texts in a data partition.

        Parameters:
        - data_partition: A list of DataPoint objects to be tokenized.

        Returns:
        - tuple of lists containing tokenized input IDs, attention masks, and labels.
        """
        input_ids, attention_masks, labels = [], [], []

        for datapoint in data_partition:
            encoded_dict = self.tokenizer.encode_plus(
                datapoint.text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids.append(encoded_dict["input_ids"][0])
            attention_masks.append(encoded_dict["attention_mask"][0])
            labels.append(datapoint.label)

        return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(labels)
