import logging
from functools import reduce
from typing import List, Tuple

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from src.datasets.retrieval import DataPoint

from .model import Model

logger = logging.getLogger(__name__)


# ------------------------------------ MODEL -----------------------------------


class NaiveBayes(Model):
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.log_priors = None
        self.log_likelihoods = None
        self.classes = None
        self.vectorizer = CountVectorizer()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Naive Bayes model according to X, y.

        Parameters:
        - X: numpy.ndarray, shape (n_samples, n_features), Training data with binary or count features
        - y: numpy.ndarray, shape (n_samples,), Target values

        The method modifies the model's log_priors and log_likelihoods attributes based on the training data.
        """
        # Calculate class prior probabilities: P(y)
        self.classes = np.unique(y)
        class_counts = np.bincount(y, minlength=len(self.classes))
        self.log_priors = np.log(class_counts) - np.log(class_counts.sum())

        # Calculate feature likelihoods: P(X|y)
        feature_counts = np.zeros((len(self.classes), X.shape[1]))

        for i, c in enumerate(self.classes):
            feature_counts[i, :] = X[y == c].sum(axis=0)

        # Laplace smoothing
        feature_counts += self.alpha
        feature_sum = feature_counts.sum(axis=1, keepdims=True)

        self.log_likelihoods = np.log(feature_counts) - np.log(feature_sum)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the given input features using the Naive Bayes model.

        Parameters:
        - X: numpy.ndarray, shape (n_samples, n_features), Feature matrix for which to predict class labels

        Returns:
        - predictions: numpy.ndarray, The predicted class labels.
        """
        log_posteriors = X @ self.log_likelihoods.T + self.log_priors
        return log_posteriors.argmax(axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate the accuracy of the Naive Bayes model on the given test data and labels.

        Parameters:
        - X: numpy.ndarray, Feature matrix for the test set
        - y: numpy.ndarray, True class labels for the test set

        Returns:
        - accuracy: float, The accuracy of the model on the test set.
        """
        return (self.predict(X) == y).mean()


# -------------------------------- PREPROCESSING -------------------------------


class NaiveBayesDataProcessor:
    def __init__(
        self, data: List[List[DataPoint]], representation_type="count", vectorizer_params={}
    ):
        self.data = data
        self.representation_type = representation_type
        self.vectorizer_params = vectorizer_params
        self.vectorizer = None
        self.transformer = None

    def preprocess(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Preprocess data by converting text into numerical features using different representation types.

        Parameters:
        - data: List of lists where each sublist contains DataPoint objects for a dataset partition (train, validation, test).
        - representation_type: The type of feature representation to use: 'count', 'binary', or 'tfidf'.
        - vectorizer_params: Parameters to pass to the CountVectorizer.

        Returns:
        - A list of tuples, where each tuple contains feature vectors (X) and labels (y) for a dataset partition.
        """

        # Flatten the list of lists of DataPoint objects into a single list of DataPoint objects.
        all_data = reduce(lambda x, y: x + y, self.data)

        # Extract texts and labels from DataPoint objects.
        texts = [datapoint.text for datapoint in all_data]
        labels = [datapoint.label for datapoint in all_data]

        # Update vectorizer parameters based on the representation type.
        if self.representation_type == "binary":
            self.vectorizer_params["binary"] = True

        # Initialize the CountVectorizer with the provided parameters.
        self.vectorizer = CountVectorizer(**self.vectorizer_params)

        # Transform the texts to a feature matrix using the vectorizer.
        X = self.vectorizer.fit_transform(texts)

        # If 'tfidf' representation is selected, apply the TfidfTransformer.
        if self.representation_type == "tfidf":
            transformer = TfidfTransformer()
            X = transformer.fit_transform(X)

        # Calculate the cumulative sum for partition sizes, which will serve as split indices.
        partition_sizes = [len(part) for part in self.data]
        partition_cumsum = np.cumsum(partition_sizes[:-1])

        # Split the feature matrix and label array according to partitions.
        try:
            if sparse.issparse(X):
                X_partitions = [
                    X[partition_cumsum[i - 1] if i > 0 else 0 : partition_cumsum[i]]
                    for i in range(len(partition_cumsum))
                ]
                X_partitions.append(X[partition_cumsum[-1] :])
            else:
                X = X.toarray()
                X_partitions = np.split(X, partition_cumsum)

            y_partitions = np.split(np.array(labels), partition_cumsum)
            y_partitions.append(labels[partition_cumsum[-1] :])

        except Exception as e:
            logger.error(f"Failed to split data: {e}")
            raise e

        return list(zip(X_partitions, y_partitions))

    def get_feature_names(self):
        """
        Returns the feature names extracted by the CountVectorizer.

        Returns:
        - A list of feature names (words).
        """
        if self.vectorizer:
            return self.vectorizer.get_feature_names_out()
        else:
            logger.error("Vectorizer not initialized.")
            return []
