# Linear Regression and Logistic Regression with GD vs SGD vs Analytical Solution

This experiment was conducted as part of COMP551. We explore the performance of linear regression and logistic regression models across the Boston Housing dataset and Wine dataset, respectively. We observe the impact of hyperparameters and other design choices, and compare GD, SGD and Analytical solutions.

## Results

Overfitting on the wine dataset.

Importance of hyperparameters, esp. batch size and learning rate. For Boston, batch size=64 and lr=0.1 were best, while for Wine, batch size=8 and lr=0.01 were best.

Gaussian Basis Transformation worsened the Boston model (lead to overfitting). The analytical solution outperformed SGD.

TODO: Currently broken

## Contributors

Fraser Ross Lee, Jingxiang Mo and Henri Lemoine.
