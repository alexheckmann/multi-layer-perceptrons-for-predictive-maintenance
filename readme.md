# Multi-Layer Perceptrons for Predictive Maintenance

This machine learning project aims to reduce operational costs of running manufacturing systems by
preemptively detecting machine failure. The project is based
on
the [Machine Predictive Maintenance Classification Dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
and compares PyTorch and Matlab implementations of Multi-Layer Perceptrons (MLP).

For the full project report, feel free to read the summary, also available in this repository.

## TLDR

Predictive maintenance is a game-changer in the manufacturing industry. This research paper explores the problem of
multivariate time series classification for
operational data from a real-world software system. It's not just about saving money, it's about extending the lifespan
of machinery and enhancing safety by spotting potential failures before they cause accidents. The study compares the
performance of two different Multi-Layer Perceptron implementations: PyTorch and Matlab. Holdout validation is
employed to test the accuracy of the models. The results demonstrate the Python interpreter's overhead, showing a
considerable amount of additional training time for machine learning in Python with more fine-tuning required to achieve
good results. Consequently, the study concludes that
Matlab's MLP is the superior model in terms of training time
and validation accuracy.

This project was conducted as the capstone project for the Deep Learning course at City, University of London,
achieving a distinction.

## The comparison

In this project, we compare the performance of two implementations of MLP.

Multilayer Perceptron (MLP) is a type of feedforward neural network that is commonly used for classification tasks.

## The dataset

The dataset was taken from Kaggle and is provided by HTW Berlin. It contains a synthetic dataset that reflects real
predictive maintenance from the industry.
For this project, the whole dataset containing 10000 rows was used.
The dataset contains 8 columns, with the first 6 columns being features and the last two columns labels for binary and
multi-class classification.

The following steps were taken to preprocess the data for our experiments:

1) **Initial Data Analysis**. Assessing the distribution of predictors and check the target variable for imbalance.
2) **Feature selection**. Feature selection of relevant predictors for the model based on domain knowledge.
3) **Encoding categorical variables**. We transformed the categorical variables into one-hot encoded variables.
4) **Splitting**. The data was split into training and test sets using a 70-30 ratio.
5) **Oversampling**. Oversampling was performed on the training set using SMOTE-NC to balance the target variable.
   Undersampling was not feasible due to the massive imbalance in the target variable of 237 to 9663.
6) **Normalization**. Scaling the numerical features to a range of 0 to 1 to prevent issues of sensitivity to the
   magnitude of scale.

## The models

The best performing model was achieved with a model with 2 hidden neurons, trained for 50 epochs, and a learning rate of
0.3. ReLU was used as the activation function for the hidden layers, while the output layer used a sigmoid activation.

## Results

The results revealed several key findings.

Firstly, the Matlab implementation of MLP outperformed the PyTorch implementation in terms of training time.
The Matlab implementation took 52ms to train, while the PyTorch implementation took 250ms, a speed difference of 5x.

Secondly, the Matlab implementation achieved a validation accuracy of 0.99999, while the PyTorch implementation achieved a
validation accuracy of only 0.966. This shows that there is a significant difference in the implementation of the two
models.

## Remarks

The project was optimized for running on GPU in Google Colab.

The data could not be included in this repository due to its size. However, the data can be downloaded from the
the link provided above and should be put into a dedicated "data" folder to seemlessly run the code.
