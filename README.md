# ML-Workout
## Assignment 1:
## Prototype-Based Classification for Animals with Attributes Dataset
### Overview
This repository contains the implementation of prototype-based classification methods for the "Animals with Attributes" dataset (version 1 - AwA v1). The goal is to predict the class of unseen animals using two different methods, taking into account a unique challenge: the training set only contains examples from 40 "seen classes," leaving 10 "unseen classes" without any training data.

### Dataset
The dataset consists of images of animals, with each input representing an image, and the output being the class (animal category). There are a total of 50 classes, and each input has 4096 features, pre-extracted by a deep learning model.

### Problem Statement
The challenge is to compute the mean of the 10 unseen classes for prototype-based classification. Class attribute vectors (ac ∈ R^85) are provided for each class (both seen and unseen). Each class attribute vector contains 85 binary-valued attributes representing the class, providing information about its characteristics.

### Methods
#### Method 1
Model the mean (μc ∈ R4096) of each unseen class (c = 41,...,50) as a convex combination of the means of the 40 seen classes. The similarity vector sc is defined as the inner product of the class attribute vectors. Normalize the similarity vector, and compute the means for the unseen classes.

#### Method 2
Train a linear model to predict the mean (μc ∈ R4096) of any class using its class attribute vector ac ∈ R85. The linear model is trained using {(ac, μc)}40 as training data. The solution to the multi-output regression problem is computed, and the means for the unseen classes are obtained. The implementation involves trying different values of λ (0.01, 0.1, 1, 10, 20, 50, 100) to find the best test set accuracy.

## Assignment 2:
## Kernel Ridge Regression and Landmark-Ridge

### Overview
This repository contains the implementation of Kernel Ridge Regression and Landmark-Ridge for a given dataset. The models are trained and tested on the provided dataset, split into training and test sets. The RBF kernel with bandwidth parameter γ = 0.1 is used for both models.

### Dataset
The training and test datasets are available in the files ridgetrain.txt and ridgetest.txt. Each line in the files represents an example, with the first number being the input (a single feature), and the second number being the output.

### Kernel Ridge Regression
Training
For kernel ridge regression, the model is trained with the regularization hyperparameter λ = 0.1, and predictions are made for the test data. The process is repeated for λ = 1, 10, 100.

Testing and Visualization
All test inputs and their true outputs are plotted on a 2D graph in blue color. The corresponding predicted outputs are also plotted in red color. This visualization is done for each value of λ. The root-mean-squared-error (RMSE) on the test data is reported for each case.

### Landmark-Ridge
Landmark-Based Features
For Landmark-Ridge, landmark-based features are extracted using the RBF kernel. Data with these features are then used to train a linear ridge regression model. The regularization hyperparameter is fixed at λ = 0.1. The process is repeated for different numbers of uniformly randomly chosen landmark points (L = 2, 5, 20, 50, 100) from the training set.

### Testing and Visualization
Results for each case are plotted similarly to Kernel Ridge Regression. The RMSE for each case is reported, and observations from the plots are discussed. This also addresses the question of what value of L seems to be good enough.

