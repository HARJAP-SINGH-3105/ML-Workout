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

## K-Means Clustering with Feature Transformation
### Overview
This repository contains the implementation of the K-means clustering algorithm applied to a provided toy dataset (kmeans_data.txt) consisting of points in two dimensions. The dataset has two clusters (K = 2), but the standard K-means may not work well due to non-spherical and non-linearly separable clusters. Two approaches are considered to handle this issue:

## 1. Hand-crafted Features
### Data Exploration
Before proposing a feature transformation, the original data is plotted in 2D using a scatter plot to identify potential transformations that might work.

### Feature Transformation
A feature transformation is proposed to make the clusters more amenable to K-means clustering. Apply the K-means algorithm to this transformed version of the data to verify if the transformation works.

### Clustering Results Visualization
Plot the obtained clustering results by showing points in cluster 1 in red and points in cluster 2 in green, all in the original 2D space.

## 2. Landmark-Based Approach with Kernels
### Landmark-Based Features
Utilize the landmark-based approach to extract features. The RBF kernel (γ = 0.1) is employed for this purpose. For experimentation, L = 1 landmark point is randomly chosen from the dataset. Implement the standard K-means algorithm on these features.

### Multiple Runs with Different Landmarks
Perform 10 runs of the algorithm, each time with a different randomly chosen landmark point. Visualize the clustering results for each run, indicating the chosen landmark point in blue. Justify why correct clustering is achieved in some cases and not-so-correct clustering in other cases

## Visualization of MNIST Digits Using PCA and t-SNE
### Overview
This repository contains code to visualize a subset of MNIST digits data using Principal Component Analysis (PCA) and t-distributed Stochastic Neighbor Embedding (t-SNE). The dataset comprises 10,000 examples from digits 0-9 (10 classes).

### Dataset
The dataset is provided as a pickle file, where the fields X and Y contain the digit input features (784 dimensional) and labels (0-9), respectively, for the 10,000 examples.

### Visualization
### PCA (Principal Component Analysis)
Implement PCA to project the input features to two dimensions.
Visualize the projected data in a scatter plot, coloring each point based on its class.

### t-SNE (t-distributed Stochastic Neighbor Embedding)
Implement t-SNE to project the input features to two dimensions.
Visualize the projected data in a scatter plot, coloring each point based on its class.

### Implementation
The implementation includes Python code using existing library functions, such as those from scikit-learn, for PCA and t-SNE. The visualization plots are created for both methods, highlighting the different classes with distinct colors


