<br />
<div align="center">
    <img src="icon.png" alt="Logo" width="120" height="120">
  </a>

  <h3 align="center">KNN Classification for Diabetes</h3>

  <p align="center">
    Intermediate Python Project
    <br />
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/othneildrew/Best-README-Template/issues/new?labels=enhancement&template=feature-request---.md">View Dataset</a>
  </p>
</div>

<!-- ABOUT THE PROJECT -->
## About The Project
This was an intermediate python project that I pursued on my own since I was interested in learning more about ML techniques. At this point, I was unfamiliar with notebooks (Jupyter Notebook, Google Colab, etc.) and wanted to use PyCharm to work on this project since I was more familiar with it through my college class. Although this project is not yet fully completed, I will be updating this repository as I continue working on this project.

## Prerequisites

Before I started coding, I installed the following required imports

Required Imports
  ```sh
import statistics
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
  ```


## Methodology

The goal of this project was to build a diabetes classification model using the K-Nearest Neighbors (KNN) algorithm. Below is a step-by-step breakdown of the methodology:

### 1. Dataset Acquisition
- The dataset was sourced from Kaggle, containing over 100,000 rows and 6+ features.
- Features included medical indicators such as glucose levels, blood pressure, BMI, age, insulin levels, and other health-related attributes.

### 2. Data Preprocessing
- **Missing Data Handling**: Checked for missing values and applied imputation techniques (e.g., mean, median) or removed incomplete rows if necessary.
- **Normalization**: Used Min-Max scaling to ensure all features had a similar range, preventing any single feature from disproportionately influencing the KNN model.
- **Feature Selection**: Identified and retained the most relevant features while removing those that did not contribute significantly to the classification process.
- **Data Splitting**: Split the dataset into training and testing sets (typically an 80-20 or 70-30 split) to evaluate model performance effectively.

### 3. Model Selection: K-Nearest Neighbors (KNN)
- Chose KNN as the classification algorithm due to its simplicity and effectiveness for pattern recognition in medical datasets.
- **Distance Metric**: Used Euclidean distance to measure similarity between data points.
- **Choosing K**: Experimented with different values of K to balance bias and variance. A small K can lead to overfitting, while a large K may cause underfitting.

### 4. Model Training and Evaluation
- **Training the Model**: Applied the KNN classifier to the training dataset.
- **Cross-Validation**: Used cross-validation techniques to assess performance and prevent overfitting.
- **Performance Metrics**: Evaluated the model using:
  - **Accuracy**: The percentage of correct predictions.
  - **Precision, Recall, and F1-Score**: Measured the effectiveness of the model in correctly classifying diabetic and non-diabetic cases.

### 5. Model Optimization
- **Hyperparameter Tuning**: Adjusted the number of neighbors (K) and experimented with different weighting strategies (e.g., giving closer neighbors more influence).
- **Final Model Selection**: Chose the K value that provided the best balance between accuracy and generalizability.

### 6. Final Model Evaluation
- Tested the final model on the test dataset to assess real-world performance.
- Compared predictions with actual values to determine accuracy and reliability.
- Ensured that the final model was effective for classifying diabetes based on the given medical attributes.


## Testing

After training and testing the K-Nearest Neighbors (KNN) model, I evaluated its performance on the test dataset. The model achieved an accuracy of **93%**, demonstrating strong predictive capabilities in classifying whether a sample had diabetes. 

To further validate the model's performance, I used a **confusion matrix** to analyze the number of correct and incorrect predictions:

```sh

[[ 4  7]
 [ 0 89]]

True Positives (TP): 89
False Positives (FP): 7
False Negatives (FN): 0
True Negatives (TN): 4

```

To better visaulize the confusion matrix, I used seaborn to represent the confusion matrix as a heatmap

![confusion matrix](https://github.com/user-attachments/assets/2b01315e-94cd-4cbc-a16f-c512ff9c1cf9)









