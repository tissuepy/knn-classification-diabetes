from driver import Diabetes
import statistics
import csv
import matplotlib.pyplot as plt
import numpy as np
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
import seaborn as sns
from plotting import plot_error_vs_k  # Importing the functions from plotting.py




data_file = "diabetes_prediction_dataset.csv"
diabetes_file = []

def filereader(filename):
    '''
    :purpose: loop through the file and convert each line of the file to a list
    :param filename: a file that contains data about various restaurants and details about them
    :return: a 2D list
    '''
    with open(data_file, 'r') as file:
        csvReader = csv.reader(file)
        next(csvReader)
        for line in csvReader:
            diabetes_file.append(line)
        return diabetes_file

def object_conversion(diabetes_file):

    diabetes_objects = []
    for row in diabetes_file:
        gender = row[0]
        age = row[1]
        hypertension = row[2]
        heart_disease = row[3]
        smoking_history = row[4]
        bmi = row[5]
        hba1c_level = row[6]
        blood_glucose_level = row[7]
        diabetes = row[8]

        row_object = Diabetes(gender, age, hypertension, heart_disease, smoking_history, bmi, hba1c_level, blood_glucose_level, diabetes)
        diabetes_objects.append(row_object)

    return diabetes_objects

def remove_duplicates(diabetes_objects):
    '''
    :param restaurant_objects: a list of restaurants that has been converted to objects
    :return: a list of restaurants that is filtered for duplicates
    '''
    seen_data = []
    unique_data = []

    for obj in diabetes_objects:
        if obj not in seen_data:
            unique_data.append(obj)
            seen_data.append(obj)
    return unique_data

def min_max_normalization(diabetes_objects):

    ages = [obj.age for obj in diabetes_objects]
    bmis = [obj.bmi for obj in diabetes_objects]
    glucose_levels = [obj.blood_glucose_level for obj in diabetes_objects]
    hba1c_levels = [obj.hba1c_level for obj in diabetes_objects]

    min_ages = min(ages)
    max_ages = max(ages)

    min_bmis = min(bmis)
    max_bmis = max(bmis)

    min_glucose = min(glucose_levels)
    max_glucose = max(glucose_levels)

    min_hba1c_level = min(hba1c_levels)
    max_hba1c_level = max(hba1c_levels)


    for obj in diabetes_objects:
        obj.age = (obj.age - min_ages) / (max_ages - min_ages)
        obj.bmi = (obj.bmi - min_bmis) / (max_bmis - min_bmis)
        obj.age = (obj.blood_glucose_level - min_glucose) / (max_glucose - min_glucose)
        obj.hba1c_level = (obj.hba1c_level - min_hba1c_level) / (max_hba1c_level - min_hba1c_level)
    return diabetes_objects

def euc_distance_func(test_i, normal_diabetes_data):

    distances = []

    for testing_data in normal_diabetes_data:
        euclidean_distance = test_i.euclidean_distance(testing_data)
        label = testing_data.diabetes
        distances.append((euclidean_distance, label))
    return distances

def optimal_k(distances, k):
    for i in range(len(distances)):
        for j in range(i + 1, len(distances)):
            if distances[i] > distances[j]:
                distances[i], distances[j] = distances[j], distances[i]
    return distances[:k]

def classification(optimal_k_output):

    labels = []
    for i in optimal_k_output:
        labels.append(i[1])

    label_counter = Counter(labels)
    most_frequent_label = label_counter.most_common(1)[0][0]

    if most_frequent_label == 1:
        return("Diabetes")
    else:
        return("Not Diabetes")

def cf_matrix(y_pred, y_true):

    cm = confusion_matrix(y_pred, y_true)
    print("Confusion Matrix:")
    print(cm)

    print(f"The amount of True Positives is : {cm[0][0]}")
    print(f"The amount of True Negatives is : {cm[1][1]}")
    print(f"The amount of False Positives is : {cm[0][1]}")
    print(f"The amount of False Negatives is : {cm[1][0]}")

    accuracy_score = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[1][1] + cm[0][1] + cm[1][0])
    print(f"Accuracy Score: {accuracy_score}")

    # heatmap for the confusion matrix representation

    plt.figure()
    sns.heatmap(cm, annot=True, cmap="Reds",xticklabels=["Not Diabetes","Diabetes"],yticklabels=["Not Diabetes","Diabetes"])
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Confusion Matrix Heatmap for Diabetes Data")
    plt.show()

def main():

    diabetes_output = filereader(data_file)

    diabetes_objects_output = object_conversion(diabetes_output[:1000])
    #print(diabetes_objects_output)

    normalized_diabetes_objects = min_max_normalization(diabetes_objects_output)
    #print(normalized_diabetes_objects)

    ###### TEST SAMPLE ########

    test_sample = Diabetes(gender="Male", age=50, hypertension=1, heart_disease=1, smoking_history="Yes", bmi=700,
                           hba1c_level=100, blood_glucose_level=10, diabetes=None)

    euclidean_distances = euc_distance_func(test_sample, normalized_diabetes_objects[1:])
    #print(euclidean_distances)

    k = 10
    optimal_k_output = optimal_k(euclidean_distances, k)
    #print(optimal_k_output)


    ###### CLASSIFICATION OF THE TEST SAMPLE ########

    knn_output = classification(optimal_k_output)
    print(f"The test sample is classified as {knn_output}")


    ###### CONFUSION MATRIX LOGIC ########

    y_true = []
    y_pred = []

    test_samples = normalized_diabetes_objects[:100]
    training_samples = normalized_diabetes_objects[100:]

    for test_sample in test_samples:
        true_label = "Diabetes" if test_sample.diabetes == 1 else "Not Diabetes"
        y_true.append(true_label)  # Store actual label as a string

        euclidean_distances = euc_distance_func(test_sample, training_samples)
        optimal_k_output = optimal_k(euclidean_distances, k)
        predicted_label = classification(optimal_k_output)

        y_pred.append(predicted_label)

    cf_matrix(y_true, y_pred)


    ###### PLOTTING FOR ERRORS AND ACCURACIES ########

    errors = [10, 15, 20]
    k = range(1, len(errors) + 1)

    plot_error_vs_k(errors, k)

if __name__ == "__main__":
    main()
