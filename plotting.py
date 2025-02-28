import matplotlib.pyplot as plt

def plot_error_vs_k(error_values, k):

    plt.figure()
    plt.plot(k, error_values, marker='o', color='b', linestyle='-', markersize = 10)
    plt.title("Error vs K-Values")
    plt.xlabel("K (# of Neighbors)")
    plt.ylabel("Error Rate")
    plt.grid(True)
    plt.show()

def error_calc(true_labels, predicted_labels):

    true_labels = []
    predicted_labels = []