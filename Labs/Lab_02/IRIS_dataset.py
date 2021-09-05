# -*- coding: utf-8 -*-

# Spyder Editor

import numpy as np
import matplotlib.pyplot as plt


def load(filename):
    list_of_columns = []
    list_of_labels = []
    labels_mapping = {"Iris-setosa": 0,
                      "Iris-versicolor": 1, "Iris-virginica": 2}
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if len(data) == 5:
                # This check is necessary to avoid the last line where there is only a \n
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                    # Convert values to float
                # Delete \n at the end of the line
                data[4] = data[4].rstrip('\n')
                # Now create a 1-dim array and reshape it as a column vector,
                # then append it to the appropriate list
                list_of_columns.append(np.array(data[0:4]).reshape((4, 1)))
                # Append the value of the class to the appropriate list
                list_of_labels.append(labels_mapping[data[4]])
    # We have column vectors, we need to create a 4x150 matrix, so we have to
    # stack horizontally all the column vectors
    dataset_matrix = np.hstack(list_of_columns[:])
    # Create a 1-dim array with class values
    class_label_array = np.array(list_of_labels)
    return dataset_matrix, class_label_array


def custom_hist(attr_index, xlabel, D, L):
    # Function used to plot histograms. It receives the index of the attribute to plot,
    # the label for the x axis, the dataset matrix D and the array L with the values
    # for the classes
    plt.hist(D[attr_index, L == 0], color="#1e90ff",
             ec="#0000ff", density=True, alpha=0.6)
    plt.hist(D[attr_index, L == 1], color="#ff8c00",
             ec="#d2691e", density=True, alpha=0.6)
    plt.hist(D[attr_index, L == 2], color="#90ee90",
             ec="#006400", density=True, alpha=0.6)
    plt.legend(["Setosa", "Versicolor", "Virginica"])
    plt.xlabel(xlabel)
    plt.show()
    return


def custom_scatter(i, j, xlabel, ylabel, D, L):
    # Function used for scatter plots. It receives the indexes i, j of the attributes
    # to plot, the labels for x, y axes, the dataset matrix D and the array L with the
    # values for the classes
    plt.scatter(D[i, L == 0], D[j, L == 0], color="#1e90ff")
    plt.scatter(D[i, L == 1], D[j, L == 1], color="#ff8c00")
    plt.scatter(D[i, L == 2], D[j, L == 2], color="#90ee90")
    plt.legend(["Setosa", "Versicolor", "Virginica"])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    return


def mcol(vector, shape0):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    # The text suggested to write it because we will use it a lot.
    return vector.reshape(shape0, 1)


def mrow(vector, shape1):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    # The text suggested to write it because we will use it a lot.
    return vector.reshape(1, shape1)


if __name__ == '__main__':
    D, L = load("iris.csv")
    list_of_plot_labels = ["Sepal length",
                           "Sepal width", "Petal length", "Petal width"]
    # #Plot sepal length:
    # custom_hist(0, list_of_plot_labels[0], D, L)
    # #Plot sepal width:
    # custom_hist(1, list_of_plot_labels[1], D, L)
    # #Plot petal length:
    # custom_hist(2, list_of_plot_labels[2], D, L)
    # #Plot petal width:
    # custom_hist(3, list_of_plot_labels[3], D, L)
    # These plots are commented because they are included in the following for loops
    # #Visualize pairs of values in scatter plots:
    for i in range(4):
        for j in range(4):
            if (i == j):
                # Then plot histogram
                custom_hist(i, list_of_plot_labels[i], D, L)
            else:
                # Else use scatter plot
                custom_scatter(
                    i, j, list_of_plot_labels[i], list_of_plot_labels[j], D, L)
    # Compute mean over columns of the dataset matrix (mean over columns means that
    # for the first row we get a value, for the second row we get a value, ecc.)
    mu = D.mean(1)
    # Attention! mu is a 1-D array!
    # We want to subtract mean to all elements of the dataset (with broadcasting)
    # We need to reshape the 1-D array mu to a column vector 4x1 (where 4 is the first shape of D)
    mu = mcol(mu, D.shape[0])
    # Now we can subtract (with broadcasting). C stands for centered
    DC = D - mu
    # We can plot again the center data:
    for i in range(4):
        for j in range(4):
            if (i == j):
                custom_hist(i, list_of_plot_labels[i], DC, L)
            else:
                custom_scatter(
                    i, j, list_of_plot_labels[i], list_of_plot_labels[j], DC, L)

    # Conclusions: the centered data is really easy to visualize.
    # In fact, in the centered histograms we can see that values around 0 are near
    # the mean value, values that are distant from 0 are distant from the mean.
    # In scatter plots the situation is even more interesting, we have two
    # centered attributes on the axes so if a value is near 0,0 then it is near
    # the mean value for both attributes, if a value is near 0,3 (or 3,0) then it
    # is near the mean for an attribute and distant from the mean for the other one,
    # if a value is in a corner (for example 3,3) then it is distant from the mean
    # for both attributes.
