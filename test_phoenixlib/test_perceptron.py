import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(1,"..\\")
from phoenixlib.perceptron import Perceptron
from testutil import plot_decision_regions


if __name__ == "__main__":
	print("Testing Perceptron")
	# Read the Iris dataset. The Iris dataset is supposed to contain 150 entries. The first 50 are Iris-setosa,
	# the next 50 are Iris-versicolor, and the last 50 are Iris-virginica.
	#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, encoding="utf-8")
	df = pd.read_csv("iris.data", header=None, encoding="utf-8", names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species Type"])
	print(df.tail())

	# Select the first 100 Iris species (setosa and versicolor) in the Iris dataset.
	Y = df.iloc[0:100, 4].values
	# Prepare the labels; Iris-setosa is 0 and Iris-versicolor is 1.
	Y = np.where(Y == "Iris-setosa", 0, 1)
	# Prepare the corresponding feature-vectors for the features sepal length and petal length
	X = df.iloc[0:100, [0,2]].values
	# Visualize the two types of Iris flowers using a scatter plot.
	plt.scatter(X[:50, 0], X[:50, 1], color="red", marker="o", label="Iris-setosa")
	plt.scatter(X[50:, 0], X[50:, 1], color="blue", marker="s", label="Iris-versicolor")
	plt.xlabel("Sepal length [cm]")
	plt.ylabel("Petal length [cm]")
	plt.legend(loc="upper left")
	plt.show()
	# Train a perceptron to differentiate between the two. X (100x2) has the feature vectors and Y (100x1)  has the corresponding labels.
	pcptron = Perceptron(learning_rate=0.1, num_iters=10)
	pcptron.fit(X,Y)
	# Plot the misclassification error for each epoch.
	plt.plot(range(1, len(pcptron.epoch_errors)+1), pcptron.epoch_errors, marker="o")
	plt.xlabel("Epochs")
	plt.ylabel("Number of updates")
	plt.show()
	# Plot the decision regions and show the training points on it to better understand the decision boundary.
	plot_decision_regions(X, Y, classifier=pcptron)
	plt.xlabel("Sepal length [cm]")
	plt.ylabel("Petal length [cm]")
	plt.legend(loc="upper left")
	plt.show()
