import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(1,"..\\")
from phoenixlib.adaline_sgd import AdalineSGD
from testutil import plot_decision_regions

if __name__ == "__main__":
	print("Testing AdalineSGD")
	print("-------------------------------------------------------------------------")
	# Read the Iris dataset. The Iris dataset is supposed to contain 150 entries. The first 50 are Iris-setosa,
	# the next 50 are Iris-versicolor, and the last 50 are Iris-virginica.
	#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, encoding="utf-8")
	df = pd.read_csv("iris.data", header=None, encoding="utf-8", names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species Type"])
	print(df.tail())
	print("-------------------------------------------------------------------------")

	# Select the first 100 Iris species (setosa and versicolor) in the Iris dataset.
	Y = df.iloc[0:100, 4].values
	# Prepare the labels; Iris-setosa is 0 and Iris-versicolor is 1.
	Y = np.where(Y == "Iris-setosa", 0, 1)
	# Prepare the corresponding feature-vectors for the features sepal-length and petal-length
	X = df.iloc[0:100, [0,2]].values

	# Standardize the data. Standardizing a data like an array involves adjusting the data such 
	# that the array's mean is 0 and standard deviation is 1. To standardize the elements of an
	# array X, replace every element X_i with (X_i - mean(X))/std-dev(X).
	X_std = np.copy(X)
	X_std[:,0] = (X_std[:,0] - X_std[:,0].mean())/X_std[:,0].std()
	X_std[:,1] = (X_std[:,1] - X_std[:,1].mean())/X_std[:,1].std()

	# Use Adaline with Stochastic Gradient Descent on the standardized data.
	adln_sgd = AdalineSGD(learning_rate=0.01, num_iterations=15, random_seed=1)
	adln_sgd.fit(X_std, Y)
    # Plot the decision boundary
	plot_decision_regions(X_std, Y, classifier=adln_sgd)
	plt.title("Adaline - with Stochastic Gradient Descent")
	plt.xlabel("Sepal length [standardized]")
	plt.ylabel("Petal length [standardized]")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.show()
    # Plot the errors vs epochs.
	plt.plot(range(1, len(adln_sgd.epoch_errors) + 1), adln_sgd.epoch_errors, marker="o")
	plt.xlabel("Epochs")
	plt.ylabel("Average loss")
	plt.tight_layout()
	plt.show()
