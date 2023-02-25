import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(1,"..\\")
from phoenixlib.adaline import Adaline
from testutil import plot_decision_regions

if __name__ == "__main__":
	print("Testing Adaline")
	# Read the Iris dataset. The Iris dataset is supposed to contain 150 entries. The first 50 are Iris-setosa,
	# the next 50 are Iris-versicolor, and the last 50 are Iris-virginica.
	#df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", header=None, encoding="utf-8")
	df = pd.read_csv("iris.data", header=None, encoding="utf-8", names=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "Species Type"])
	print(df.tail())

	# Select the first 100 Iris species (setosa and versicolor) in the Iris dataset.
	Y = df.iloc[0:100, 4].values
	# Prepare the labels; Iris-setosa is 0 and Iris-versicolor is 1.
	Y = np.where(Y == "Iris-setosa", 0, 1)
	# Prepare the corresponding feature-vectors for the features sepal-length and petal-length
	X = df.iloc[0:100, [0,2]].values

	
	# We are going to train an Adaline model with two learning rates --- one too high and the other
	# too low --- and then plot the errors to understand the impact of learning rate on errors.
	
	# Create a figure with two sub-plots, one next to the other.
	fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
	# Train an Adaline model with a high learning rate.
	adln = Adaline(learning_rate=0.1, num_iterations=15).fit(X,Y)
	# The mean-squared-error actually increases with every epoch, though gradually. To visualize it
	# better, we plot the errors on the logarithmic scale, otherwise the increase is not very 
	# perceptible.
	ax[0].plot(range(1, len(adln.epoch_errors)+1), np.log10(adln.epoch_errors), marker='o')
	ax[0].set_xlabel('Epoch')
	ax[0].set_ylabel("log(Mean squared error)")
	ax[0].set_title("Adaline - high learning rate (0.1)")
	# Train an Adaline model with a low learning rate.
	adln = Adaline(learning_rate=0.0001, num_iterations=15).fit(X,Y)
	# The mean-squared-error decreases very slowly. Note that this time we plot the mean-squared-error
	# instead of the log of mean-squared-error.
	ax[1].plot(range(1, len(adln.epoch_errors)+1), adln.epoch_errors, marker='o')
	ax[1].set_xlabel('Epoch')
	ax[1].set_ylabel("Mean squared error")
	ax[1].set_title("Adaline - low learning rate (0.0001)")
	plt.show()

	# We shall now demonstrate the effect of standardization on the learning rate.
	# Standardization makes gradient-descent work faster because we can use a higher 
	# learning rate on data that is standardized when compared to data that is not 
	# standardized. Standardized data has more tolerance for a higher learning rate
	# without running an equivalent risk of the error increasing. Standardization involves 
	# adjusting the values of the feature vectors such that they all have a mean of 0 
	# and a std. deviation of 1. To standardize a feature vector x(j), replace it with 
	# {x(j)-mean(x(j))}/std-dev(x(j))
	X_std = np.copy(X)
	X_std[:,0] = (X_std[:,0] - X_std[:,0].mean())/X_std[:,0].std()
	X_std[:,1] = (X_std[:,1] - X_std[:,1].mean())/X_std[:,1].std()
	# Use Adaline on the standardized data, and now a higher learning rate is Okay.
	adln = Adaline(learning_rate=0.5, num_iterations=15).fit(X_std,Y)
	plot_decision_regions(X_std, Y, classifier=adln)
	plt.title("Adaline (on standardized data) - Gradient Descent")
	plt.xlabel("Sepal length [standardized]")
	plt.ylabel("Petal length [standardized]")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.show()
	# Now plot the mean-squared-errors against the epochs. You will see the errors go down
	# significantly for the same number of epochs as earlier --- thanks to standardization.
	plt.plot(range(1, len(adln.epoch_errors)+1), adln.epoch_errors, marker='o')
	plt.xlabel("Epochs")
	plt.ylabel("Mean squared error")
	plt.title("Adaline (on standardized data) - high learning rate (0.5)")
	plt.tight_layout()
	plt.show()
