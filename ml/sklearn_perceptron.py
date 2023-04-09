import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

def plot_decision_regions(X, Y, classifier, test_idx=None, resolution=0.02):
	# Setup marker-generator and colour-map
	markers = ('o', 's', '^', 'v', '<')
	colours = ("red", "blue", "lightgreen", "gray", "cyan")
	cmap = matplotlib.colors.ListedColormap(colours[:len(np.unique(Y))])
	# Get the sepal length range [min-1, max+1) for plotting
	sepal_min, sepal_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	# Get the petal length range [min-1, max+1) for plotting
	petal_min, petal_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	# Get meshgrids for these sepal and petal ranges
	sepal_range, petal_range = np.meshgrid(np.arange(sepal_min, sepal_max, resolution), np.arange(petal_min, petal_max, resolution))
	# Generate all combinations of the points in the sepal and petal ranges (the points are separated by resolution units)
	# and run the classifier on each such point. A decision boundary should emerge.
	labels = classifier.predict(np.array([sepal_range.ravel(), petal_range.ravel()]).T)
	labels = labels.reshape(sepal_range.shape)
	plt.contourf(sepal_range, petal_range, labels, alpha=0.3, cmap=cmap)
	plt.xlim(sepal_range.min(), sepal_range.max())
	plt.ylim(petal_range.min(), petal_range.max())
	# Plot the training dataset on the contour plot for better understanding of where the decision boundary lies.
	for idx,cl in enumerate(np.unique(Y)):
		plt.scatter(x=X[Y==cl, 0], y=X[Y==cl, 1], alpha=0.8, c=colours[idx], marker=markers[idx], label=f'Class {cl}', edgecolor="black")
	# Highlight test examples
	if test_idx:
		# plot all examples
		X_test, Y_test = X[test_idx, :], Y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], 
					c='none', edgecolor='black', alpha=1.0,
					linewidth=1, marker='o',
					s=100, label='Test set')


if __name__ == "__main__":
	print("Using Scikit-Learn's Perceptron Model")
	
	# Load the Iris dataset
	iris = sklearn.datasets.load_iris()
	X = iris.data[:, [2,3]]
	Y = iris.target
	print("Loaded the Iris dataset ({} data points)".format(len(X)))
	print("The class labels of the Iris dataset are {}".format(np.unique(Y)))

	# Generate the training and test sets.
	X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)
	print("Checking that the train_test_split( ) method has stratified the data correctly for test_size=0.3")
	print("\t Labels count in Y = {}".format(np.bincount(Y)))
	print("\t Labels count in Y_test = {}".format(np.bincount(Y_test)))
	print("\t Labels count in Y_train = {}".format(np.bincount(Y_train)))

	# Standardize the data
	std_scaler = sklearn.preprocessing.StandardScaler()
	std_scaler.fit(X_train)
	X_train_std = std_scaler.transform(X_train)
	X_test_std = std_scaler.transform(X_test)

	# Train a Perceptron model
	perceptron = sklearn.linear_model.Perceptron(eta0=0.1, random_state=1)
	perceptron.fit(X_train_std, Y_train)
	Y_pred = perceptron.predict(X_test_std)
	print(f"Perceptron accuracy = {(Y_test == Y_pred).sum() * 100 / len(Y_test):.2f}%")
	print(f"sklearn.metrics.accuracy_score(Y_test, Y_pred) = {sklearn.metrics.accuracy_score(Y_test, Y_pred):.4f}")
	print(f"perceptron.score(X_test_std, Y_test) = {perceptron.score(X_test_std, Y_test):.4f}")

	X_combined_std=np.vstack((X_train_std, X_test_std))
	Y_combined = np.hstack((Y_train, Y_test))
	plot_decision_regions(X=X_combined_std, Y=Y_combined, classifier=perceptron, test_idx=range(105, 150))
	plt.xlabel("Petal length [standardized]")
	plt.ylabel("Petal width [standardized]")
	plt.legend(loc='upper left')
	plt.tight_layout()
	plt.show()
