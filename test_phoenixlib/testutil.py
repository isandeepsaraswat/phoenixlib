from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, Y, classifier, resolution=0.02):
	# Get the sepal length range [min-1, max+1) for plotting
	sepal_min, sepal_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	# Get the petal length range [min-1, max+1) for plotting
	petal_min, petal_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	# Get meshgrids for these sepal and petal ranges
	sepal_range, petal_range = np.meshgrid(np.arange(sepal_min, sepal_max, resolution), np.arange(petal_min, petal_max, resolution))
	# Generate all combinations of the points in the sepal and petal ranges (the points are separated by resolution units)
	# and run the classifier on each such point. A decision boundary should emerge.
	labels = classifier.predict(np.array([sepal_range.ravel(), petal_range.ravel()]).T)
	# Plot the labels on a contour plot to see the decision boundary.
	colors = ("red", "blue", "lightgreen", "gray", "cyan")
	cmap = ListedColormap(colors[:len(np.unique(Y))])
	labels = labels.reshape(sepal_range.shape)
	plt.contourf(sepal_range, petal_range, labels, alpha=0.1, cmap=cmap)
	plt.xlim(sepal_range.min(), sepal_range.max())
	plt.ylim(petal_range.min(), petal_range.max())
	markers = ("o", "s", "^", "v", "<")
	# Plot the training dataset on the contour plot for better understanding of where the decision boundary lies.
	for idx,cl in enumerate(np.unique(Y)):
		plt.scatter(x=X[Y==cl, 0], y=X[Y==cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=f'Class {cl}', edgecolor="black")
