from __future__ import annotations
import numpy as np

class AdalineSGD:
	def __init__(self, learning_rate=0.01, num_iterations=50, random_seed=1):
		self.__eta = learning_rate
		self.__num_epochs = num_iterations
		self.__random_seed = random_seed
		self.__prng = np.random.RandomState(self.__random_seed)
		self.__w = None
		self.__b = 0.0
		self.epoch_errors = []

	def fit(self, X : np.ndarray, Y : np.ndarray) -> AdalineSGD:
		# Initialize the weights from a normal distribution with a mean of 0 
		# and a standard devitation of 0.01.
		self.__w = self.__prng.normal(0.0, 0.01, X.shape[1])
		self.__b = 0.0
		self.epoch_errors = []
		for _ in range(self.__num_epochs):
			# We must shuffle the input in every epoch to prevent cycles
			X, Y = self.__shuffle(X, Y)
			losses = []
			# Update the weights and bias using the Stochastic Gradient Descent 
			# algorithm, also called iterative or online gradient descent.
			for x, y in zip(X, Y):
				# For each x, the net input is wx + b
				net_input = np.dot(x, self.__w) + self.__b
				# Compute the output for each x
				net_output = self.activation(net_input)
				# Compute the error
				error = y - net_output
				# Based on the error, update the weights and bias.
				self.__w += 2.0 * self.__eta * error * x
				self.__b += 2.0 * self.__eta * error
				# Save the squared-error for each input
				losses.append(error**2)
			# Save the mean-sqaured-error for this epoch epoch
			self.epoch_errors.append(np.mean(losses))
		return self

	def __shuffle(self, X : np.ndarray, Y : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		permuted_idxs = self.__prng.rgen.permutation(len(Y))
		return X[permuted_idxs], Y[permuted_idxs]

	def activation(self, X : np.ndarray) -> np.ndarray:
		# For Adaline, the activation function sigma is identity
		#     sigma(z) = z
		return X

	def predict(self, X):
		# The model has been trained.
		assert self.__w is not None
		# Use the model's weights and bias to compute the output for each x in X
		net_output = X.dot(self.__w) + self.__b
		# Since this is a binary classifier with classes 0 and 1, we treat any
		# output >= 0.5 to be 1 and any output < 0.5 to be 0.
		return np.where(net_output >= 0.5, 1, 0)
