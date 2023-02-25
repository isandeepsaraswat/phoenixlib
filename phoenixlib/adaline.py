import numpy as np

class Adaline:
	def __init__(self, learning_rate=0.01, num_iterations=50, random_seed=1):
		self.__eta = learning_rate
		self.__n_iter = num_iterations
		self.__random_seed = random_seed
		self.__w = None
		self.__b = 0.0
		self.epoch_errors = []

	def fit(self, X, Y):
		"""Uses full batch gradient descent."""
		prng = np.random.RandomState(self.__random_seed)
		self.__w = prng.normal(0.0, 0.01, X.shape[1])
		self.__b = 0.0
		self.epoch_errors = []
		for _ in range(self.__n_iter):
			# Compute the input for each x (= w.x + b)
			net_input = X.dot(self.__w) + self.__b
			# Compute the output for each x
			net_output = self.activation(net_input)
			# Compute the errors
			errors = Y - net_output
			# Based on the errors, update the weights and bias
			self.__w += 2 * self.__eta * X.T.dot(errors)/X.shape[0]
			self.__b += 2 * self.__eta * errors.mean()
			# Note the mean-squared error for this epoch
			self.epoch_errors.append((errors**2).mean())
		return self

	def activation(self, X):
		# The activation function sigma(z) for Adaline is identity:
		#    sigma(z) = z
		return X

	def predict(self, X):
		# The model has been trained (w and b have been learnt).
		# Use them to compute the model's output for each x
		net_output = X.dot(self.__w) + self.__b
		# Since this is a binary classifier with classes 0 and 1, 
		# we treat any output >= 0.5 to be 1 and any output < 0.5 
		# to be 0.
		return np.where(net_output >= 0.5, 1, 0)

		
