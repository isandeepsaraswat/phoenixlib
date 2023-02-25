import numpy as np

class Perceptron:
	def __init__(self, learning_rate=0.01, num_iters=50, random_seed=1):
		self.__eta = learning_rate
		self.__n_iter = num_iters
		self.__radom_seed = random_seed
		self.__w = None
		self.__b = 0.0
		self.epoch_errors = []

	def fit(self, X, Y):
		prng = np.random.RandomState(self.__radom_seed)
		self.__w = prng.normal(0.0, 0.01, X.shape[1])
		self.__b = 0.0
		self.epoch_errors = []
		for _ in range(self.__n_iter):
			errors = 0
			for x,y in zip(X,Y):
				delta = self.__eta * (y - self.predict(x))
				self.__w += delta * x
				self.__b += delta
				errors += int(delta != 0.0)
			self.epoch_errors.append(errors)
		return self

	def predict(self, X):
		return np.where(np.dot(X, self.__w) + self.__b >= 0.0, 1, 0)
