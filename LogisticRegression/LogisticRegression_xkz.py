import numpy as np

class logistic(object):
	def __init__(self, x, y, alpha = 0.1):
		self.x = x
		self.y = y
		self.alpha = alpha
		self.theta = np.zeros((x.shape[1], 1))
		self.x_t = x.T
		
	def sigmod(self, s):
		return 1 / (1 + np.exp(-s))
		
	def normalization(self, mult):
		for i in range(1, self.x.shape[1]):
#			sigma = np.std(self.x[:, i])
#			mean = np.mean(self.x[:, i])
#			self.x[:, i] = (self.x[:, i] - mean) / sigma
#			
#			xmin = np.min(self.x[:, i])
#			xmax = np.max(self.x[:, i])
#			self.x[:, i] = (self.x[:, i] - xmin) / (xmax - xmin)
			self.x[:, i] = self.x[:, i] / mult
					
	def gd(self, times = 2000):
		self.normalization(100)
		for i in range(times):
			dvalue = self.sigmod(self.x.dot(self.theta)) - self.y
			self.theta = self.theta - self.alpha * self.x_t.dot(dvalue)
		
	def sgd(self, times = 5000):
		self.normalization(100)
		for i in range(times):
			for j in range(self.x.shape[0]):
				loc = np.random.randint(0, self.x.shape[0] - 1)
				self.dvalue = self.sigmod(self.x[loc].dot(self.theta)) - self.y[loc]
				self.theta = self.theta - self.alpha * self.x[loc].reshape(-1, 1) * self.dvalue
				
	def newton(self, times = 10):
		for i in range(times):
			N = self.x.shape[0]
			ht = self.sigmod(self.x.dot(self.theta))
			J = (1 / N) * self.x_t.dot(ht - self.y)
			H = (1 / N) * self.x_t.dot(np.diag(ht.flatten()).dot(np.diag(1 - ht.flatten()).dot(self.x)))
			self.theta = self.theta - np.linalg.inv(H).dot(J)
		
		