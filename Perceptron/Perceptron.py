import numpy as np

class perceptron(object):
	def __init__(self, x, y, alpha = 0.01):
		self.x = x
		self.y = y
		self.alpha = alpha
		self.w = np.ones((x.shape[1], 1))
		self.b = np.zeros((1, y.shape[1]))
		
	def judge(self, j):
		return (j >= 0) + 0
		
	def normalization(self):
		for i in range(1, self.x.shape[1]):
			sigma = np.std(self.x[:, i])
			mean = np.mean(self.x[:, i])
			self.x[:, i] = (self.x[:, i] - mean) / sigma
		
	def train(self, times = 5):
		self.normalization()
		for time in range(times):
			for i in range(self.x.shape[0]):
				dvalue = self.y[i] - self.judge(self.x[i].dot(self.w) + self.b[0])
				self.w += self.alpha * self.x[i].reshape(-1, 1).dot(dvalue).reshape(-1, 1)
				self.b += self.alpha * dvalue[0]
	
	def test(self, t_x, t_y):
		right = 0
		for i in range(t_x.shape[0]):
			h = self.judge(t_x[i].dot(self.w) + self.b[0])
			if h == t_y[i]:
				right += 1
		self.accuracy = right / t_x.shape[0]