import numpy as np

class softmax(object):
	def __init__(self, x, y, alpha = 0.01):
		self.x = x
		self.y = y
		self.alpha = alpha
		self.classes = np.unique(y)
		self.classes_num = self.classes.shape[0]
		self.theta = np.ones((x.shape[1], self.classes_num))
		self.x_t = x.T
		self.y_a = np.zeros((self.y.shape[0], self.classes_num))
		for i in range(self.y.shape[0]):
			self.y_a[i][list(self.classes).index(self.y[i][0])] = 1
		
	def normalization(self):
		for i in range(1, self.x.shape[1]):
			sigma = np.std(self.x[:, i])
			mean = np.mean(self.x[:, i])
			self.x[:, i] = (self.x[:, i] - mean) / sigma	
	
	def gd(self, times = 500):
		self.normalization()
		for i in range(times):
			p = np.exp(self.x.dot(self.theta))
			for i in range(p.shape[0]):
				p[i, :] = p[i, :] / np.sum(p[i, :])	
			self.theta = self.theta + self.alpha * self.x_t.dot(self.y_a - p)
			
	def sgd(self, times = 2000):
		self.normalization()
		for i in range(1):
			for j in range(self.x.shape[0]):
				loc = np.random.randint(0, self.x.shape[0] - 1)
				p = np.exp(self.x[loc].dot(self.theta))
				p = p / np.sum(p)
				self.theta = self.theta + self.alpha * self.x[loc].reshape(-1, 1).dot((self.y_a[loc] - p).reshape(1, -1))
	
	def newton(self, times = 5):
		M = self.x.shape[0]
		for i in range(times):
			p = np.exp(self.x.dot(self.theta))
			for i in range(p.shape[0]):
				p[i, :] = p[i, :] / np.sum(p[i, :])
			for loc in range(self.classes_num):		
				J = (1 / M) * self.x_t.dot((p[:, loc] - self.y_a[:, loc]).reshape(-1, 1))
				H = (1 / M) * self.x_t.dot(np.diag(p[:, loc].flatten()).dot(np.diag(1 - p[:, loc].flatten()).dot(self.x)))
				self.theta[:, loc] = self.theta[:, loc] - (np.linalg.inv(H).dot(J)).flatten()
			
	def test(self, t_x, t_y):
		right = 0
		p = np.exp(t_x.dot(self.theta))
		for i in range(p.shape[0]):
			p[i, :] = p[i, :] / np.sum(p[i, :])
			max_one = np.max(p[i, :])
			hyp = self.classes[list(p[i]).index(max_one)]
			if hyp == t_y[i][0]:
				right += 1
		self.accuracy = right / t_x.shape[0]