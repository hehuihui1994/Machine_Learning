import math

class naivebayes(object):
	def __init__(self, alpha = 1):
		self.classes = {}
		self.documents = []
		self.words = []
		self.P = {}
		
	def read_train_file(self, fname):
		fobj = open(fname, 'r')
		for eachline in fobj:
			eachline = eachline.split('\n')[0]
			oneclass = eachline.split(' ', 1)[0]
			content = eachline.split(' ', 1)[1].split()
			self.documents.append(content)
			self.words.extend(content)
			if oneclass not in self.classes.keys():
				self.classes[oneclass] = {"documents": [], "words": {}}
			self.classes[oneclass]["documents"].append(len(self.documents) - 1)
			for word in content:
				if word not in self.classes[oneclass]["words"].keys():
					self.classes[oneclass]["words"][word] = 0
				self.classes[oneclass]["words"][word] += 1
		self.N = len(self.words)
		self.words = list(set(self.words))
		self.V = len(self.words)
					
	def multi_nomial(self):
		for oneclass in self.classes.keys():
			class_words_num = sum(self.classes[oneclass]["words"].values())
			self.P[oneclass] = class_words_num / self.N
			for word in self.words:
				if word in self.classes[oneclass]["words"].keys():
					self.P[(word, oneclass)] = (self.classes[oneclass]["words"][word] + 1) / (class_words_num + self.V)
				else:
					self.P[(word, oneclass)] = 1 / (class_words_num + self.V)
					
	def bernoulli(self):
		for oneclass in self.classes.keys():
			self.P[oneclass] = len(self.classes[oneclass]["documents"]) / len(self.documents)
			for word in self.words:
				include_num = 0
				for doc_num in self.classes[oneclass]["documents"]:
					if word in self.documents[doc_num]:
						include_num += 1
				self.P[(word, oneclass)] = (include_num + 1) / (sum(self.classes[oneclass]["words"].values()) + 2)
	
	def predict(self, fname):
		fobj = open(fname, 'r')
		for eachline in fobj:
			eachline = eachline.split('\n')[0]
			content = eachline.split()
			pre = {}
			for oneclass in self.classes.keys():
				pre[oneclass] = math.log(self.P[oneclass])
				for word in content:
					if word in self.words:
						pre[oneclass] += math.log(self.P[(word, oneclass)])
			result = list(pre.keys())[list(pre.values()).index(max(pre.values()))]
			print(eachline, '->', result)
	