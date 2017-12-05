from Bayes import naivebayes

nb = naivebayes()
nb.read_train_file("train.txt")
nb.bernoulli()
nb.predict("test.txt")

