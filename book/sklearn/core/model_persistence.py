from sklearn import svm
from sklearn import datasets
import pickle

clf = svm.SVC()
X, y = datasets.load_iris(return_X_y=True)
clf.fit(X, y)
print("predict:", clf.predict(X[0:1])[0], " target:", y[0])

# dump to file
path_name = "/tmp/test.model"
f = open(path_name, "wb")
pickle.dump(clf, f)
f.close()

f = open(path_name, "rb")
clf2 = pickle.load(f)
f.close()
clf2.predict(X[0:1])
print("predict:", clf2.predict(X[0:1])[0], " target:", y[0])
