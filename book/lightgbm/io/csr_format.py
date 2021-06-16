import scipy
from sklearn import datasets
from sklearn.model_selection import train_test_split
import lightgbm as lgb

cancer = datasets.load_breast_cancer()
X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)

row = []
col = []
dat = []
i = 0
for arr in X_train:
    for k, v in enumerate(arr):
        row.append(i)
        col.append(int(k))
        dat.append(float(v))
    i += 1

csr = scipy.sparse.csr_matrix((dat, (row, col)))
train_data = lgb.Dataset(csr)
