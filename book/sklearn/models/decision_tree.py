import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()  # use CART algorithm
clf.fit(X_train, y_train,)
clf.score(X_test, y_test)

output_file_path = '/tmp/my_tree.dot'
tree_pic = export_graphviz(clf, out_file=output_file_path)
print("run cmd:\ndot -Tpng /tmp/my_tree.dot -o /tmp/my_tree.png")
