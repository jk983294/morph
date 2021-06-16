from book.sklearn.core.dataset import load_housing_data
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


def explore_data(housing):
    print(housing.columns)
    print(housing.head())
    # print(housing.info())
    # print(housing["ocean_proximity"].value_counts())
    # print(housing.describe())

    # housing.hist(bins=50, figsize=(20, 15))
    # housing["median_income"].hist()
    # housing["income_cat"].hist()
    # plt.show()


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)


def print_diff_per_split(housing, strat_test_set, test_set):
    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()
    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    print(compare_props)


def visualize_scatter(strat_train_set):
    housing = strat_train_set.copy()
    # housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="population", figsize=(10, 7),
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
                 sharex=False)
    plt.legend()
    plt.show()


def visualize_corr(strat_train_set):
    housing = strat_train_set.copy()
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
    # scatter_matrix(housing[attributes], figsize=(12, 8))

    # through matrix we find median_income most correlate with median_house_value
    housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
    plt.axis([0, 16, 0, 550000])
    plt.show()


def learning(strat_train_set):
    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
    sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
    print("incomplete rows count", len(sample_incomplete_rows))


if __name__ == '__main__':
    np.random.seed(42)
    housing = load_housing_data()
    housing["income_cat"] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1, 2, 3, 4, 5])
    explore_data(housing)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    print(len(train_set), "train +", len(test_set), "test")
    print(len(strat_train_set), "train +", len(strat_test_set), "test")

    print_diff_per_split(housing, strat_test_set, test_set)

    # visualize_scatter(strat_train_set)
    # visualize_corr(strat_train_set)

    # Prepare the data for Machine Learning algorithms
    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()
