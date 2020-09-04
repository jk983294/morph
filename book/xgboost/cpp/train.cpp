#include <xgboost/c_api.h>
#include <ztool.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace std;

const char* const csv_path = "factor.csv";

#define safe_xgboost(call)                                                                             \
    {                                                                                                  \
        int err = (call);                                                                              \
        if (err != 0) {                                                                                \
            fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
            exit(1);                                                                                   \
        }                                                                                              \
    }

bool read_file(const string& path, std::vector<string>& contents) {
    ifstream ifs(path, ifstream::in);

    if (ifs.is_open()) {
        string s;
        while (getline(ifs, s)) {
            if (!s.empty()) contents.push_back(s);
        }
        ifs.close();
        return true;
    } else {
        return false;
    }
}

void split_train_test_set(const std::vector<string>& contents, string& header, std::vector<string>& train,
                          std::vector<string>& test) {
    random_device rd;
    mt19937 generator(rd());
    std::uniform_real_distribution<double> urd1(0.0, 1.0);
    double test_percent = 0.2;
    header = contents.front();
    for (size_t i = 1; i < contents.size(); ++i) {
        if (urd1(generator) < test_percent) {
            test.push_back(contents[i]);
        } else {
            train.push_back(contents[i]);
        }
    }
}

std::tuple<vector<float>, vector<float>> handle_csv_contents(const std::vector<string>& contents, int column) {
    vector<float> features, labels;
    for (size_t i = 0; i < contents.size(); ++i) {
        vector<string> lets = ztool::split(contents[i], ',');
        for (int j = 0; j < column; ++j) {
            if (j == 0)
                labels.push_back(std::stof(lets[j]));
            else
                features.push_back(std::stof(lets[j]));
        }
    }
    return {features, labels};
}

int main(int argc, char** argv) {
    std::vector<string> contents, train_contents, test_contents;
    string header;
    read_file(csv_path, contents);
    split_train_test_set(contents, header, train_contents, test_contents);
    std::vector<string> columns = ztool::split(header, ',');
    vector<float> train_mat, train_labels, test_mat, test_labels;
    std::tie(train_mat, train_labels) = handle_csv_contents(train_contents, columns.size());
    std::tie(test_mat, test_labels) = handle_csv_contents(test_contents, columns.size());

    // load the data
    DMatrixHandle dtrain, dtest;
    safe_xgboost(XGDMatrixCreateFromMat(train_mat.data(), train_contents.size(), columns.size() - 1, NAN, &dtrain));
    safe_xgboost(XGDMatrixSetFloatInfo(dtrain, "label", train_labels.data(), train_labels.size()));
    safe_xgboost(XGDMatrixCreateFromMat(test_mat.data(), test_contents.size(), columns.size() - 1, NAN, &dtest));
    safe_xgboost(XGDMatrixSetFloatInfo(dtest, "label", test_labels.data(), test_labels.size()));

    // create the booster
    BoosterHandle booster;
    DMatrixHandle eval_dmats[2] = {dtrain, dtest};
    safe_xgboost(XGBoosterCreate(eval_dmats, 2, &booster));

    // configure the training
    // available parameters are described here:
    //   https://xgboost.readthedocs.io/en/latest/parameter.html
    safe_xgboost(XGBoosterSetParam(booster, "tree_method", "auto"));
    safe_xgboost(XGBoosterSetParam(booster, "booster", "gbtree"));
    safe_xgboost(XGBoosterSetParam(booster, "objective", "reg:squarederror"));
    safe_xgboost(XGBoosterSetParam(booster, "min_child_weight", "1"));
    safe_xgboost(XGBoosterSetParam(booster, "eta", "0.1"));
    safe_xgboost(XGBoosterSetParam(booster, "max_depth", "5"));
    safe_xgboost(XGBoosterSetParam(booster, "verbosity", "1"));

    // train and evaluate for n_trees iterations
    int n_trees = 50;
    const char* eval_names[2] = {"train", "test"};
    const char* eval_result = nullptr;
    for (int i = 0; i < n_trees; ++i) {
        safe_xgboost(XGBoosterUpdateOneIter(booster, i, dtrain));
        safe_xgboost(XGBoosterEvalOneIter(booster, i, eval_dmats, eval_names, 2, &eval_result));
        printf("%s\n", eval_result);
    }

    bst_ulong num_feature = 0;
    safe_xgboost(XGBoosterGetNumFeature(booster, &num_feature));
    printf("num_feature: %zu\n", num_feature);

    // predict
    bst_ulong out_len = 0;
    const float* out_result = nullptr;

    safe_xgboost(XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result));
    printf("y pred: ");
    for (int i = 0; i < 5; ++i) {
        printf("%1.4f ", out_result[i]);
    }
    printf("\n");

    // print true labels
    safe_xgboost(XGDMatrixGetFloatInfo(dtest, "label", &out_len, &out_result));
    printf("y actual: ");
    for (int i = 0; i < 5; ++i) {
        printf("%1.4f ", out_result[i]);
    }
    printf("\n");

    safe_xgboost(XGBoosterSaveModel(booster, "/tmp/test.xgb.model"));

    // free everything
    safe_xgboost(XGBoosterFree(booster));
    safe_xgboost(XGDMatrixFree(dtrain));
    safe_xgboost(XGDMatrixFree(dtest));
    return 0;
}
