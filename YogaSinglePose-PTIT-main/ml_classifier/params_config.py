# params_config.py - Hyperparameter spaces for ML models
from skopt.space import Integer, Real, Categorical
from skopt.space import Categorical


# Dùng Integer cho các giá trị nguyên (n_neighbors, max_depth, n_estimators…)

# Dùng Real cho các giá trị số thực (learning_rate, alpha, var_smoothing…)

# Dùng Categorical cho danh sách rời rạc (kernel, activation, solver…)


knn_params = {
    'n_neighbors': Integer(3, 19),
    'weights': Categorical(['uniform', 'distance']),
    'metric': Categorical(['euclidean', 'manhattan'])
}

svm_params = {
    'C': Real(1e-4, 1e3, prior='log-uniform'),  # mở rộng
    'kernel': Categorical(['linear', 'rbf', 'poly']),  # thêm polynomial
    'gamma': Real(1e-5, 10, prior='log-uniform')
}


dt_params = {
    'max_depth': Integer(2, 50),  # mở rộng
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),  # thêm
    'criterion': Categorical(['gini', 'entropy'])
}


rf_params = {
    'n_estimators': Integer(100, 500),  # tăng số cây
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'max_features': Categorical(['sqrt', 'log2', None]),  # thêm
    'criterion': Categorical(['gini', 'entropy'])
}


gbt_params = {
    'n_estimators': Integer(100, 500),
    'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 15),
    'subsample': Real(0.5, 1.0),
    'min_samples_leaf': Integer(1, 10)  # thêm
}

xgb_params = {
    'n_estimators': Integer(100, 1000),
    'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 15),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'min_child_weight': Integer(1, 10)  # thêm
}


lgbm_params = {
    'n_estimators': Integer(100, 1000),
    'learning_rate': Real(0.005, 0.3, prior='log-uniform'),
    'max_depth': Integer(-1, 15),  # -1 = không giới hạn
    'num_leaves': Integer(15, 255),
    'subsample': Real(0.5, 1.0),
    'feature_fraction': Real(0.5, 1.0)  # thêm
}


catboost_params = {
    'iterations': Integer(50, 200),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'depth': Integer(3, 9),
    'l2_leaf_reg': Integer(3, 10),
    'bagging_temperature': Real(0.0, 1.0)  # thêm
}

gnb_params = {
    'var_smoothing': Real(1e-10, 1e-6, prior='log-uniform')
}

mnb_params = {
    'alpha': Real(0.001, 10, prior='log-uniform'),
    'fit_prior': Categorical([True, False])
}

mlp_params = {
    'hidden_layer_sizes': Categorical([50, 100, 150, 200, 300]),
    'activation': Categorical(['relu', 'tanh', 'logistic']),
    'solver': Categorical(['adam', 'sgd']),
    'alpha': Real(1e-5, 0.1, prior='log-uniform'),
    'learning_rate_init': Real(1e-4, 0.1, prior='log-uniform'),
    'max_iter': Integer(200, 500)
}