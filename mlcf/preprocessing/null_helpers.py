import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

from mlcf.configs.constants import NULL_PREDICTOR_SEED


def find_column_mode(data):
    result = stats.mode(data)
    return result.mode[0]


def find_column_mean(data):
    return np.mean(data).round()


def find_column_median(data):
    return np.median(data).round()


def base_regressor(column_type):
    if column_type == 'numerical':
        model = LinearRegression()
    elif column_type == 'categorical':
        model = LogisticRegression()
    else:
        raise ValueError(
            "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model


def base_knn(column_type, n_neighbors):
    if column_type == 'numerical':
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
    elif column_type == 'categorical':
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
    else:
        raise ValueError(
            "Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model


def base_random_forest(column_type, n_estimators, max_depth, class_weight=None, min_samples_leaf=None, oob_score=None):
    if column_type == 'numerical':
        model = RandomForestRegressor(n_estimators = n_estimators,
                                      max_depth = max_depth,
                                      random_state = NULL_PREDICTOR_SEED)
    elif column_type == 'categorical':
        model = RandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       random_state = NULL_PREDICTOR_SEED,
                                       class_weight=class_weight,
                                       min_samples_leaf=min_samples_leaf,
                                       oob_score=oob_score)
    else:
        raise ValueError("Can only support numerical or categorical columns, got column_type={0}".format(column_type))
    return model
