import pandas as pd
from pandas.core.series import Series#for type hint that helps intellisense
import autosklearn.regression
import autosklearn.classification
import sklearn.model_selection

SECONDS_PER_DATASET = 60
SECONDS_PER_MODEL = 30

def run_regression_ensemble(X: Series, y: Series) -> autosklearn.regression.AutoSklearnRegressor:
    X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=SECONDS_PER_DATASET,
                    per_run_time_limit=SECONDS_PER_MODEL,
            )
    automl.fit(X_train, y_train)
    return automl

def run_classification_ensemble(X: Series, y: Series) -> autosklearn.classification.AutoSklearnClassifier:
    X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=SECONDS_PER_DATASET,
                    per_run_time_limit=SECONDS_PER_MODEL,
            )
    automl.fit(X_train, y_train)
    return automl