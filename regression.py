import pandas as pd
from pandas.core.series import Series#for type hint that helps intellisense
import autosklearn.regression
import sklearn.model_selection

def run_regression_ensemble(X: Series, y: Series) -> autosklearn.regression.AutoSklearnRegressor:
    X_train, X_test, y_train, y_test = \
                sklearn.model_selection.train_test_split(X, y, random_state=1)

    automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=120,
                    per_run_time_limit=30,
            )
    automl.fit(X_train, y_train)
    return automl