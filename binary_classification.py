import pandas, sklearn.linear_model

def logistic_regression(df: pandas.DataFrame, target:str):
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy().ravel()
    print(X.shape,y.shape)
    return sklearn.linear_model.LogisticRegression(max_iter=4000).fit(X, y)

def test_regression(df,scorer,target):
    X = df.drop(columns=[target]).to_numpy()
    y = df[target].to_numpy().ravel()
    return scorer(X,y)
