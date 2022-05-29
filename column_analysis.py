import csv
import pandas
from enum import Enum, auto
from binary_classification import logistic_regression, test_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class ColumnType(Enum):
    ID = auto()
    CATEGORICAL = auto()
    NUMERIC = auto()

def clean(df: pandas.DataFrame,replace_mean = False):
    '''
    Cleans a dataframe in-place by filling in missing values with the mean (if numeric) or most frequent value (if not)
    '''
    means = df.mean(numeric_only=True)
    modes = df.mode()

    for i in df.columns:
        if pandas.api.types.is_numeric_dtype(df[i]):
            df[i] = df[i].map(lambda item: (50*means[i] if replace_mean else means[i]) if pandas.isna(item) else item)
        else:
            df[i] = df[i].map(lambda item: modes[i][0] if pandas.isna(item) else item)



def tag_columns(df: pandas.DataFrame) -> dict:
    '''
    Guesses the column type of each column of a dataframe:
    Columns with all unique values that aren't floating-point are likely IDs & shouldn't be included
    '''
    result = {}
    for i in df.columns:
        if len(pandas.unique(df[i])) == len(df) and ((not pandas.api.types.is_numeric_dtype(df[i])) or all(int(x) == x for x in df[i])):
            result[i] = ColumnType.ID
        elif pandas.api.types.is_numeric_dtype(df[i]):
            result[i] = ColumnType.NUMERIC
        else:
            result[i] = ColumnType.CATEGORICAL

    return result

def load_clean(filename,**kwargs):
    '''
    Drop-in replacement for pd.read_csv that also fills in missing values,
    converts columns to numeric types, and drops ID columns. Also takes the same kwargs!
    '''
    df = pandas.read_csv(filename,**kwargs) #.head()
    clean(df,True)
    tags = tag_columns(df)

    id_cols = [i for i in df.columns if tags[i] == ColumnType.ID]
    df = df.drop(columns=id_cols)

    cat_cols = [i for i in df.columns if tags[i] == ColumnType.CATEGORICAL]
    #df = pandas.get_dummies(df,columns=cat_cols)

    le = LabelEncoder()
    for i in cat_cols:
        df[i] = le.fit_transform(df[[i]].to_numpy().ravel())


    return df #removed tags from return value because it's never being used anywhere.

if __name__ == '__main__':
    df = load_clean('data/bnp-paribas/train.csv')

    train, test = train_test_split(df,test_size=0.2)
    reg = logistic_regression(train,'target')
    print("baseline: ", df['target'].mean())
    print(test_regression(test,reg.score,'target'))
