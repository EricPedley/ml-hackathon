import pandas as pd
import sqlite3
from io import StringIO
import re


database_path = 'data/ucimlrepo.db'

def find_target_col_index(name: str):
    '''
    Searches UCI ML Repo SQL metadata for a column labelled as the target for the database with the given folder name.
    If there is a marked target column, returns it, otherwise returns None.
    '''
    conn = sqlite3.connect(database_path)


    query = f'''SELECT b.id
               FROM donated_datasets a
               LEFT JOIN attributes b ON a.ID = b.datasetID
               WHERE a.name LIKE '{name}' OR a.name LIKE '{name.replace('-',' ')}'
    '''#gets all attribute ids for the named dataset
    all_ids = [i[0] for i in pd.read_sql(query,conn).values]

    query = f'''SELECT b.id
               FROM donated_datasets a
               LEFT JOIN attributes b ON a.ID = b.datasetID
               WHERE b.role='Target' AND (a.name LIKE '{name}' OR a.name LIKE '{name.replace('-',' ')}')
    '''#gets all the ids that have the target role
    result = pd.read_sql(query,conn)

    #I wish there was a better way but it seems like the IDs are ordered by the order of the columns.
    #Hopefully this works in enough situations?
    return all_ids.index(result.values[0]) if result.values.size > 0 else None

def find_train_data(name: str):
    '''
    Searches UCI ML Repo SQL metadata for a column labelled as the target for the database with the given folder name.
    Finds name of training data or None.
    '''
    conn = sqlite3.connect(database_path)

    name = name.title()#uppercases first letter of each word
    query = f'''SELECT b.trainFile
               FROM donated_datasets a
               LEFT JOIN tabular b ON a.ID = b.datasetID
               WHERE a.name LIKE '{name}' OR a.name LIKE '{name.replace('-',' ')}'
    '''
    result = pd.read_sql(query,conn)
    return result.values[0][0] if result.values.size > 0 else None

def find_legacy_info(name):
    '''
    Returns info from legacy database about given dataset.
    '''
    conn = sqlite3.connect(database_path)

    query = f'''SELECT *
               FROM datasets_old_schema a
               WHERE a.name LIKE '{name}' OR a.name LIKE '{name.replace('-',' ')}'
    '''
    result = pd.read_sql(query,conn)

    return result.values[0] if result.values.size > 0 else None

if __name__ == '__main__':
    print(find_target_col_index('adult'))
    #print(find_legacy_info('balloons'))
    print(find_train_data('Mushroom'))
