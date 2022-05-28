import os

data_folder_names = os.listdir('data')

def process_uci_dataset(folder_path):
    return 'uci'

def process_kaggle_dataset(folder_path):
    return 'kaggle'

for folder in data_folder_names:
    folder_contents = os.listdir(f'data/{folder}')
    is_uci_dataset = any('.names' in f for f in folder_contents)
    if is_uci_dataset:
        results = process_uci_dataset(f'data/{folder}')
    else:
        results = process_kaggle_dataset(f'data/{folder}')
    print(folder, results)