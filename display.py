import os
import pandas as pd
import matplotlib.pyplot as plot

dataset_names = os.listdir('results')

for name in dataset_names:
    df = pd.read_csv(f'results/{name}')

    unique_rows = []
    seentypes = set()
    for idx,row in df.iterrows():
        if row['type'] not in seentypes:
            seentypes.add(row['type'])
            unique_rows.append(row)

    figure = plot.figure(figsize = (10,5))

    models = [i['type'] for i in unique_rows[:5]]

    losses = [i['cost'] for i in unique_rows[:5]]

    plot.bar(models,losses, width = 0.5)

    #print(models,losses)

    plot.xlabel("Model")

    plot.ylabel("Loss")

    plot.title(f"{name}: Best 5 losses")
    plot.show()
