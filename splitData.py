import pandas as pd
from sys import argv as args

# Read the data
fileName = args[1] if len(args) > 1 else 'data/weblogs.csv'
df = pd.read_csv(fileName)
# Value counts
amntToSelect = 100

# Randomly select the amntToSelect rows for 1 and 0
df1 = df[df['ROBOT'] == 1].sample(amntToSelect)
df0 = df[df['ROBOT'] == 0].sample(amntToSelect)
fullDF = pd.concat([df1, df0])
# Remove the rows from the original data
df = df.drop(fullDF.index)
# Save the data
fullDF.to_csv('data/weblogs_test.csv', index=False)
df.to_csv('data/weblogs_train.csv', index=False)