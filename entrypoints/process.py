import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--local', action='store_true')
parser.add_argument('--option', action='store_true')
parser.add_argument('--ratio', '-r', type=float, default=0.2)
args = parser.parse_args()

if args.local:
    base_path = './'
else:
    base_path = '/opt/ml/processing/'


df = pd.read_csv(base_path + 'input/dataset.csv')

added = df.copy()
added['new'] = 1
train, test = train_test_split(added, test_size=args.ratio)
train, validation = train_test_split(train, test_size=args.ratio)

os.makedirs(base_path + 'output/train', exist_ok=True)
os.makedirs(base_path + 'output/validation', exist_ok=True)
os.makedirs(base_path + 'output/test', exist_ok=True)

train.to_csv(base_path + "output/train/train.csv")
validation.to_csv(base_path + "output/validation/validation.csv")
test.to_csv(base_path + "output/test/test.csv")
print('Finished running processing job')
