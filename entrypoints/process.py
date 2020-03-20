import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ratio', '-r', type=float, default=0.2)
args = parser.parse_args()

base_path = '/opt/ml/processing'
input_path = os.path.join(base_path, 'input')
output_path = os.path.join(base_path, 'output')

df = pd.read_csv(os.path.join(input_path, 'dataset.csv'))

df['new'] = 1
os.makedirs(output_path, exist_ok=True)

df.to_csv(os.path.join(output_path, 'out.csv'))
print('Finished running processing job')
