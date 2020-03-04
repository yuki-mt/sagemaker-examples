import argparse
import os
import subprocess


if __name__ == '__main__':
    if os.path.isfile('requirements.txt'):
        subprocess.run('pip install -r requirements.txt', shell=True)

    parser = argparse.ArgumentParser()
    # any hyperparameters sent by your notebook
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    args, _ = parser.parse_known_args()

    print('=========Environment Variables==========')
    print(os.environ)
    print('=========/Environment Variables==========')

    print('=========Root dir==========')
    for d in os.scandir('/opt/ml/input/data/training'):
        print(d.name)
    print('=========/Root dir==========')
