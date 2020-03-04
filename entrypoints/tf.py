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

    print('data stored in S3 is downloaded to ' + os.environ['SM_CHANNEL_TRAINING'])
    print('data files are:')
    print(os.listdir(os.environ['SM_CHANNEL_TRAINING']))

    """
    Write any code to train your model here
    """

    model_path = os.path.join(os.environ['SM_MODEL_DIR'], 'my_model.txt')
    print('model path is:', model_path)

    with open(model_path, 'w') as f:
        f.write('This file will be stored in S3 as output model')
