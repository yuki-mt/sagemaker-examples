import argparse
from datetime import datetime
import os

from sagemaker.tensorflow import TensorFlow

MODEL_NAME = 'mymodel'
BUCKET_NAME = 'your_bucket_name'
HYPER_PARAMS = {
    'batch_size': 64,
}
IAM_ROLE = os.getenv('AWS_IAM_ROLE')
MAX_RUN_TIME = 24 * 60 * 60 * 3  # 3 days


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, required=True,
                        help='s3 path of training dataset (e.g. s3://...')
    parser.add_argument('--instance_type', '-i', type=str, default='ml.p2.xlarge')

    return parser.parse_args()


def main():
    args = get_args()

    job_name = MODEL_NAME + '-' + datetime.now().strftime('%Y-%m-%d-%H%M%S')
    code_location = 's3://{}/{}/jobs'.format(BUCKET_NAME, MODEL_NAME)
    output_path = code_location + '/'

    # local mode does not support spot training.
    is_spot = args.instance_type != 'local'
    checkpoint_s3_path = '{}{}/checkpoint/'.format(output_path, job_name) if is_spot else None

    estimator = TensorFlow(
        entry_point="entrypoints/tf.py",
        role=IAM_ROLE,
        code_location=code_location,
        train_instance_count=1,
        framework_version="1.11.0",  # this is version of tensorflow
        py_version="py3",
        train_instance_type=args.instance_type,
        train_use_spot_instances=is_spot,
        train_max_run=MAX_RUN_TIME,
        train_max_wait=MAX_RUN_TIME,
        checkpoint_s3_uri=checkpoint_s3_path,
        script_mode=True,
        dependencies=['requirements.txt'],  # set more path to file or directory to upload them
        output_path=output_path,
        hyperparameters=HYPER_PARAMS)

    estimator.fit(inputs=args.dataset_path, job_name=job_name)


if __name__ == '__main__':
    main()
