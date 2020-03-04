from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput

instance_type = "ml.c5.xlarge"
# instance_type = "local"

code_location = 's3://your_bucket/jobs'

train_data_path = FileSystemInput('fs-xxx', 'FSxLustre', '/fsx', 'ro')

role = 'IAM_ROLE'

estimator = TensorFlow(
    entry_point="entrypoints/fsx.py",
    base_job_name='my-test',
    role=role,
    code_location=code_location,
    train_instance_count=1,
    framework_version="1.11.0",  # this is version of tensorflow
    py_version="py3",
    train_instance_type=instance_type,
    train_use_spot_instances=True,
    train_max_wait=86400,
    script_mode=True,
    subnets=['subnet-xxx', 'subnet-xxx'],
    output_path=code_location + '/',
    hyperparameters={'batch_size': 64, 'epochs': 2})


estimator.fit(train_data_path)
