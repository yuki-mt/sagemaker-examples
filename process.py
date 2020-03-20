from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.session import Session
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os


instance_type = "ml.c5.xlarge"
role = 'ROLE_ARN'
base_path = '/opt/ml/processing'
input_path = os.path.join(base_path, 'input')
output_path = os.path.join(base_path, 'output')
bucket_name = Session().default_bucket()
dest_path = f's3://{bucket_name}/process/output'

sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     role=role,
                                     base_job_name='process-job-test',
                                     instance_count=1,
                                     instance_type=instance_type)

sklearn_processor.run(
    code='entrypoints/process.py',
    arguments=['-r', '0.1'],
    inputs=[
        ProcessingInput(source='dataset/dataset.csv', destination=input_path)
    ],
    outputs=[
        ProcessingOutput(source=os.path.join(output_path), destination=dest_path)
    ]
)
