from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput


instance_type = "ml.c5.xlarge"
role = 'ROLE_ARN'
base_path = '/opt/ml/processing/output'
dest_base_path = 's3://your_bucket/jobs/'

sklearn_processor = SKLearnProcessor(framework_version='0.20.0',
                                     role=role,
                                     base_job_name='matoba-job-test',
                                     instance_count=1,
                                     instance_type=instance_type)

sklearn_processor.run(
    code='entrypoints/process.py',
    arguments=['-r', '0.1'],
    inputs=[ProcessingInput(
        source='input/dataset.csv',
        destination=base_path + 'input')],
    outputs=[ProcessingOutput(source=base_path + 'train',
                              destination=dest_base_path + 'train'),
             ProcessingOutput(source=base_path + 'validation',
                              destination=dest_base_path + 'validation'),
             ProcessingOutput(source=base_path + 'test',
                              destination=dest_base_path + 'test')]
)
