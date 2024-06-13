import kaggle as kg
import boto3
from dotenv import load_dotenv
import os

load_dotenv(verbose=True, dotenv_path='./src/.env')
my_bucket = os.getenv('S3_BUCKET_ADDRESS')
print(f"S3_BUCKET_ADDRESS: {my_bucket}")

print('Pulling data from Kaggle')

# Checking authentication
try:
    kg.api.authenticate()
    print("Authentication to Kaggle successful!")
except Exception as e:
    print(f"Authentication failed! Error: {e}")

# Attempting download
try:
    file_path = './data_raw/'
    kg.api.dataset_download_files(dataset="edizaguirre/plants-dataset",
                                  path=file_path,
                                  unzip=True)
    print(f"File download successful! Data is in {file_path}")
except Exception as e:
    print(f"Download failed! Error: {e}")

train_tfrecord_file = f'{file_path}leaves.tfrecord'
val_tfrecord_file = f'{file_path}test_leaves.tfrecord'

# Create an S3 client
s3 = boto3.resource('s3')

print('Attempting upload to S3 bucket:')

try:
    my_bucket = os.getenv('S3_BUCKET_ADDRESS')
    if my_bucket is None:
        raise ValueError("Environment variable 'S3_BUCKET_ADDRESS' not set")
except Exception as e:
    print(f"Error getting environment variable: {e}")

try:
    s3.Bucket(my_bucket).upload_file(train_tfrecord_file, 'raw_data/leaves.tfrecord')
    s3.Bucket(my_bucket).upload_file(val_tfrecord_file, 'raw_data/test_leaves.tfrecord')
except Exception as e:
    print(f"Error uploading files to S3: {e}")
else:
    print(f"Successfully uploaded raw data to {my_bucket}/raw_data")
