import boto3
from loguru import logger
from src.constants import S3_BUCKET_NAME
import pickle

class S3:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.s3_resource = boto3.resource('s3')

    def check_file_exists_client(self, prefix: str, bucket_name: str=S3_BUCKET_NAME):
        objects = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        if 'Contents' in objects.keys():
            contents = objects['Contents']           
            return len(contents) > 0
        
        else:
            return False
    
    def check_file_exists_resource(self, key: str, bucket_name: str=S3_BUCKET_NAME):
        try:
            object = self.s3_resource.Object(bucket_name=bucket_name, key=key).get()
            if object:
                return True
        except self.s3_resource.meta.client.exceptions.NoSuchKey:
            return False
        except Exception as e:
            raise e
    
    def get_data(self, key: str, bucket_name: str=S3_BUCKET_NAME):
        if self.check_file_exists_resource(key, bucket_name):
            data = self.s3_resource.Object(bucket_name=bucket_name, key=prefix).get()['Body'].read()
            return data
        
        else:
            logger.error(f"'{key} not found in '{bucket_name}'")
            return

    def upload_file(self, source_path: str, destination_path: str, bucket_name: str=S3_BUCKET_NAME):

        return self.s3_resource.Bucket(bucket_name).upload_file(source_path, destination_path)
    
    def upload_data(self, content, destination_path, bucket_name=S3_BUCKET_NAME):

        return self.s3_resource.Object(bucket_name, destination_path).put(Body=content)