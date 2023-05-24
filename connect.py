import os
from dotenv import load_dotenv
import mysql.connector
import boto3
from mysql.connector import errors


def connect_to_s3():
    # load environment variables from .env file
    load_dotenv()

    # access specific s3 bucket
    bucket_name = os.getenv('AWS_BUCKET')

    # create an S3 client
    s3 = boto3.client('s3')

    return s3, bucket_name

def list_files_in_folder(folder_name):
    # connect to S3
    s3, bucket_name = connect_to_s3()

    # list all the objects in the specified folder
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)

    files = []
    
    # print the filenames of all the objects in the specified folder
    for obj in response['Contents']:
        files.append(obj['Key'])

    return files

def download_file_from_s3(s3_path, local_path):
    # connect to S3
    s3, bucket_name = connect_to_s3()

    # create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # download the file from S3
    s3.download_file(bucket_name, s3_path, local_path)

    print(f'Downloaded {s3_path} to {local_path}')


    print(f'Downloaded {s3_path} to {local_path}')
    
def database():

    # Database credentials
    DB_CONNECTION = os.getenv('DB_CONNECTION')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DB_DATABASE = os.getenv('DB_DATABASE')
    DB_USERNAME = os.getenv('DB_USERNAME')
    DB_PASSWORD = os.getenv('DB_PASSWORD')

    # Connect to the database
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_DATABASE,
            user=DB_USERNAME,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM foqus;")
        rows = cursor.fetchall()
        cursor.close()
        print(rows)
        return cursor
    except errors.ProgrammingError as err:
        print(f"Error connecting to MySQL: {err}")

