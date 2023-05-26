import os
import mysql.connector
import boto3
from dotenv import load_dotenv
from visua_to_coco import convert_to_coco

def import_from_S3(S3directory, filename, outputDirectory):
    try:

        # load environment variables from .env file
        load_dotenv()

        # access specific s3 bucket
        bucket_name = os.getenv('AWS_BUCKET')

        # create an S3 client
        s3 = boto3.client('s3')

        # list all of the buckets in your account
        response = s3.list_buckets()
        print(response)


        # print(S3directory + filename)
        # # Foqus/spect8-static/video-analysis/113.json
        # source = "video-analysis/113.json"
        # s3.download_file(bucket_name, source, outputDirectory + filename)
        # Specify the S3 directory, filename, and output directory
        # S3directory = "video-analysis/"
        # filename = "113.json"
        # outputDirectory = "output/"

        # Construct the source path
        source = S3directory + filename

        # Download the file from S3
        print(source)
        dot_count = source.count('.')
        if dot_count == 2:
            print("The string contains exactly two dots.")
            filename = '.'.join(filename.split('.')[:-2])
        else:
            print("The string does not contain exactly two dots.")

        print(bucket_name, source, outputDirectory + filename)
        s3.download_file(bucket_name, source, outputDirectory + filename)

    except Exception as e:
            print(f"Error occurred during S3 download: {str(e)}")

def main():
    try:
        load_dotenv()

        conn = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USERNAME'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_DATABASE'),
            port=os.getenv('DB_PORT')
        )

        cursor = conn.cursor()

        number_of_videos = 10
        query = "select logograb_videos.url,logograb_video_analysis.json_url  from logograb_videos inner join logograb_video_analysis on logograb_videos.id = logograb_video_analysis.video_id limit " + str(number_of_videos) + ";"

        cursor.execute(query)

        result = cursor.fetchall()

        for video_url, analysis_url in result:
            print(video_url, analysis_url)
            if (analysis_url is not None):
                import_from_S3(S3directory='', filename=analysis_url, outputDirectory='')
                json_file_path = os.path.join(outputDirectory, analysis_url)
                images_path = "/path/to/images"  # Specify the desired output path for images
                video_path = "/path/to/video.mp4"  # Specify the path to the corresponding video file
                existing_coco_file = "existing_coco.json"  # Specify the path to an existing COCO JSON file if available
                with open(json_file_path) as f:
                    input_data = json.load(f)
                convert_to_coco(input_data, images_path, video_path, existing_coco_file)
            if (video_url is not None):
                filename = video_url.replace('http://spect8-static.s3.amazonaws.com/', '/videos/')
                import_from_S3(S3directory="visua/", filename=filename, outputDirectory="/videos")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()