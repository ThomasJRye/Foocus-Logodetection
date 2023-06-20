import os
import sys
import mysql.connector
import boto3
from dotenv import load_dotenv
from visua_to_coco import convert_to_coco
import json
from tqdm import tqdm
import cv2

def import_from_S3(S3directory, filename, outputDirectory=None):
    try:

        # Access specific S3 bucket
        bucket_name = os.getenv('AWS_BUCKET')

        # Create an S3 client
        s3 = boto3.client('s3')

        # Construct the source path
        source = S3directory + filename

        # Check if the file exists in S3
        response = s3.head_object(Bucket=bucket_name, Key=source)
        error_code = getattr(response, 'response', {}).get('Error', {}).get('Code')
        output_directory = os.path.join(outputDirectory, '') if outputDirectory else ''

        if error_code == '404':
            response = s3.head_object(Bucket=bucket_name, Key=source.replace('visua/', ''))
            error_code = getattr(response, 'response', {}).get('Error', {}).get('Code')

            if error_code == '404':
                print(source, "not found")
                return False
            else:   
                print(bucket_name, source, output_directory + filename)
                s3.download_file(bucket_name, source.replace('visua/', ''), output_directory + filename)
                return True


        print(bucket_name, source, output_directory + filename)
        s3.download_file(bucket_name, source, output_directory + filename)

        # Return True to indicate successful download
        return True
    
    except FileNotFoundError:
        print(f"File not found on S3: {filename}")
        return False
    except Exception as e:
        print(f"Error occurred during S3 download: {str(e)}")
        # Return False to indicate failure in download
        return False

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
        number_of_videos = 1000
        query = "SELECT logograb_videos.url, logograb_video_analysis.json_url FROM logograb_videos INNER JOIN logograb_video_analysis ON logograb_videos.id = logograb_video_analysis.video_id ORDER BY logograb_videos.id DESC LIMIT " + str(number_of_videos) + ";"
        print(query)
        cursor.execute(query)

        result = cursor.fetchall()
        image_id = 3000

        output_directory = sys.argv[1] if len(sys.argv) > 1 else None

        for video_url, analysis_url in result:
            found = True
            print(video_url, analysis_url)
            if analysis_url is not None:
                import_from_S3(S3directory="", filename=analysis_url, outputDirectory=output_directory)
                json_file_path = analysis_url
                # existing_coco_file = "existing_coco.json"  # Specify the path to an existing COCO JSON file if available
                with open(json_file_path) as f:
                    input_data = json.load(f)

                if video_url is not None:
                    filename = video_url.replace('http://spect8-static.s3.amazonaws.com/', '')
                    if not import_from_S3(S3directory="", filename=filename, outputDirectory=output_directory):
                        continue

                    video_path = os.path.join(output_directory, filename)  # Specify the path to the corresponding video file

                    image_id = convert_to_coco(input_data, os.path.join(output_directory, 'images'), video_path, image_id)

                    video_file = os.path.join(video_path, os.path.basename(filename))

                    if not os.path.isfile(video_file):
                        print(f"Video file '{video_file}' not found.")
                        continue

                    try:
                        # Open the video file
                        cap = cv2.VideoCapture(video_file)

                        # Check if the video file was successfully opened
                        if not cap.isOpened():
                            print(f"Failed to open video file '{video_file}'")
                            continue

                        # Read frames from the video
                        while True:
                            ret, frame = cap.read()

                            # Check if a frame was successfully read
                            if not ret:
                                print("Failed to capture frame from video")
                                break

                            # Process the frame here

                        # Release the video file capture
                        cap.release()
                        os.remove(video_file)

                    except Exception as e: 
                        print(f"An error occurred while processing the video: {str(e)}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
