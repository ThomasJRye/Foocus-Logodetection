
def import_from_S3(S3directory, filename, outputDirectory):

    # load environment variables from .env file
    load_dotenv()

    # access specific s3 bucket
    bucket_name = os.getenv('AWS_BUCKET')

    # create an S3 client
    s3 = boto3.client('s3')

    # list all of the buckets in your account
    response = s3.list_buckets()
    print(response)


    print(S3directory + filename)

    s3.download_file(bucket_name, S3directory + filename, outputDirectory + filename)

def importer(number):
    
    load_dotenv()
    host=os.getenv('DB_HOST')

    conn = mysql.connector.connect(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USERNAME'),
        password= os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_DATABASE'),
        port=os.getenv('DB_PORT')
    )

    cursor = conn.cursor()

    query = "SELECT url FROM logograb_videos LIMIT 10;"

    cursor.execute(query)

    result = cursor.fetchall()
    video_url = result[0][0]
    filename = video_url.replace('http://spect8-static.s3.amazonaws.com/', '')
    print(filename)

    import_from_S3('', filename, '.\\visua\\videos\\')




def main():
    # Your program's main logic goes here
    importer(2)



    if __name__ == "__main__":
        main()                 