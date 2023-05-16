import json
import cv2
import matplotlib.pyplot as plt
import numpy as np



def convert_to_coco(input_data):
    detections = input_data["data"]["detections"]
    images = []
    annotations = []
    categories = []

    category_ids = set()

    for index, detection in enumerate(detections):
        if detection["confidence"] < 0.8 and detection["confidence"] != 'tiny':
            continue
        image_id = index + 1
        image = {
            "id": image_id,
            "width": input_data["data"]["mediaInfo"]["width"],
            "height": input_data["data"]["mediaInfo"]["height"],
            "file_name": f"frame_{image_id}.jpg"

        }
        images.append(image)

        print(detection["coordinates"][0])
        print(detection["name"])
        x, y, width, height = calculate_bbox(detection["coordinates"][0])
        timeBegin = detection["timeBegin"]
        timeEnd = detection["timeEnd"]

        frame = getFrame(timeBegin, 30, "/Users/thomasrye/Documents/github/Foocus-Logodetection/videos/210509_Eurosport_Fotballdirekte_Eliteserien.mp4")
        plot(frame, [x, y, width, height])


        annotation = {
            "id": index + 1,
            "image_id": image_id,
            "category_id": detection["visualClassId"],
            "bbox": [x, y, width, height],
            "area": detection["area"],
            "iscrowd": 0,
        }
        annotations.append(annotation)


        if detection["visualClassId"] not in category_ids:
            categories.append({
                "id": detection["visualClassId"],
                "name": detection["name"]
            })
            category_ids.add(detection["visualClassId"])

    coco_output = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco_output

def calculate_bbox(coordinates):
    x_coords = []
    y_coords = []

    even = True
    for i in range(len(coordinates)):
        if even:
            x_coords.append(coordinates[i])
        else:
            y_coords.append(coordinates[i])
        even = not even

    min_x = min(x_coords)
    min_y = min(y_coords)
    max_x = max(x_coords)
    max_y = max(y_coords)

    # Calculate the four corner points of the bounding box
    top_left = [min_x, min_y]
    top_right = [max_x, min_y]
    bottom_right = [max_x, max_y]
    bottom_left = [min_x, max_y]

    return [top_left, top_right, bottom_right, bottom_left]


def getFrame(time, fps, video_url):
    cap = cv2.VideoCapture(video_url)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    time_in_seconds = sum(float(x) * 60 ** i for i, x in enumerate(time.split(":")[::-1]))
    frame_number = min(int(time_in_seconds * fps), total_frames - 1)
    print(f"time: {time}, frame_number: {frame_number}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        return frame
    else:
        print(f"Failed to capture frame at time {time}.")
        return None




def plot(frame, rect=None):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print("plotting:")
    print(rect)

    # Plot the frame using matplotlib
    
    if rect is not None:
        rect = np.array(rect, dtype=np.int32)
        color = (0, 0, 255)
        thickness = 2
        cv2.polylines(frame, [rect], True, color, thickness)
    plt.imshow(frame)
    plt.axis("off")
    plt.show()

def video_to_frames(video, framecount = 100):

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set the starting frame number
    frame_number = 0

    # Loop through each frame in the video
    while video.isOpened():
        # Read the next frame
        ret, frame = video.read()

        # Check if frame was read successfully
        if ret:
            # Save the frame as a JPEG image
            filename = f"frame_{frame_number:06d}.jpg"
            cv2.imwrite('./images/' + filename, frame)

            # Increment the frame number
            frame_number += 1
        else:
            break

video_to_frames(cv2.VideoCapture("/Users/thomasrye/Documents/github/Foocus-Logodetection/videos/210509_Eurosport_Fotballdirekte_Eliteserien.mp4"))

with open("visua_analyses/1383.json") as f:
    input_data = json.load(f)
   

coco_output = convert_to_coco(input_data)

with open("coco_annotations.json", "w") as outfile:
    json.dump(coco_output, outfile, indent=2)

print("COCO annotations file saved as 'coco_annotations.json'")

