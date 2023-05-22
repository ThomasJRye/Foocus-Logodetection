import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def convert_to_coco(input_data, images_path):
    detections = input_data["data"]["detections"]
    images = []
    annotations = []
    categories = []

    category_ids = set()

    for index, detection in enumerate(detections):
        if detection["confidence"] > 0.75 and detection["size"] != 'tiny':
            image_id = index + 1
            image = {
                "id": image_id,
                "width": input_data["data"]["mediaInfo"]["width"],
                "height": input_data["data"]["mediaInfo"]["height"],
                "file_name": f"{images_path}/frame_{image_id:06d}.jpg"
            }
            images.append(image)

            x, y, width, height = calculate_bbox(detection["coordinates"][0])
            timeBeginStr = detection["timeBegin"]
            timeEndStr = detection["timeEnd"]

            
            timeBegin = time_in_sec(timeBeginStr)
            timeEnd = time_in_sec(timeEndStr)
            
            video_path = "/Users/thomasrye/Documents/github/Foocus-Logodetection/videos/210509_Eurosport_Fotballdirekte_Eliteserien.mp4"
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            start_frame = getFrame(timeBegin, fps, video_path)
            end_frame = getFrame(timeEnd, fps, video_path)

            similarity_score = calculate_frame_similarity(start_frame, end_frame)
            if (similarity_score < 40):
                print("___")
                print(detection["name"])

                print(image_id)
                print(timeBeginStr)
                print(timeEndStr)
                print("score:", similarity_score)
                plot(start_frame, [x, y, width, height], detection["name"], f"{images_path}/frame_{image_id:06d}.jpg")

                annotation = {
                    "id": index + 1,
                    "image_id": image_id,
                    "category_id": detection["visualClassId"],
                    "bbox": [x, y, width, height],
                    "area": width*height,
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
    x_coords = coordinates[::2]
    y_coords = coordinates[1::2]

    min_x = min(x_coords)
    min_y = min(y_coords)
    width = max(x_coords) - min_x
    height = max(y_coords) - min_y

    return [min_x, min_y, width, height]

def time_in_sec(time):
    return sum(float(x) * 60 ** i for i, x in enumerate(time.split(":")[::-1]))

def getFrame(time, fps, video_url):
    cap = cv2.VideoCapture(video_url)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    frame_number = min(int(time * fps), total_frames - 1)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()

    if ret:
        return frame
    else:
        print(f"Failed to capture frame at time {time}.")
        return None

def plot(frame, rect=None, name=None, save_path=None):
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if rect is not None:
        x, y, w, h = map(int, rect)
        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if name is not None:
            cv2.putText(frame_rgb, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    plt.imshow(frame_rgb)
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def calculate_frame_similarity(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    return mse

# Remove the "images" directory
os.system("rm -r images")

# Create the "images" directory
os.makedirs("images")

with open("visua_analyses/1383.json") as f:
    input_data = json.load(f)

images_path = "./images"
coco_output = convert_to_coco(input_data, images_path)

with open("coco_annotations.json", "w") as outfile:
    json.dump(coco_output, outfile, indent=2)

print("COCO annotations file saved as 'coco_annotations.json'")
