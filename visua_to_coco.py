import os
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def convert_to_coco(input_data, images_path, video_path, image_id, existing_coco_file=None):

    detections = input_data["data"]["detections"]

    categories = []

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    # grouped_detections = group_simultaneous_detections(detections, fps, video_path)
    index = 0

    index += 1
    

    found_detection = False
    # List to store rectangles of all detections
    all_rectangles = []
    timeBegin = 0
    for detection in detections:
        
        old_time = timeBegin
        timeBeginStr = detection["timeBegin"]
        timeBegin = time_in_sec(timeBeginStr)
        print(timeBegin)
        print(old_time)
        # your conditions for detection confidence, size, and area percentage
        if ((detection["confidence"] > 0.7 and detection["size"] != "tiny") or 
        (detection["confidence"] > 0.99 and detection["size"] == "tiny")) and detection["areaPercentage"] > 0.005 and timeBegin > (old_time + 3):
        # if ((detection["confidence"] > 0.7 and detection["size"] != "tiny") or 
        # (detection["confidence"] > 0.99 and detection["size"] == "tiny")) and detection["areaPercentage"] > 0.005:
                x, y, width, height = calculate_bbox(detection["coordinates"][0])

                current_rectangle = [x, y, width, height]
                similar_to_previous = False  # initialize the flag as False
                
                # Loop through all the previous rectangles
                for previous_detection in all_rectangles:
                    # If the names are the same and the similarity score is above the threshold
                    if previous_detection['name'] == detection["name"] and \
                    rectangle_similarity(current_rectangle, previous_detection['rectangle']) > 0.05:
                        similar_to_previous = True  # set the flag to True
                        break  # break the inner loop

                if similar_to_previous:
                    continue  # if flag is True, skip to next detection

                
                # timeEndStr = detection["timeEnd"]

                
                # timeEnd = time_in_sec(timeEndStr)
                
                

                start_frame = getFrame(timeBegin, fps, video_path)
                # end_frame = getFrame(timeEnd, fps, video_path)

                # similarity_score = calculate_frame_similarity(start_frame, end_frame)
                # if (similarity_score < 70):
                #     found_detection = True
                #     print("___")
                #     print(detection["name"])

                #     print(image_id)
                #     print(timeBeginStr)
                #     print(timeEndStr)
                #     print("score:", similarity_score)
                #     # groupFrame = plot(, [x, y, width, height], detection["name"])

                #     annotation = {
                #         "id": index + 1,
                #         "image_id": image_id,
                #         "category_id": detection["visualClassId"],
                #         "bbox": [x, y, width, height],
                #         "area": width*height,
                #         "iscrowd": 0,
                #     }
                #     # annotations.append(annotation)

                #     if detection["visualClassId"] not in category_ids:
                #         categories.append({
                #             "id": detection["visualClassId"],
                #             "name": detection["name"]
                #         })
                #         category_ids.add(detection["visualClassId"])
                # Append the current rectangle and the detection's name to the list

                all_rectangles.append({
                    'rectangle': current_rectangle,
                    'name': detection["name"]
                })
                image_id += 1

                # plt.imshow(start_frame)
                # plt.axis("off")
                # plt.savefig(f"{images_path}/{image_id:06d}.jpg", bbox_inches='tight', pad_inches=0, dpi=300)
                # plt.close()

                # Convert the image to a supported format (e.g., JPEG)
                image = Image.fromarray(start_frame)
                image = image.convert("RGB")
                image.save(images_path + '/' + str(image_id) + '.jpg', "JPEG")
                print("detected", detection["name"])
                
                # plot(start_frame, name=str(image_id) + ".jpg", save_path=images_path)
                # if found_detection:
                #     # Append the current rectangle and the detection's name to the list
                #     all_rectangles.append({
                #         'rectangle': current_rectangle,
                #         'name': detection["name"]
                #     })
                #     image_id += 1
                #     plt.imshow(start_frame)
                #     plt.axis("off")
                #     plt.savefig(f"{images_path}/{image_id:06d}.jpg", bbox_inches='tight', pad_inches=0, dpi=300)
                #     plt.close()

    return image_id


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
    # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = frame
    if rect is not None:
        x, y, w, h = map(int, rect)
        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if name is not None:
            cv2.putText(frame_rgb, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    if save_path is not None:
        cv2.imwrite(save_path, frame_rgb)

    return frame_rgb


def group_simultaneous_detections(detections, fps, cap):
    # Convert the `timeBegin` values to seconds and get the associated frames
    for detection in detections:
        detection['timeBeginSec'] = time_in_sec(detection['timeBegin'])
        detection['frame'] = getFrame(detection['timeBeginSec'], fps, cap)
        
    return group_simultaneous_detections_rec(detections[1:], fps, cap, [[detections[0]]])

def group_simultaneous_detections_rec(detections, fps, cap, grouped_detections):
    if len(detections) == 0:
        return grouped_detections

    detection = detections[0]
    timeBegin = detection['timeBeginSec']
    frame = detection['frame']

    for detection_group in grouped_detections:
        grouped_detection = detection_group[0]
        # groupTimeBegin = grouped_detection['timeBeginSec']
        groupFrame = grouped_detection['frame']
        # if abs(groupTimeBegin - timeBegin) < 50:
        similarity_score = calculate_frame_similarity(frame, groupFrame)

        if similarity_score < 75:
            detection_group.append(detection)
            return group_simultaneous_detections_rec(detections[1:], fps, cap, grouped_detections)

    new_group = [detection]
    grouped_detections.append(new_group)

    return group_simultaneous_detections_rec(detections[1:], fps, cap, grouped_detections)

def rectangle_similarity(coordinatesA, coordinatesB):
    bbox1 = calculate_bbox(coordinatesA)
    bbox2 = calculate_bbox(coordinatesB)

    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bbox1[0], bbox2[0])
    yA = max(bbox1[1], bbox2[1])
    xB = min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2])
    yB = min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = bbox1[2] * bbox1[3]
    boxBArea = bbox2[2] * bbox2[3]

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou



def calculate_frame_similarity(frame1, frame2):
    mse = np.mean((frame1 - frame2) ** 2)
    return mse


# Remove the "images" directory
# os.system("rm -r images")

# Create the "images" directory
# os.makedirs("images")

with open("visua_analyses/1383.json") as f:
    input_data = json.load(f)

# images_path = "./bigData1/images"
# video_path = "/Users/thomasrye/Documents/github/Foocus-Logodetection/videos/210509_Eurosport_Fotballdirekte_Eliteserien.mp4"
# existing_coco_file = "bigData1/labels.json"

# convert_to_coco(input_data, images_path, video_path)
