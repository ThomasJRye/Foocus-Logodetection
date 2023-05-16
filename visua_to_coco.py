import json



def convert_to_coco(input_data):
    detections = input_data["data"]["detections"]
    images = []
    annotations = []
    categories = []

    category_ids = set()

    for index, detection in enumerate(detections):
        image_id = index + 1
        image = {
            "id": image_id,
            "width": input_data["data"]["mediaInfo"]["width"],
            "height": input_data["data"]["mediaInfo"]["height"],
            "file_name": f"frame_{image_id}.jpg"
        }
        images.append(image)

        print(detection["coordinates"][0])

        x, y, width, height = calculate_bbox(detection["coordinates"][0])
        
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
    x1 = coordinates[0]
    y1 = coordinates[0]
    return x_min, y_min, x_max - x_min, y_max - y_min

with open("visua_analyses/1383.json") as f:
    input_data = json.load(f)
   

coco_output = convert_to_coco(input_data)

with open("coco_annotations.json", "w") as outfile:
    json.dump(coco_output, outfile, indent=2)

print("COCO annotations file saved as 'coco_annotations.json'")

