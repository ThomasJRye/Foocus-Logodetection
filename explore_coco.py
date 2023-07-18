import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import requests
from pycocotools.coco import COCO
import os

def main():
    coco_annotation_file_path = "Thomas_Foocus_COCO/instances_default.json"
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)

    # Category IDs.
    cat_ids = coco_annotation.getCatIds()
    print(f"Number of Unique Categories: {len(cat_ids)}")
    print("Category IDs:")
    print(cat_ids)  # The IDs are not necessarily consecutive.

    # All categories.
    cats = coco_annotation.loadCats(cat_ids)
    cat_names = [cat["name"] for cat in cats]
    print("Categories Names:")
    print(cat_names)

    # Iterate over all image IDs.
    img_ids = coco_annotation.getImgIds()
    i = 0
    for img_id in img_ids:
        if i < 0:
            i += 1
            continue
        img_info = coco_annotation.loadImgs([img_id])[0]
        img_file_name = img_info["file_name"]
        # image_path = os.getcwd() + '/revData/data/' + img_file_name
        image_path = os.getcwd() + '/' + img_file_name
        print(f"Processing Image ID: {img_id}, File Name: {img_file_name}, Image path: {image_path}")

        # Get all the annotations for the specified image.
        ann_ids = coco_annotation.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = coco_annotation.loadAnns(ann_ids)
        print(f"Annotations for Image ID {img_id}:")
        print(anns)

        try:
            # Use URL to load image.
            im = Image.open(image_path)

            # Save image and its labeled version.
            plt.axis("off")
            plt.imshow(np.asarray(im))

            # Plot segmentation and bounding box.
            coco_annotation.showAnns(anns, draw_bbox=True)

            # Add class labels to annotations.
            for ann in anns:
                category_id = ann["category_id"]
                if category_id > 32:
                    continue
                
                category = coco_annotation.loadCats(category_id)[0]
                class_name = category["name"]
                bbox = ann["bbox"]
                x, y, w, h = bbox
                plt.text(x, y - 10, class_name, color="red", fontsize=12, weight="bold")

            plt.savefig(f"{os.getcwd()}/annotated_images/{img_id}_annotated.jpg", bbox_inches="tight", pad_inches=0)
            plt.close()

        except IOError:
            print(f"Error: Image {img_id} does not exist in the directory.")

    return
    
if __name__ == "__main__":
    main()
