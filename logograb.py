import cv2
from detect_utils import predict, draw_boxes
from model import get_model

def count_detections(video_path, detection_threshold=0.5, device='cpu', model_name='v1'):
    # Load the model.
    model = get_model(device=device, model_name=model_name)

    # Open the video file.
    video = cv2.VideoCapture(video_path)

    # Get the video properties.
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to save the output video.
    output_path = 'output.mp4'
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    category_counts = {}

    while True:
        # Read the next frame from the video.
        ret, frame = video.read()

        if not ret:
            break

        # Convert the frame to PIL Image format.
        frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection on the frame.
        boxes, classes, _ = predict(frame_pil, model, device, detection_threshold)

        # Count the detections for each category.
        for category in classes:
            category_counts[category] = category_counts.get(category, 0) + 1

        # Draw bounding boxes on the frame.
        frame_with_boxes = draw_boxes(boxes, classes, _, frame)

        # Write the frame with bounding boxes to the output video.
        output.write(frame_with_boxes)

        # Increment the frame count.
        frame_count += 1
        print(f'Processed frame {frame_count}/{total_frames}')

    # Release the video capture and output writer objects.
    video.release()
    output.release()

    return category_counts, output_path

# Example usage
video_path = 'path/to/video.mp4'
category_counts, output_path = count_detections(video_path, detection_threshold=0.5, device='cuda', model_name='v1')

# Print the category counts
for category, count in category_counts.items():
    print(f'{category}: {count}')

print(f'Output video saved to: {output_path}')
