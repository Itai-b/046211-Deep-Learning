import os
import cv2
import ultralytics
from ultralytics import YOLO


## Load Video potential:
#import kagglehub
# Download latest version
#path = kagglehub.dataset_download("gracehephzibahm/pothole-severity-classification")
#print("Path to dataset files:", path)

## Code snippet to use for severity boxes in non-yolo models
#if label.item() == PotholeSeverityLevels.MINOR_POTHOLE.value:
#    color_rgb = (21,176,26) # plt's #15b01a GREEN for MINOR POTHOLE
#if label.item() == PotholeSeverityLevels.MEDIUM_POTHOLE.value:
#    color_rgb = (255,166,0) # plt's #ffa500 ORANGE for MEDIUM POTHOLE
#if label.item() == PotholeSeverityLevels.MAJOR_POTHOLE.value:
#    color_rgb = (229,0,0) # plt's #e50000 RED for MAJOR POTHOLE

# For matplotlib RGB:
colors_rgb = {
    'minorpothole': (21,176,26),   #15b01a GREEN
    'mediumpothole': (255,166,0),  #ffa500 ORANGE
    'majorpothole': (229,0,0)      #e50000 RED
}

# For CV2 BGR:
colors_bgr = {
    'minorpothole': (26,176,21),   #15b01a GREEN
    'mediumpothole': (0,166,255),  #ffa500 ORANGE
    'majorpothole': (0,0,229)      #e50000 RED
}


def video_inference(input_video_path, output_video_path, model_name, model_path):
    if model_name == 'YOLO':
        model = YOLO(model_path)


    # Open video
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Run YOLO with streaming
    if model_name == 'YOLO':
        for result in model.predict(source=input_video_path, stream=True, conf=0.5, verbose=False):
            frame = result.plot()  # Draw bounding boxes
            out.write(frame)  # Save frame to output video

    else:
        # TODO: include the data_process show function for the rcnn resnet model
        frame = 0

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Processed video saved as: {output_video_path}")
        
