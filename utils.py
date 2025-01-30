import os
import json
import shutil
import xmltodict
import cv2
import numpy as np
import motion_blur

def calc_model_size(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    model_size_mb = (param_size + buffer_size) / 1024 ** 2
    
    return num_params, model_size_mb

def update_json(history, model_name, json_name="models_data.json", save_path=""):
    # Construct the full path to the JSON file
    json_path = os.path.join(save_path, json_name)
    
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # If the JSON file doesn't exist, create it with an empty dictionary
    if not os.path.exists(json_path):
        with open(json_path, 'w') as f:
            json.dump({}, f)
    
    # Load the existing JSON data, handling empty or invalid JSON files
    try:
        with open(json_path, 'r') as f:
            content = f.read().strip()  # Handle files with only whitespace
            data = json.loads(content) if content else {}
    except json.JSONDecodeError:
        print(f"Warning: {json_name} is corrupted. Initializing it as an empty dictionary.")
        data = {}

    # Update the JSON data with the new history
    data[model_name] = history

    # Save the updated JSON back to the file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def voc_to_yolo(voc_path, yolo_path):
    """
    Convert Annotations to YOLO Format
    YOLO format stores annotations in .txt files, with each line representing a bounding box in the format:
    class x_center y_center width height
    """
    os.makedirs(yolo_path, exist_ok=True)
    for xml_file in os.listdir(voc_path):
        if not xml_file.endswith('.xml'):
            continue
        with open(os.path.join(voc_path, xml_file)) as f:
            voc_data = xmltodict.parse(f.read())
        
        # Get image dimensions
        img_filename = voc_data['annotation']['filename']
        #img_path = os.path.join(img_dir, img_filename) # not needed
        img_width = int(voc_data['annotation']['size']['width'])
        img_height = int(voc_data['annotation']['size']['height'])

        # Parse annotations
        annotations = voc_data['annotation'].get('object', [])
        if not isinstance(annotations, list):
            annotations = [annotations]

        yolo_annotations = []
        for obj in annotations:
            cls = 0  # Assuming a single class (pothole), TODO: add the pothole severities
            bbox = obj['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            yolo_annotations.append(f"{cls} {x_center} {y_center} {width} {height}")

        # Save YOLO annotations
        yolo_filename = os.path.splitext(xml_file)[0] + '.txt'
        with open(os.path.join(yolo_path, yolo_filename), 'w') as f:
            f.write('\n'.join(yolo_annotations))


def organize_split_from_json(json_path, base_dir="data", output_dir="splitted_data"):
    """
    Organize images, annotations, and YOLO labels into train/val/test directories based on a JSON file.

    Args:
        json_path (str): Path to the JSON file containing the dataset split indices.
        base_dir (str): Base directory containing the original 'images', 'annotations', and 'yolo_labels'.
        output_dir (str): Base directory for the organized dataset splits.
    """
    with open(json_path, "r") as f:
        split_data = json.load(f)
    
    # Define input directories
    input_dirs = {
        "images": os.path.join(base_dir, "images"),
        "annotations": os.path.join(base_dir, "annotations"),
        "yolo_labels": os.path.join(base_dir, "yolo_labels"),
    }
    
    # Define output directories
    splits = ["train", "val", "test"]
    output_dirs = {split: {
        "images": os.path.join(output_dir, split, "images"),
        "annotations": os.path.join(output_dir, split, "annotations"),
        "labels": os.path.join(output_dir, split, "labels"),  # YOLO needs this folder as 'labels'
    } for split in splits}
    
    # Create output directories
    for split in splits:
        for category in output_dirs[split]:
            os.makedirs(output_dirs[split][category], exist_ok=True)
    
    # Copy files to respective directories
    for split, indices in split_data.items():
        for idx in indices:
            # Input file paths
            image_file = os.path.join(input_dirs["images"], f"img-{idx}.jpg")
            annotation_file = os.path.join(input_dirs["annotations"], f"img-{idx}.xml")
            yolo_label_file = os.path.join(input_dirs["yolo_labels"], f"img-{idx}.txt")
            
            # Output file paths
            image_dest = os.path.join(output_dirs[split]["images"], f"img_{idx}.jpg")
            annotation_dest = os.path.join(output_dirs[split]["annotations"], f"img_{idx}.xml")
            yolo_label_dest = os.path.join(output_dirs[split]["labels"], f"img_{idx}.txt")
            
            # Copy files if they exist
            for src, dest in [
                (image_file, image_dest),
                (annotation_file, annotation_dest),
                (yolo_label_file, yolo_label_dest)
            ]:
                if os.path.exists(src):
                    shutil.copy(src, dest)
    
    print(f"Data organized into {output_dir} with structure:")
    print(f"  train/images, train/annotations, train/labels")
    print(f"  val/images, val/annotations, val/labels")
    print(f"  test/images, test/annotations, test/labels")

def ensure_directory_clean(path):
    """Ensure the directory is clean by removing it if it exists and recreating it."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def copy_labels(source_label_dir, target_label_dir):
    """Copy all label files from source to target."""
    ensure_directory_clean(target_label_dir)
    for filename in os.listdir(source_label_dir):
        if filename.lower().endswith(('.txt', '.xml')):  # Adjust extensions if needed
            source_path = os.path.join(source_label_dir, filename)
            target_path = os.path.join(target_label_dir, filename)
            shutil.copyfile(source_path, target_path)

def noise_test(source_dir, target_dir, kernel_type, amp=0):
    """Apply noise, save blurred images, and copy corresponding labels."""
    
    source_image_dir = os.path.join(source_dir, 'images')
    source_label_dir = os.path.join(source_dir, 'labels')
    source_annot_dir = os.path.join(source_dir, 'annotations')
    target_image_dir = os.path.join(target_dir, 'images')
    target_label_dir = os.path.join(target_dir, 'labels')
    target_annot_dir = os.path.join(target_dir, 'annotations')
    
    #create target directories
    os.makedirs(target_image_dir, exist_ok=True)
    os.makedirs(target_label_dir, exist_ok=True)
    os.makedirs(target_annot_dir, exist_ok=True)
    
    for filename in os.listdir(source_image_dir):
        if filename.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(source_image_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                blurred_image = motion_blur.motion_blur(image, kernel_type=kernel_type, a=amp)
                output_path = os.path.join(target_image_dir, filename)
                cv2.imwrite(output_path, blurred_image)
            else:
                print(f"Failed to load image: {filename}")

    # Copy corresponding labels
    copy_labels(source_label_dir, target_label_dir)
    copy_labels(source_annot_dir, target_annot_dir)
    
def add_noise_to_test(data_dir):
    """
    Add different motion blur noise to the test set.
    """
    test_dir = os.path.join(data_dir, 'test')
    # Run tests
    noise_test(test_dir, os.path.join(data_dir, 'test_nat'), kernel_type='natural')
    noise_test(test_dir, os.path.join(data_dir, 'test_eli001'), kernel_type='ellipse', amp=0.01)
    noise_test(test_dir, os.path.join(data_dir, 'test_eli005'), kernel_type='ellipse', amp=0.05)
    noise_test(test_dir, os.path.join(data_dir, 'test_eli01'), kernel_type='ellipse', amp=0.1)
    noise_test(test_dir, os.path.join(data_dir, 'test_uni001'), kernel_type='uniform', amp=0.01)
    noise_test(test_dir, os.path.join(data_dir, 'test_uni005'), kernel_type='uniform', amp=0.05)
    noise_test(test_dir, os.path.join(data_dir, 'test_uni01'), kernel_type='uniform', amp=0.1)


if __name__ == "__main__":
    voc_to_yolo(
        voc_path='data/chitholian_annotated_potholes_dataset/annotations',
        yolo_path='data/chitholian_annotated_potholes_dataset/yolo_labels'
    )
    organize_split_from_json(
        json_path='./data/chitholian_annotated_potholes_dataset/our_split.json',
        base_dir='./data/chitholian_annotated_potholes_dataset/',
        output_dir='./data/potholes_dataset'
    )

