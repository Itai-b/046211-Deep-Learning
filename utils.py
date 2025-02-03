import os
import json
import shutil
import xmltodict
import cv2
import numpy as np
import pandas as pd
from IPython.display import display, HTML 
import matplotlib.pyplot as plt
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

def update_json(history, model_name, json_name="models_data.json", save_path="./"):
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
            name = obj['name']
            if name == 'minor_pothole' or name == 'pothole':
                cls = 0
            elif name == 'medium_pothole':
                cls = 1
            elif name == 'major_pothole':
                cls = 2
            else:
                raise Exception(f"Error in creating yolo annotations from VOC. Name was: {name}, expected: minor/medium/major_pothole.")
            
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


def create_augmented_train(data_dir):
    """
    Create an augmented train set for YOLO (not supporting kornia).
    The augmented train is created using random motion blur to an image with Kornia.
    """
    source_dir = os.path.join(data_dir, 'train')
    target_dir = os.path.join(data_dir, 'train_augmented')

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
    
    # Iterate over images in train and create an augmented copy of them under 'train_augmented'
    for filename in os.listdir(source_image_dir):
        if filename.lower().endswith(('.png', '.jpg')):
            image_path = os.path.join(source_image_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                blurred_image = motion_blur.motion_blur(image, kernel_type='kornia')
                output_path = os.path.join(target_image_dir, filename)
                cv2.imwrite(output_path, blurred_image)
            else:
                print(f"Failed to load image: {filename}")

    # Copy corresponding labels to the 'train_augmented' dir
    copy_labels(source_label_dir, target_label_dir)
    copy_labels(source_annot_dir, target_annot_dir)


def plot_test_results(noise_type=None, show_augmentations=False, show_severity=False):
    # Ensure plot is centered
    display(HTML("<style>.output_wrapper, .output { display: flex; justify-content: center; }</style>"))
    
    with open("models_data.json", "r") as f:
        data = json.load(f)

    relevant_noises = ['uni', 'eli', 'uni001', 'uni005', 'uni01', 'eli001', 'eli005', 'eli01', 'nat']
    if noise_type is not None and noise_type not in relevant_noises:
        raise ValueError(f"Invalid noise type. Choose from: {relevant_noises}")
    
    noise_name = ""
    noise_amplitude = ""
    if noise_type is not None:
        if noise_type.startswith("uni"):
            noise_name = "Synthesized Uniform-Kernel Motion Blur Noise"
            if noise_type in ['uni001', 'uni005', 'uni01']:
                noise_amplitude = noise_type[4:]
                noise_amplitude = f"0.{noise_amplitude}"
        elif noise_type.startswith("eli"):
            noise_name = "Synthesized Ellipse-Kernel Motion Blur Noise"
            if noise_type in ['eli001', 'eli005', 'eli01']:
                noise_amplitude = noise_type[4:]
                noise_amplitude = f"0.{noise_amplitude}"
        elif noise_type == "nat":
            noise_name = "Natural Recorded Camera Shake Kernels"
    
    
    # Filter out models that have "_aug" in their name
    if show_severity is False:
        filtered_models = {model: {"test_map50": model_data['test_map50'], "fps": model_data['fps'],
                                "uni001_map50": model_data['uni001_map50'], "uni005_map50": model_data['uni005_map50'],
                                "uni01_map50": model_data['uni01_map50'], "eli001_map50": model_data['eli001_map50'],
                                "eli005_map50": model_data['eli005_map50'], "eli01_map50": model_data['eli01_map50'],
                                "nat_map50": model_data['nat_map50']}
                            for model, model_data in data.items() if (("_aug" not in model or (show_augmentations)) and ("_severity" not in model))}
    else:
        filtered_models = {model: {"test_map50": model_data['test_map50'], "fps": model_data['fps'],
                                "uni001_map50": model_data['uni001_map50'], "uni005_map50": model_data['uni005_map50'],
                                "uni01_map50": model_data['uni01_map50'], "eli001_map50": model_data['eli001_map50'],
                                "eli005_map50": model_data['eli005_map50'], "eli01_map50": model_data['eli01_map50'],
                                "nat_map50": model_data['nat_map50']}
                            for model, model_data in data.items() if (("_aug" not in model or (show_augmentations)) and ("_severity" in model))}
        
    # Assign colors to models based on their base name (without "_aug")
    base_model_names = {}

    for model in filtered_models.keys():
        base_name = model.replace("_aug", "")
        if base_name not in base_model_names:
            base_model_names[base_name] = len(base_model_names)
    
    # Generate a list of unique colors based on the base model names
    colors = plt.cm.get_cmap('tab10', len(base_model_names))

    if noise_type is not None and noise_type in ['uni', 'eli']:
        noise_levels = ["001", "005", "01"]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        
        for i, level in enumerate(noise_levels):
            noise_key = f"{noise_type}{level}"
            for model, model_data in filtered_models.items():
                base_name = model.replace("_aug", "")
                color = colors(base_model_names[base_name])  # Assign color based on base name
                marker = 'v' if '_aug' in model else 'o'  # Use triangle for "_aug", dot for others
                
                axes[i].scatter(model_data['fps'], model_data[f"{noise_key}_map50"], label=model, color=color, 
                                marker=marker, s=100)
            
            # add a decimal point to the noise level after the first digit
            if level == "001":
                level = "0.01"
            elif level == "005":
                level = "0.05"
            elif level == "01":
                level = "0.1"
            axes[i].set_title(f"a={level}", fontsize=14)
            axes[i].set_xlabel("FPS", fontsize=12)
            axes[i].grid(True)
        
        axes[0].set_ylabel("mAP@50", fontsize=12)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Models", bbox_to_anchor=(1, 1), loc='upper left')
        fig.suptitle(f"Benchmark on the {'Severity ' if (show_severity) else ''}Test Set with {noise_name}", fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        # Plot the scatter of mAP@50 vs fps for each model
        plt.figure(figsize=(10, 6))
        for model, model_data in filtered_models.items():
            base_name = model.replace("_aug", "")
            color = colors(base_model_names[base_name])  # Assign color based on base name
            marker = 'v' if '_aug' in model else 'o'  # Use triangle for "_aug", dot for others
            data_type = "test_map50" if noise_type is None else f"{noise_type}_map50"
            plt.scatter(model_data['fps'], model_data[data_type], label=model, color=color, 
                        marker=marker, s=100)

        plt.title(f"Benchmark on the {'Severity ' if (show_severity) else ''}Test Set {'with ' + noise_name + ' ' if (noise_type is not None) else ''} {('(a=' + noise_amplitude + ')' if ((noise_type is not None) and (noise_type.startswith('uni') or noise_type.startswith('eli'))) else '')}", fontsize=13)
        plt.xlabel("FPS", fontsize=12)
        plt.ylabel("mAP@50", fontsize=12)
        plt.legend(title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def display_results_table(with_augmentations=False):
    # Ensure plot is centered and text (including titles) is left-aligned
    display(HTML("""
    <style>
    .output_wrapper, .output { display: flex; justify-content: center; }
    table.dataframe th, table.dataframe td { text-align: left; }
    </style>
    """))

    # Load JSON data
    with open("models_data.json", "r") as f:
        data = json.load(f)
        
    # Filter models based on the "_aug" condition
    filtered_models = {model: model_data for model, model_data in data.items() if "_aug" not in model or with_augmentations}
    
    # Remove unwanted keys
    for model_data in filtered_models.values():
        model_data.pop('train_losses', None)
        model_data.pop('val_maps', None)
        model_data.pop('best_val_map', None)
    
    if (not with_augmentations):
        for model_data in filtered_models.values():
            model_data.pop('uni001_map50', None)
            model_data.pop('uni005_map50', None)
            model_data.pop('uni01_map50', None)
            model_data.pop('eli001_map50', None)
            model_data.pop('eli005_map50', None)
            model_data.pop('eli01_map50', None)
            model_data.pop('nat_map50', None)
    
    # Create a DataFrame for display
    df = pd.DataFrame.from_dict(filtered_models, orient='index').reset_index()
    df.rename(columns={'index': 'Model Name'}, inplace=True)
    
    # Convert "map50" columns to percentage and round to 2 decimal points
    for column in df.columns:
        if "map50" in column.lower():
            df[column] = (df[column] * 100).round(2).astype(str) + '%'
        elif "model_parameters" in column.lower():
            df[column] = df[column].astype(int).apply(lambda x: f"{x:,}")
        else:
            df[column] = df[column].round(2)

    
    # Rename specific columns
    df.rename(columns={
        'train_time': 'Train Time (seconds)',
        'model_size': 'Model Size (MB)',
        'model_parameters': 'Model Parameters'
    }, inplace=True)
    
    if (not with_augmentations):
        df.rename(columns={
        'test_map50': 'test mAP@50'
        } , inplace=True)
    else:
        df.rename(columns={
        'test_map50': 'test clean mAP@50',
        'uni001_map50': 'test uniform 0.01 mAP@50',
        'uni005_map50': 'test uniform 0.05 mAP@50',
        'uni01_map50': 'test uniform 0.1 mAP@50',
        'eli001_map50': 'test ellipse 0.01 mAP@50',
        'eli005_map50': 'test ellipse 0.05 mAP@50',
        'eli01_map50': 'test ellipse 0.1 mAP@50',
        'nat_map50': 'test natural mAP@50'
        }, inplace=True)
    
    # Display the table
    display(df)
    
def plot_loss_and_map(model_name=None):
    import json
    import matplotlib.pyplot as plt
    from IPython.display import HTML, display

    # Ensure plot is centered
    display(HTML("<style>.output_wrapper, .output { display: flex; justify-content: center; }</style>"))
    
    with open("models_data.json", "r") as f:
        data = json.load(f)

    # Filter out models that have "_aug" or "_severity" in their name
    filtered_models = {model: {"loss": model_data['train_losses'], "map50": model_data['val_maps']}
                        for model, model_data in data.items() if "_aug" not in model and "_severity" not in model}

    # Generate distinct colors for each model (using a color palette from seaborn)
    import seaborn as sns
    colors = sns.color_palette("husl", len(filtered_models))

    # Create the figure with two subplots (left for training loss, right for val map@50)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the training losses on the left with logarithmic scale
    for idx, (model, metrics) in enumerate(filtered_models.items()):
        if model == model_name:
            ax[0].plot(metrics["loss"], label=f"{model} (highlighted)", color=colors[idx], linewidth=3, zorder=10)
        else:
            ax[0].plot(metrics["loss"], label=model, color=colors[idx], linewidth=1)

    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Loss")
    ax[0].set_yscale('log')  # Set y-axis to logarithmic scale
    ax[0].legend()

    # Plot the validation map@50 on the right
    for idx, (model, metrics) in enumerate(filtered_models.items()):
        if model == model_name:
            ax[1].plot(metrics["map50"], label=f"{model} (highlighted)", color=colors[idx], linewidth=3, zorder=10)
        else:
            ax[1].plot(metrics["map50"], label=model, color=colors[idx], linewidth=1)

    ax[1].set_title("Validation mAP@50")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("mAP@50")
    ax[1].legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

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
    

