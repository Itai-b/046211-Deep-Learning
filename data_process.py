import os
import shutil
import xmltodict
import numpy as np
import json
import re
from enum import Enum
import cv2
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import kagglehub as kh
import utils

# Set random seed for reproducibility
seed = 211
np.random.seed(seed)
torch.manual_seed(seed)

# device - cpu or gpu?
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pin_memory = True if device == "cuda:0" else False


# Global Variables (normalize() function overwrites train_mean and train_std values)
train_mean = torch.tensor([0.5288, 0.5161, 0.4917])
train_std = torch.tensor([0.1826, 0.1745, 0.1723])

# Data directories
img_dir = 'data/chitholian_annotated_potholes_dataset/images'
ann_dir = 'data/chitholian_annotated_potholes_dataset/annotations'

def download_data():
    """
    Download data from Kaggle
    """
    kaggle_datapath = 'chitholian/annotated-potholes-dataset'
    data_path = 'data/chitholian_annotated_potholes_dataset'
    if not os.path.exists(data_path):
        #Load the data from kaggle
        data = kh.dataset_download(kaggle_datapath)
        # Move the data to the correct location
        shutil.move(data, data_path)
        # if the data is from chitholian, we need to split it to two folders "images" and "annotations"
        if 'chitholian' in kaggle_datapath:
            # Create the folders
            os.makedirs(os.path.join(data_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(data_path, 'annotations'), exist_ok=True)
            # Move the files
            annotated_images_dir = os.path.join(data_path, 'annotated-images') 
            for file in os.listdir(annotated_images_dir):
                if file.endswith('.xml'):
                    shutil.move(os.path.join(annotated_images_dir, file), os.path.join(data_path, 'annotations', file))
                else:
                    shutil.move(os.path.join(annotated_images_dir, file), os.path.join(data_path, 'images', file))
            # remove the empty folder
            os.rmdir(annotated_images_dir)
    else:
        print('Data already exists\n')

def data_preprocessing():
    """
    Download the data, split it into train, validation, and test sets, and resize the images.
    Then add yolo appropriate annotations and add motion blur noise to the test set (natural, uniform and ellipse noise with different amplitudes).
    """
    download_data()
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), # uint8 values in [0, 255] -> float tensor with values [0, 1]
    ])
    # Initialize the dataset
    dataset = PotholeDetectionDataset(img_dir, ann_dir, transform=transform)
    
    # Split the dataset to train, validation, and test sets (70-10-20)

    # Maintain the original indices while splitting
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.125, random_state=seed)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    print(f"Train set size: {len(train_set)} - {len(train_set)/len(dataset)*100:.2f}%")
    print(f"Validation set size: {len(val_set)} - {len(val_set)/len(dataset)*100:.2f}%")
    print(f"Test set size: {len(test_set)} - {len(test_set)/len(dataset)*100:.2f}%\n")
    
    # Save train, val, and test indices to JSON:
    split_data = {
        "train": train_set.indices,
        "val": val_set.indices,
        "test": test_set.indices
    }
    
    with open('./data/chitholian_annotated_potholes_dataset/our_split.json', "w") as file:
        json.dump(split_data, file, indent=4)
    
    utils.voc_to_yolo(
        voc_path='data/chitholian_annotated_potholes_dataset/annotations',
        yolo_path='data/chitholian_annotated_potholes_dataset/yolo_labels'
    )
    utils.organize_split_from_json(
        json_path='./data/chitholian_annotated_potholes_dataset/our_split.json',
        base_dir='./data/chitholian_annotated_potholes_dataset/',
        output_dir='./data/potholes_dataset/'
    )
    
    utils.add_noise_to_test(data_dir='./data/potholes_dataset/')  

class PotholeSeverity(Enum):
    """
    Enum class for the severity of potholes.
    The severity levels ranges from 0 (no pothole) to 4 (major pothole):
        0 - No pothole (background, shouldn't be a detection target)
        temporary 1 - general pothole, no specific severity)
        1 - Minor pothole (road damage that is non-dangerous for padestrians)
        2 - Medium pothole (road damage that is dangerous for padestrians, but not for vehicles)
        3 - Major pothole (road damage that is dangerous for both vehicles and padestrians)
    """
    NO_POTHOLE = 0
    POTHOLE = 1 # TODO - this label is for temporary until we classify each saverity in the data.
    #MINOR_POTHOLE = 1
    #MEDIUM_POTHOLE = 2
    #MAJOR_POTHOLE = 3

def get_label_name(label):
    return PotholeSeverity(label).name

class PotholeDetectionDataset:
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        
        # Preprocess data
        self.img_files, self.ann_files = self._preprocess_dataset()

    
    @staticmethod
    def _extract_index(filename):
        # Use a regex to extract the numeric index from the file name
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else float('inf')

    def _preprocess_dataset(self):
        # Get images from folder
        img_files = sorted(os.listdir(self.img_dir), key=self._extract_index)
        ann_files = sorted(os.listdir(self.ann_dir), key=self._extract_index)

        valid_img_files = []
        valid_ann_files = []

        # Parse images and annotated boxes to return only the valid images and boxes
        for img_file, ann_file in zip(img_files, ann_files):
            img_path = os.path.join(self.img_dir, img_file)
            ann_path = os.path.join(self.ann_dir, ann_file)
            
            # Load and validate
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]
            boxes, _ = self.parse_voc_annotation(ann_path)
            
            if self._check_boxes_validity(boxes, img_width, img_height):
                valid_img_files.append(img_file)
                valid_ann_files.append(ann_file)
                assert len(valid_img_files) == len(valid_ann_files)
        
        print(f'Number of valid images: {len(valid_img_files)}')
        return valid_img_files, valid_ann_files
    
    @staticmethod
    def _check_boxes_validity(boxes, img_width, img_height):
        for xmin, ymin, xmax, ymax in boxes:
            width = xmax - xmin
            height = ymax - ymin
            
            if width <= 0 or height <= 0:
                return False
            if xmin < 0 or ymin < 0 or xmax > img_width or ymax > img_height:
                return False
        return True

    def parse_voc_annotation(self, ann_path):
        with open(ann_path) as f:
            ann_data = xmltodict.parse(f.read())
        
        boxes = []
        labels = []
        objects = ann_data["annotation"].get("object", [])
        if not isinstance(objects, list):
            objects = [objects]
        
        for obj in objects:
            bbox = obj["bndbox"]
            xmin = int(float(bbox["xmin"]))
            ymin = int(float(bbox["ymin"]))
            xmax = int(float(bbox["xmax"]))
            ymax = int(float(bbox["ymax"]))
            boxes.append((xmin, ymin, xmax, ymax))
            labels.append(PotholeSeverity.POTHOLE.value)
        
        return boxes, labels
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        ann_path = os.path.join(self.ann_dir, self.ann_files[idx])
        
        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape[:2]  # (height, width)
        
        # Load annotations
        boxes, labels = self.parse_voc_annotation(ann_path)
        
        # Apply transform
        if self.transform is not None:
            img_pil = torchvision.transforms.ToPILImage()(img)
            img = self.transform(img_pil)
            new_size = (img.shape[2], img.shape[1])  # (width, height)
            
            # Adjust bounding boxes
            orig_h, orig_w = original_size
            new_w, new_h = new_size
            x_scale = new_w / orig_w
            y_scale = new_h / orig_h
            boxes = [
                (int(xmin * x_scale), int(ymin * y_scale), int(xmax * x_scale), int(ymax * y_scale))
                for xmin, ymin, xmax, ymax in boxes
            ]
            
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor([(xmax - xmin) * (ymax - ymin) for xmin, ymin, xmax, ymax in boxes], dtype=torch.float32),
        }
        return img, target

class NoisySubset(PotholeDetectionDataset):
    def __init__(self, original_subset, noise_fn, noise_params=None):
        self.original_subset = original_subset
        self.noise_fn = noise_fn
        self.noise_params = noise_params or {}
        self.noisy_images = []
        
        # Precompute noisy images
        for idx in range(len(original_subset)):
            img, target = original_subset[idx]

            # Apply noise
            noisy_img = self.noise_fn(image=img, **self.noise_params)

            self.noisy_images.append(noisy_img)

    def __len__(self):
        return len(self.original_subset)

    def __getitem__(self, idx):
        return self.noisy_images[idx], self.original_subset[idx][1]  # Noisy image and target

def collate_fn(batch):
    images = []
    targets = []
    for img, target in batch:
        images.append(img)
        targets.append(target)
    # Stack images into a single tensor
    images = torch.stack(images)
    return images, targets

def normalize(train_set):
    """
    check the mean and std of the training set (use before normalizing the images)
    """
    global train_mean, train_std
    
    # normalize the images according to the mean and std of the training set
    # DataLoader for train set
    train_loader = DataLoader(train_set, batch_size=32, shuffle=False, num_workers=2, collate_fn=collate_fn)

    # Initialize accumulators for mean and std
    n_pixels = 0
    mean_sum = torch.zeros(3)
    squared_sum = torch.zeros(3)

    for imgs, _ in train_loader:
        for img in imgs:
            img = img.view(3, -1)  # Flatten each channel
            n_pixels += img.size(1)  # Add the number of pixels per channel
            mean_sum += img.sum(dim=1)
            squared_sum += (img ** 2).sum(dim=1)

    # Compute mean and std
    train_mean = mean_sum / n_pixels
    train_std = torch.sqrt(squared_sum / n_pixels - train_mean ** 2)

    print(f"Training Set Mean: {train_mean}")
    print(f"Training Set Std: {train_std}\n")

def load_data(transform, input_size=300):
    """
    Using the PotholeDetectionDataset class, and relevant transformation.
    Split the data into train, validation, and test sets.
    """
    # Initialize the dataset
    dataset = PotholeDetectionDataset(img_dir, ann_dir, transform=transform)

    # Split the dataset to train, validation, and test sets (70-10-20)

    # Maintain the original indices while splitting
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.125, random_state=seed)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    print(f"Train set size: {len(train_set)} - {len(train_set)/len(dataset)*100:.2f}%")
    print(f"Validation set size: {len(val_set)} - {len(val_set)/len(dataset)*100:.2f}%")
    print(f"Test set size: {len(test_set)} - {len(test_set)/len(dataset)*100:.2f}%\n")
    
    return train_set, val_set, test_set

def convert_to_imshow_format(image, mean=train_mean, std=train_std):
    """
    Converts a normalized image tensor to the format expected by plt.imshow.
    Args:
        image (torch.Tensor): Normalized image tensor of shape [C, H, W].
        mean (torch.Tensor): Mean used for normalization (1D tensor of length 3 for RGB).
        std (torch.Tensor): Std used for normalization (1D tensor of length 3 for RGB).
    Returns:
        np.ndarray: Image array in HWC format, scaled to [0, 255] and uint8 type.
    """
    # Denormalize the image
    image = image * std[:, None, None] + mean[:, None, None]  # Reshape mean and std for broadcasting
    image = image.clamp(0, 1)  # Ensure values are in [0, 1] range
    
    # Convert to numpy and scale to [0, 255]
    image = (image.numpy() * 255).astype(np.uint8)
    
    # Convert from CHW to HWC
    return image.transpose(1, 2, 0)

def show_images(images, targets, title=""):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(images), figsize=(len(images)*3, 5))

    for idx, (img, target) in enumerate(zip(images, targets)):
        if isinstance(img, torch.Tensor):
            img_np = convert_to_imshow_format(img, train_mean, train_std)
            img_np = np.ascontiguousarray(img_np)  # Ensure compatibility with OpenCV
        else:
            img_np = np.ascontiguousarray(img)
        
        # Draw bounding boxes and labels
        for box, label in zip(target["boxes"], target["labels"]):
            cv2.rectangle(img_np,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (255, 0, 0), 2)  
            # label_name = get_label_name(label.item())
            # cv2.putText(img_np,
            #             label_name,
            #             (int(box[0]), int(box[1] - 5)),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.5,
            #             (255, 0, 0), 2)  # Red text
        
        # Display the image
        axes[idx].imshow(img_np)
        axes[idx].axis("off")
        axes[idx].set_title(f"Image #{target['image_id'].item()}")

    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.85)

    plt.tight_layout()
    plt.show()
    
def visualize_predictions(images, targets, all_predictions, threshold=0.5, show_severity=False, title=""):
    """
    Display images with ground truth and predicted bounding boxes.
    """
    fig, axs = plt.subplots(1, len(images), figsize=(15, 5))

    for i, (img, target) in enumerate(zip(images, targets)):
        # Convert image to imshow format
        img_np = convert_to_imshow_format(img, train_mean, train_std)
        img_np = np.ascontiguousarray(img_np)  # Ensure compatibility with OpenCV
        
        # Draw ground truth boxes (green)
        for box in target["boxes"]:
            cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(img_np, "GT", (int(box[0]), int(box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Access predictions for the specific image_id
        predictions = all_predictions.get(target['image_id'].item(), {})
        if predictions:
            for box, label, score in zip(predictions["boxes"], predictions["labels"], predictions["scores"]):
                if score < threshold:
                    continue
                xmin, ymin, xmax, ymax = map(int, box)
                cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                if show_severity:
                    cv2.putText(img_np, 
                                f"P: {get_label_name(label)}: {score:.2f}", 
                                (xmin, ymin - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 0, 0), 2)  # Red text, thickness=2
                else:
                    cv2.putText(img_np, 
                                f"P: {score:.2f}", 
                                (xmin, ymax + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 0, 0), 2)
        
        # Display the image with annotations
        axs[i].imshow(img_np)
        axs[i].axis("off")
        axs[i].set_title(f"Image #{target['image_id'].item()}")

    # Add a main title
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.9)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('data/potholes_dataset'):
        data_preprocessing()  
    else:
        print('Data already was proccesed\n')
    