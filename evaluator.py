import random
import torch
from torch.utils.data import DataLoader
import torchvision
import time
import data_process
from torchmetrics.detection import MeanAveragePrecision

def preprocess_bbox(prediction, threshold=0.5):
    
    """Non-max suppression is the final step of these object detection algorithms and is 
       used to select the most appropriate bounding box for the object.
       The NMS takes two things into account
        -The objectiveness score is given by the model
        -The overlap or IOU of the bounding boxes"""
    
    processed_bbox={}
    
    boxes=prediction["boxes"][prediction["scores"]>=threshold]
    scores=prediction["scores"][prediction["scores"]>=threshold]
    labels=prediction["labels"][prediction["scores"]>=threshold]
    nms=torchvision.ops.nms(boxes,scores,iou_threshold=threshold)
            
    processed_bbox["boxes"]=boxes[nms]
    processed_bbox["scores"]=scores[nms]
    processed_bbox["labels"]=labels[nms]
    
    return processed_bbox

def evaluate_model(model, data_loader, threshold=0.5, num_iterations=1, batch_size=5, device="cuda"):
    metric=MeanAveragePrecision(box_format='xyxy',class_metrics=True)
    metric.to(device)
    
    all_predictions={}
    total_avg_time = 0
    
    model.eval()
    with torch.no_grad():
        for imgs,targets in data_loader:
            imgs=[img.to(device) for img in imgs]
            targets=[{k:v.to(device) for (k,v) in d.items()} for d in targets]
            
            start_time = time.time()
            for _ in range(num_iterations):
                predictions=model(imgs)
            end_time = time.time()
            
            total_avg_time += (end_time - start_time) / (num_iterations * batch_size)
            
            results=[]
            for prediction, target in zip(predictions, targets):
                image_id = target["image_id"].item()
                results.append(preprocess_bbox(prediction, threshold))
                all_predictions[image_id]=prediction
            metric.update(results,targets)
    
    avg_time_per_batch = total_avg_time / len(data_loader)
    fps = 1 / avg_time_per_batch
    
    results=metric.compute()
    return all_predictions, results, fps

def test_model(model, test_set, index=None, title=None, history=None, show_batch=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Move the model to the GPU
    model.eval()  # Set the model to evaluation mode

    test_loader = DataLoader(test_set, batch_size=5, shuffle=False, collate_fn=data_process.collate_fn)

    predictions, results, fps = evaluate_model(model, test_loader, num_iterations=1, threshold=0.5)

    mean_ap_50=results["map_50"].item()
    print(f"Mean Average Precision @ 0.5 : {mean_ap_50:.4f}",
        f"FPS: {fps:.2f}")
    
    # Select a random batch if index is not provided
    if index is None:
        index = random.randint(0,len(list(test_loader))-1)  # Select a random batch
    elif index >= len(list(test_loader)):
        raise ValueError("Index out of range")
    
    batches = list(test_loader)
    images, targets = batches[index]

    if title is not None:
        title = f"{title} Test Batch #{index+1}"
    else:
        title = f"Test Batch #{index+1}"
    
    # Visualize the predictions on the selected batch
    if show_batch:
        data_process.visualize_predictions(images, targets, predictions, threshold=0.5, show_severity=False, title=title)
    
    if history is not None:
        history['test_map'] = mean_ap_50
        
    return history