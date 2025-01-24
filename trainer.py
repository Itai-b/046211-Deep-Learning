import os
import gc
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models.detection as detection_models
from torchmetrics.detection import MeanAveragePrecision
import optuna
from tqdm import tqdm
import evaluator

# Regular train function
def train(model, train_loader, val_loader, optimizer, lr_scheduler, num_epochs=10, device="cuda", save_path=""):
    #send model to device
    model.to(device)
    train_losses = []
    val_maps = []
    train_time = 0.0
    best_model_state_dict = model.state_dict()
    best_val_map = 0.0
    best_epoch = 0

    ## clear memory
    torch.cuda.empty_cache()
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:

            # Prepare inputs for the model
            images = [img.to(device) for img in images]

            # Convert relevant target values to device
            targets_to_device = []
            for target in targets:
                targets_to_device.append({
                    'boxes': target['boxes'].to(device),
                    'labels': target['labels'].to(device)
                })
        
            # Compute loss
            loss_dict = model(images, targets_to_device)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            
            # Backpropagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        # Step the learning rate scheduler
        lr_scheduler.step()
        
        
        # Compute validation loss
        _ , val_map, fps = evaluator.evaluate_model(model, val_loader, batch_size=val_loader.batch_size, device=device)
        mean_ap = val_map["map_50"].item()
        
        if (mean_ap > best_val_map):
            best_val_map = mean_ap
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        train_losses.append(epoch_loss)
        val_maps.append(mean_ap)
        
        end_time = time.time()
        train_time += (end_time - start_time)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Validation mAP@50: {mean_ap:.4f}, FPS: {fps:.2f}, Epoch Time: {end_time - start_time:.2f}")
        
    # Save the best trained model:
    torch.save(best_model_state_dict, save_path)
    print(f"Best Epoch: {best_epoch}, Best Val mAP@50: {best_val_map:.4f}")
    print("Model training complete. ")
    
    
################################# optuna train function #################################