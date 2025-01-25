import os
import gc
import time
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.models as models
from torchmetrics.detection import MeanAveragePrecision
import matplotlib.pyplot as plt
import numpy as np
import optuna
from tqdm import tqdm
import functools
import utils
import evaluator
import data_process

# Configuration

possible_optimizers = ['SGD', 'Adam', 'AdamW', 'RMSprop']
possible_schedulers = ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau', 'OneCycleLR']

model_name_global = None
op_train_set = None
op_val_set = None
save_path = None

def train(model, train_loader, val_loader, optimizer, lr_scheduler, num_epochs=10, device="cuda", model_name="", save_path=None, trial=None):
    global model_name_global
    
    model.to(device)
    
    train_losses = []
    val_maps = []
    fps_list = []
    train_time = 0.0
    
    best_model_state_dict = model.state_dict()
    best_val_map = 0.0
    best_epoch = 0

    stagnant_epochs = 3  # Number of epochs with insufficient improvement before pruning
    num_fouls = 0  # Number of times the model has failed to improve sufficiently
    
    # Clear memory
    torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets_to_device = [
                {'boxes': target['boxes'].to(device), 'labels': target['labels'].to(device)}
                for target in targets
            ]
            
            loss_dict = model(images, targets_to_device)
            losses = sum(loss for loss in loss_dict.values())
            epoch_loss += losses.item()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        
        _, val_map, fps = evaluator.evaluate_model(model, val_loader, batch_size=val_loader.batch_size, device=device)
        mean_ap = val_map["map_50"].item()

        # Track the best model
        if mean_ap > best_val_map:
            best_val_map = mean_ap
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        
        # Step the learning rate scheduler
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(mean_ap)
            else:
                lr_scheduler.step()

        # Optuna reporting and pruning logic
        if trial is not None:
            trial.report(mean_ap, epoch)
            
            # Check when mean_ap is zero over the last few epochs
            if epoch > 0 and mean_ap == 0.0:
                num_fouls += 1
            else:
                num_fouls = 0
            
            if num_fouls > stagnant_epochs:
                print(f"Trial pruned by Optuna at epoch {epoch}.")
                raise optuna.TrialPruned()

            # Optuna pruning based on the reported value
            if trial.should_prune():
                print(f"Trial pruned by Optuna at epoch {epoch}.")
                raise optuna.TrialPruned()
        
        # Append metrics for tracking
        train_losses.append(epoch_loss)
        val_maps.append(mean_ap)
        fps_list.append(fps)
        
        end_time = time.time()
        train_time += (end_time - start_time)
        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val mAP@50: {mean_ap:.4f}, FPS: {fps:.2f}, Epoch Time: {(end_time - start_time):.2f} seconds")
    
    if trial is not None:
        # Save the best trained model:
        torch.save(best_model_state_dict, os.path.join(save_path, f"{model_name_global}_{trial.number}_best.pth"))
    else:
        torch.save(best_model_state_dict, os.path.join(save_path, f"{model_name}_best.pth"))
    
    print(f"Best Epoch: {best_epoch}, Best Val mAP@50: {best_val_map:.4f}, Training Time: {train_time:.2f} seconds")
    print("Model training complete.")
    
    model_parameters, model_size = utils.calc_model_size(model)
    
    history = {
        "train_losses": train_losses,
        "val_maps": val_maps,
        "best_val_map": best_val_map,
        "fps": np.average(fps_list),
        "train_time": train_time,
        "model_parameters": model_parameters,
        "model_size": model_size
    }
    
    return history
    
################################# optuna functions #################################
def get_model(model_name="", trial=None, preweight_mode='fine_tuning'):
    global model_name_global
    
    if trial is not None:
        model_name = model_name_global
        preweight_mode = trial.suggest_categorical('preweight_mode', ['random', 'freezing', 'fine_tuning'])
    
    # TODO: Add more models
    
    if str.startswith(model_name, "fasterrcnn"):
        if model_name == "fasterrcnn_resnet50_fpn":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_resnet50_fpn()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
        elif model_name == "fasterrcnn_resnet50_fpn_v2":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_resnet50_fpn_v2()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1)            
        elif model_name == "fasterrcnn_mobilenet_v3_large_fpn":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_mobilenet_v3_large_fpn()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1)            
        elif model_name == "fasterrcnn_mobilenet_v3_large_320_fpn":
            if preweight_mode == 'random':
                model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn()  # No pre-trained weights
            else:
                model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1)        
        else:
            return None
        
        # Replace the classifier with a single-class output
        num_classes = len(data_process.PotholeSeverity)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes) 
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.roi_heads.box_predictor.parameters():
                param.requires_grad = True
        
        return model
          
    if str.startswith(model_name, "retinanet"):
        if model_name == "retinanet_resnet50_fpn":
            if preweight_mode == 'random':
                model = models.detection.retinanet_resnet50_fpn()
            else:
                model = models.detection.retinanet_resnet50_fpn(weights=models.detection.RetinaNet_ResNet50_FPN_Weights.COCO_V1)
        elif model_name == "retinanet_resnet50_fpn_v2":
            if preweight_mode == 'random':
                model = models.detection.retinanet_resnet50_fpn_v2()
            else:
                model = models.detection.retinanet_resnet50_fpn_v2(weights=models.detection.RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1)
        else:
            return None
        
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = models.detection.retinanet.RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=len(data_process.PotholeSeverity),
            norm_layer=functools.partial(torch.nn.GroupNorm, 32)
        )
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
        
        return model
    
    if model_name == "fcos_resnet50_fpn":
        if preweight_mode == 'random':
            model = models.detection.fcos_resnet50_fpn()
        else:
            model = models.detection.fcos_resnet50_fpn(weights=models.detection.FCOS_ResNet50_FPN_Weights.COCO_V1)
            
        # Replace the classifier with a Pothole-class output
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = models.detection.fcos.FCOSClassificationHead(
        in_channels=256,
        num_anchors=num_anchors,
        num_classes=len(data_process.PotholeSeverity),
        norm_layer=functools.partial(torch.nn.GroupNorm, 32)
        )
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the head
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
        
        return model
          
    if model_name == "ssd300_vgg16":
        if preweight_mode == 'random':
            model = models.detection.ssd300_vgg16()
        else:
            model = models.detection.ssd300_vgg16(weights=models.detection.SSD300_VGG16_Weights.COCO_V1)

        # Retrieve the list of input channels. 
        num_classes = len(data_process.PotholeSeverity)
        in_channels = models.detection._utils.retrieve_out_channels(model.backbone, (300, 300))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Image size for transforms.
        model.transform.min_size = (300,)
        model.transform.max_size = 300
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the classification and regression heads
            for param in model.parameters():
                param.requires_grad = False  # Freeze all parameters initially
            # Unfreeze the classification head
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
            # Unfreeze the box regression head
            for param in model.head.regression_head.parameters():
                param.requires_grad = True
        
        return model
    
    if model_name == "ssdlite320_mobilenet_v3_large":
        if preweight_mode == 'random':
            model = models.detection.ssdlite320_mobilenet_v3_large()
        else:
            model = models.detection.ssdlite320_mobilenet_v3_large(weights=models.detection.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1)

        # Retrieve the list of input channels. 
        num_classes = len(data_process.PotholeSeverity)
        in_channels = models.detection._utils.retrieve_out_channels(model.backbone, (320, 320))
        # List containing number of anchors based on aspect ratios.
        num_anchors = model.anchor_generator.num_anchors_per_location()
        # The classification head.
        model.head.classification_head = models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
        )

        # Image size for transforms.
        model.transform.min_size = (320,)
        model.transform.max_size = 320
        
        if preweight_mode == 'freezing':
            # Freeze all layers except the classification and regression heads
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the classification head
            for param in model.head.classification_head.parameters():
                param.requires_grad = True
            # Unfreeze the box regression head
            for param in model.head.regression_head.parameters():
                param.requires_grad = True
                
        return model
    
    return None

def get_optimizer(trial, model_parameters):
    # Suggest optimizer type
    optimizer_name = trial.suggest_categorical('optimizer', possible_optimizers)

    if optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.9, 0.99)
        lr = trial.suggest_float('lr', 5e-3, 5e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        return torch.optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_name == 'Adam':
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        beta1 = trial.suggest_float('beta1', 0.8, 0.999)
        beta2 = trial.suggest_float('beta2', 0.9, 0.999)
        return torch.optim.Adam(model_parameters, lr=lr, betas=(beta1, beta2))
    
    elif optimizer_name == 'AdamW':
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-3, 1e-2, log=True)
        beta1 = trial.suggest_float('beta1', 0.8, 0.999)
        beta2 = trial.suggest_float('beta2', 0.9, 0.999)
        return torch.optim.AdamW(model_parameters, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)
    
    else:  # RMSprop
        lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-1, 1, log=True)
        momentum = trial.suggest_float('momentum', 0.9, 0.99)
        return torch.optim.RMSprop(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)

def get_scheduler(trial, optimizer, num_epochs, steps_per_epoch):
    # Suggest scheduler type
    scheduler_name = trial.suggest_categorical('scheduler', possible_schedulers)
    
    if scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 2, 5)
        gamma = trial.suggest_float('gamma', 0.05, 0.5)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_name == 'CosineAnnealingLR':
        T_max = trial.suggest_int('T_max', 5, 15)
        eta_min = trial.suggest_float('eta_min', 1e-7, 1e-5, log=True)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_name == 'ReduceLROnPlateau':
        factor = trial.suggest_float('factor', 0.1, 0.5)
        patience = trial.suggest_int('patience', 2, 5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)
    
    else:  # OneCycleLR
        max_lr = trial.suggest_float('max_lr', 1e-4, 1e-2, log=True)
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
    
def objective(trial): 
    global op_train_set, op_val_set, save_path_global  # Declare global variables
    ## clear memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Define hyperparameter search space
    batch_size = trial.suggest_int('batch_size', 4, 8)
    num_epochs = trial.suggest_int('epochs', 10, 20)
    
    train_loader = DataLoader(
        op_train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_process.collate_fn
    )
    
    val_loader = DataLoader(
        op_val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_process.collate_fn
    )

    model = get_model(trial=trial)
    if model is None:
        raise ValueError("Invalid model")
    
    print(f"Checking Model: {model.__class__.__name__}")
    
    # Get optimizer and scheduler
    optimizer = get_optimizer(trial, model.parameters())
    scheduler = get_scheduler(trial, optimizer, num_epochs, len(train_loader))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    # Train the model
    print(f"Starting Trial #{trial.number}")
    history = train(model, train_loader, val_loader, optimizer, scheduler, num_epochs, device=device, save_path=save_path_global, trial=trial)
        
    return history['best_val_map']
    
def run_optimization(model_name, train_set, val_set, study_name="optuna_check", save_path=None, n_trials=50):
    global model_name_global, op_train_set, op_val_set, save_path_global  # Declare global variables
    
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage=f"sqlite:///data/models/optuna.db",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=data_process.seed)
    )
    
    model_name_global = model_name
    op_train_set = train_set
    op_val_set = val_set
    save_path_global = save_path
    
    study.optimize(objective, n_trials=n_trials)
    
    # Print optimization results
    trial = study.best_trial
    print(f"\nBest trial: #{trial.number}")
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Plot optimization history
    param_importances = optuna.importance.get_param_importances(study)
    
    # Save study results
    study.trials_dataframe().to_csv(os.path.join(save_path,"optimization_results.csv"))
    
    return study