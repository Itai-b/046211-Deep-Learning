import os
import json

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
