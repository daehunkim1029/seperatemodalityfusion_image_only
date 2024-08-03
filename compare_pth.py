import torch

def load_model_keys(model_path):
    # Load the model
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Extract the keys from the state_dict
    keys = set(state_dict.keys())
    return keys

def compare_model_keys(model_path1, model_path2):
    keys1 = load_model_keys(model_path1)
    keys2 = load_model_keys(model_path2)
    
    # Find the keys that are in model1 but not in model2
    only_in_model1 = keys1 - keys2
    # Find the keys that are in model2 but not in model1
    only_in_model2 = keys2 - keys1
    # Find the keys that are in both models
    in_both = keys1 & keys2
    
    print(f"Keys only in {model_path1}:")
    for key in only_in_model1:
        print(key)
        
    print(f"\nKeys only in {model_path2}:")
    for key in only_in_model2:
        print(key)
        
    print(f"\nKeys in both models:")
    for key in in_both:
        print(key)

# Example usage:
model_path1 = '/media/spalab/sdb/kypark/SeparateModalityFusion/work_dirs/dh/bevfusion_camera_only_centerhead_0724/epoch_5.pth'
model_path2 = 'pretrained/converted_mmdet3d_checkpoint.pth'

compare_model_keys(model_path1, model_path2)
