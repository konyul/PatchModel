import torch
import sys
if len(sys.argv) != 2:
    print("Usage: python script.py <path_to_model_file>")
    sys.exit(1)

file_path = sys.argv[1]
model = torch.load(file_path, map_location='cpu')
total_params = 0
for key in model['state_dict'].keys():
    total_params += model['state_dict'][key].nelement()

print(f"Total number of parameters in the model: {total_params/1e6}")
