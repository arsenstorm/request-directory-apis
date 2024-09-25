import torch
import os
from faceland import FaceLanndInference

# Load the checkpoint file (model weights)
checkpoint = torch.load('./src/faceland.pth')

# Initialize the model architecture
model = FaceLanndInference()

# Load the model weights from the checkpoint
model.load_state_dict(checkpoint)

# Set the model to evaluation mode
model.eval()

# Create a dummy input that matches the input shape of the model
# For example, the input shape is (1, 3, 112, 112)
dummy_input = torch.rand(1, 3, 112, 112)

# Export the model to ONNX format
onnx_file_name = "facelandmarks_model.onnx"
torch.onnx.export(
    model,                     # The model to be exported
    dummy_input,               # The dummy input tensor
    onnx_file_name,            # The output ONNX file path
    export_params=True,        # Store trained parameter weights inside the ONNX model file
    opset_version=11,          # Specify ONNX version (you can change this if needed)
    do_constant_folding=True,  # Perform constant folding for optimization
    input_names=['input'],     # Optional: Specify input names
    output_names=['output'],   # Optional: Specify output names
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}  # Allow dynamic batching
)

print(f"ONNX model exported as {onnx_file_name}")
