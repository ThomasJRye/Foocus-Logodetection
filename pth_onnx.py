import torch
import argparse

# Define the device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Construct the argument parser.
parser = argparse.ArgumentParser()

parser.add_argument(
    '-i', '--input', default='', 
    help='path to input model'
)
parser.add_argument('-x', '--export', default='models/prod_model',
                    help='path to export model')

args = vars(parser.parse_args())

# Load the PyTorch model
model = torch.load(args['input'], map_location=device)

# Trace the model and convert it to TorchScript format
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Save the TorchScript model to a file
traced_model.save("my_model.pt")
