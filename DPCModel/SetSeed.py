import random
import torch
import numpy as np


# Set random seed
def set_seed(seed_value=42):
    random.seed(seed_value)       # Python random module
    np.random.seed(seed_value)    # Numpy module
    torch.manual_seed(seed_value) # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU usage
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
