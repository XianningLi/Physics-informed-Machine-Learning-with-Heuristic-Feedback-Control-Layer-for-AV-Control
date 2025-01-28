import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import time
import pickle
from scipy.io import savemat
import matplotlib.pyplot as plt
import random
import numpy as np

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

# Define the neural network with constraint layer
class AMPCNetwork(nn.Module):
    def __init__(self, num_state, num_reference, num_control, control_ranges):
        super(AMPCNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_state + num_reference, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, num_control)
        )
        self.control_ranges = control_ranges  # List of tuples [(min1, max1), (min2, max2), ...]

    def forward(self, x):
        outputs = self.fc(x)
        outputs = torch.tanh(outputs)
        scaled_outputs = []
        for i, (control_min, control_max) in enumerate(self.control_ranges):
            scaled_output = outputs[:, i:i+1] * (control_max - control_min) / 2 + (control_max + control_min) / 2
            scaled_outputs.append(scaled_output)
        return torch.cat(scaled_outputs, dim=1)

# Shuffle and batch training data
def shuffle_and_split_data(dataset, batch_size):
    indices = torch.randperm(len(dataset[0]))
    shuffled_data = (dataset[0][indices], dataset[1][indices])
    batches = [
        (shuffled_data[0][i:i+batch_size], shuffled_data[1][i:i+batch_size])
        for i in range(0, len(dataset[0]), batch_size)
    ]
    return batches

# Save results
def save_results(model, train_losses, test_losses, main_time_start):
    torch.save(model, 'AMPC_complete_integrated_net.pth')
    torch.save(model.state_dict(), 'AMPC_controller_net.pth')
    with open('AMPC_losses.pkl', 'wb') as file:
        pickle.dump({'AMPC_train_losses': train_losses, 'AMPC_test_losses': test_losses}, file)
    total_time = time.time() - main_time_start
    with open('AMPC_training_summary.txt', 'w') as summary_file:
        summary_file.write(f'Final Training Loss: {train_losses[-1]:.4f}\n')
        summary_file.write(f'Final Test Loss: {test_losses[-1]:.4f}\n')
        summary_file.write(f'Total Training Time: {total_time:.2f} seconds\n')
    state_dict = model.state_dict()
    weights = {key.replace('.', '_'): param.cpu().numpy() for key, param in state_dict.items()}
    savemat("AMPC_controller_net_weights.mat", weights, do_compression=True)
    plt.figure(figsize=(10, 5))
    plt.plot([loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in train_losses], label='Training Loss')
    plt.plot([loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in test_losses], label='Test Loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()


# Main function
if __name__ == "__main__":
    set_seed(42)

    # Hyperparameters
    num_state = 6
    num_reference = 2
    num_control = 2
    batch_size = 10000
    epochs = 1000
    learning_rate = 1e-4
    l2_lambda = 0  # L2 regularization parameter
    control_ranges = [(-3, 3), (-0.3, 0.3)]
    main_time_start = time.time()

    # Load data
    train_data = pd.read_csv('Train Data with Control.csv').values
    test_data = pd.read_csv('Test Data with Control.csv').values
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_dataset = (train_tensor[:, :-2], train_tensor[:, -2:])
    test_dataset = (test_tensor[:, :-2], test_tensor[:, -2:])

    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AMPCNetwork(num_state, num_reference, num_control, control_ranges).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        train_batches = shuffle_and_split_data(train_dataset, batch_size)
        epoch_start_time = time.time()

        for inputs, targets in train_batches:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            l2_reg = sum(param.pow(2).sum() for param in model.parameters() if param.requires_grad)
            loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_batches)
        train_losses.append(avg_train_loss)

        # Evaluate on test dataset
        model.eval()
        with torch.no_grad():
            test_inputs, test_targets = test_dataset[0].to(device), test_dataset[1].to(device)
            test_outputs = model(test_inputs)
            test_loss = criterion(test_outputs, test_targets).item()
            l2_reg = sum(param.pow(2).sum() for param in model.parameters() if param.requires_grad)
            test_loss += l2_lambda * l2_reg
            test_losses.append(test_loss)

        epoch_end_time = time.time()
        print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f} seconds")

    save_results(model, train_losses, test_losses, main_time_start)
