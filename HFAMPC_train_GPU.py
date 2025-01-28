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
class HFAMPCNetwork(nn.Module):
    def __init__(self, num_state, num_reference, num_control, acc_min, acc_max, delta_f_min, delta_f_max):
        super(HFAMPCNetwork, self).__init__()
        self.num_state = num_state
        self.num_reference = num_reference
        self.num_control = num_control
        self.acc_min = acc_min
        self.acc_max = acc_max
        self.delta_f_min = delta_f_min
        self.delta_f_max = delta_f_max

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.num_state + self.num_reference, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=256),
            nn.GELU(),
            nn.Linear(in_features=256, out_features=self.num_control * self.num_state),
        )

    def forward(self, features):
        # Split input into initial state and reference values
        initial_state = features[:, :self.num_state]
        reference = features[:, self.num_state:self.num_state + self.num_reference]

        # Construct reference state to match the shape of the initial state
        ref_state = torch.zeros_like(initial_state)
        ref_state[:, 1] = reference[:, 0]  # Lateral position reference
        ref_state[:, 3] = reference[:, 1]  # Speed reference

        # Compute state error
        state_error = initial_state - ref_state

        # Compute output gain matrix
        outputs = self.fc(features)
        gain_matrix = outputs.view(-1, self.num_control, self.num_state)

        # Adjust gain matrix values for specific states
        gain_matrix[:, :, 0] = 0  # Zero gain for the first state
        gain_matrix[:, 1, 1] = -gain_matrix[:, 1, 1].clone() ** 2  # Negative square for the second state
        gain_matrix[:, 0, 3] = -gain_matrix[:, 0, 3].clone() ** 2 - 0.6  # Negative square for the fourth state
        gain_matrix[:, 0, 1] = 0  # Negative square for the second state
        gain_matrix[:, 1, 3] = 0  # Negative square for the fourth state

        # Compute control: u = K * (x - x_ref)
        control = torch.bmm(gain_matrix, state_error.unsqueeze(-1)).squeeze(-1)

        # Scale control values to the valid range
        control = torch.tanh(control)
        acc = control[..., 0:1] * (self.acc_max - self.acc_min) / 2 + (self.acc_max + self.acc_min) / 2
        delta_f = control[..., 1:2] * (self.delta_f_max - self.delta_f_min) / 2 + (
                self.delta_f_max + self.delta_f_min) / 2
        control = torch.cat([acc, delta_f], dim=-1)

        return initial_state, control, reference

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
    torch.save(model, 'HFAMPC_complete_integrated_net.pth')
    torch.save(model.state_dict(), 'HFAMPC_controller_net.pth')
    with open('HFAMPC_losses.pkl', 'wb') as file:
        pickle.dump({'HFAMPC_train_losses': train_losses, 'HFAMPC_test_losses': test_losses}, file)
    total_time = time.time() - main_time_start
    with open('HFAMPC_training_summary.txt', 'w') as summary_file:
        summary_file.write(f'Final Training Loss: {train_losses[-1]:.4f}\n')
        summary_file.write(f'Final Test Loss: {test_losses[-1]:.4f}\n')
        summary_file.write(f'Total Training Time: {total_time:.2f} seconds\n')
    state_dict = model.state_dict()
    weights = {key.replace('.', '_'): param.cpu().numpy() for key, param in state_dict.items()}
    savemat("HFAMPC_controller_net_weights.mat", weights, do_compression=True)
    plt.figure(figsize=(10, 5))
    plt.plot([loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in train_losses], label='Training Loss')
    plt.plot([loss.cpu().numpy() if torch.is_tensor(loss) else loss for loss in test_losses], label='Test Loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    # plt.savefig('HFAMPC_loss_plot.png')
    plt.show()


# Main function
if __name__ == "__main__":
    set_seed(42)

    # Hyperparameters
    num_state = 6
    num_reference = 2
    num_control = 2
    acc_min, acc_max = -3, 3
    delta_f_min, delta_f_max = -0.3, 0.3
    batch_size = 10000
    epochs = 1000
    learning_rate = 1e-4
    l2_lambda = 0  # L2 regularization parameter
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
    model = HFAMPCNetwork(num_state, num_reference, num_control, acc_min, acc_max, delta_f_min, delta_f_max).to(device)
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
            _, outputs, _ = model(inputs)
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
            _, test_outputs, _ = model(test_inputs)
            test_loss = criterion(test_outputs, test_targets).item()
            l2_reg = sum(param.pow(2).sum() for param in model.parameters() if param.requires_grad)
            test_loss += l2_lambda * l2_reg
            test_losses.append(test_loss)

        epoch_end_time = time.time()
        print(f"Epoch {epoch}/{epochs}, Training Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Time: {epoch_end_time - epoch_start_time:.2f} seconds")

    save_results(model, train_losses, test_losses, main_time_start)
