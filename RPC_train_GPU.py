import torch
from RPCModel.CustomLoss import CustomLoss
from RPCModel.IntegratedNetwork import IntegratedNetwork
from RPCModel.SetSeed import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from scipy.io import savemat


# Shuffle and batch training data
def shuffle_and_split_data(dataset, batch_size):
    indices = torch.randperm(len(dataset[0]))  # Randomly shuffle indices
    shuffled_data = (dataset[0][indices], dataset[1][indices])  # Shuffle data using indices
    batches = [
        (shuffled_data[0][i:i+batch_size], shuffled_data[1][i:i+batch_size])
        for i in range(0, len(dataset[0]), batch_size)
    ]
    return batches

# Training loop with truncated BPTT
def train(epoch, truncation_length=10):
    model.train()
    total_loss = 0
    time_start = time.time()

    # Shuffle and batch data
    train_batches = shuffle_and_split_data(train_dataset, batch_size)

    for batch_idx, (initial_state, reference) in enumerate(train_batches):
        initial_state, reference = initial_state.to(device), reference.to(device)
        optimizer.zero_grad()

        # Initialize states for truncated BPTT
        current_state = initial_state
        accumulated_loss = 0

        for t in range(0, Np, truncation_length):
            truncated_horizon = min(truncation_length, Np - t)

            # Forward pass for the truncated horizon
            states, controls = [], []
            for step in range(truncated_horizon):
                features = torch.cat([current_state, reference], dim=-1)
                _, control, _ = model.controller(features)
                controls.append(control)

                next_state = model.dynamics(current_state, control)
                states.append(next_state)
                current_state = next_state

            # Stack the truncated results
            states = torch.stack(states, dim=1)
            controls = torch.stack(controls, dim=1)

            # Compute loss for the truncated horizon
            loss = loss_function(states, controls, reference)
            accumulated_loss += loss.item()

            # Backpropagation for the truncated horizon
            loss.backward(retain_graph=True)

        # Update model parameters
        optimizer.step()
        total_loss += accumulated_loss

    avg_loss = total_loss / len(train_batches)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch}: Training loss: {avg_loss:.4f}')
    print(f'Training time: {time.time() - time_start:.2f} seconds')

# Testing loop
def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        initial_state, reference = test_dataset[0].to(device), test_dataset[1].to(device)

        # Forward pass
        states, controls = model(initial_state, reference)

        # Compute loss
        loss = loss_function(states, controls, reference)
        total_loss += loss.item()

    avg_loss = total_loss / 1
    test_losses.append(avg_loss)
    print(f'Test loss: {avg_loss:.4f}')

if __name__ == '__main__':
    main_time_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed()

    # Set vehicle dynamics parameters
    params = {
        'Cf': torch.tensor(1250),
        'Cr': torch.tensor(755),
        'a': torch.tensor(1.015),
        'b': torch.tensor(1.895),
        'm': torch.tensor(1270),
        'Iz': torch.tensor(1536.7),
    }
    dt = torch.tensor(0.5)
    Np = 10

    # Controller parameters
    controller_params = {
        'num_state': 6,
        'num_reference': 2,
        'num_control': 2,
        'acc_min': -3.0,
        'acc_max': 3.0,
        'delta_f_min': -0.3,
        'delta_f_max': 0.3,
    }

    # Initialize network
    model = IntegratedNetwork(controller_params, params, dt, Np).to(device)

    # Load train and test data
    train_data = pd.read_csv('Train Data.csv').values
    test_data = pd.read_csv('Test Data.csv').values
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_dataset = (train_tensor[:, :-2], train_tensor[:, -2:])
    test_dataset = (test_tensor[:, :-2], test_tensor[:, -2:])
    batch_size = 10000

    # Loss function and optimizer
    Qx_values = [0, 60, 500, 50, 10, 100]
    Qu_values = [5, 500]
    Qt_values = [0, 60, 1000, 70, 20, 200]
    loss_function = CustomLoss(model, Qx_values, Qu_values, Qt_values, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    test_losses = []

    # Train and test
    for epoch in range(1, 1001):
        train(epoch)
        test()

    # Save model and results
    torch.save(model, 'RPC_complete_integrated_net.pth')
    torch.save(model.controller, 'RPC_controller_net.pth')

    # Save losses to file
    data_to_save = {
        'RPC_train_losses': train_losses,
        'RPC_test_losses': test_losses,
    }
    with open('RPC_losses.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

    # Save final training loss and total training time to a text file
    final_train_loss = train_losses[-1]
    total_time = time.time() - main_time_start
    with open('RPC_training_summary.txt', 'w') as summary_file:
        summary_file.write(f'Final Training Loss: {final_train_loss:.4f}\n')
        summary_file.write(f'Total Training Time: {total_time:.2f} seconds\n')

    print(f'Final Training Loss and Total Time saved to "RPC_training_summary.txt".')
    print(f'Whole Time: {total_time:.2f} seconds')

    # Save controller model weights to .mat file
    model = torch.load("RPC_controller_net.pth", map_location=torch.device('cpu'))  # Load controller model for CPU
    model.eval()

    # Print the model structure
    print("Model structure:")
    print(model)

    print("\nSaving model weights to MATLAB .mat file...")

    # Extract weights and convert variable names
    state_dict = model.state_dict()
    weights = {key.replace('.', '_'): param.cpu().numpy() for key, param in state_dict.items()}  # Replace '.' with '_'

    # Save weights to .mat file
    savemat("RPC_controller_net_weights.mat", weights, do_compression=True)

    print("\nWeights have been saved to 'RPC_controller_net_weights.mat'.")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()
