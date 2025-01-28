import torch
from DPCModel.CustomLoss import CustomLoss
from DPCModel.IntegratedNetwork import IntegratedNetwork
from DPCModel.SetSeed import set_seed
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from scipy.io import savemat


# Shuffle and batch training data
def shuffle_and_split_data(dataset, batch_size):
    indices = torch.randperm(len(dataset))  # Randomly shuffle indices
    shuffled_data = dataset[indices]  # Shuffle dataset using indices
    batches = [
        shuffled_data[i:i+batch_size]
        for i in range(0, len(dataset), batch_size)
    ]
    return batches

# Training loop
def train(epoch):
    model.train()
    total_loss = 0
    time_start = time.time()

    # Shuffle and batch training data
    train_batches = shuffle_and_split_data(train_dataset, batch_size)

    for batch_idx, features in enumerate(train_batches):
        features = features.to(device)

        optimizer.zero_grad()
        states, initial_state, reference, control_sequence = model(features)
        loss = loss_function(states, initial_state, reference, control_sequence)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_batches)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch}: Training loss: {avg_loss:.4f}')
    print(f'Training time: {time.time() - time_start:.2f} seconds')

# Testing loop
def test():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        features = test_dataset.to(device)
        states, initial_state, reference, control_sequence = model(features)
        loss = loss_function(states, initial_state, reference, control_sequence)
        total_loss += loss.item()

    avg_loss = total_loss
    test_losses.append(avg_loss)
    print(f'Test loss: {avg_loss:.4f}')


# Main script
if __name__ == '__main__':
    main_time_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    set_seed()

    # Vehicle dynamics and controller parameters
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

    controller_params = {
        'num_state': 6,
        'num_reference': 2,
        'num_control': 2,
        'Np': Np,
        'acc_min': -3.0,
        'acc_max': 3.0,
        'delta_f_min': -0.3,
        'delta_f_max': 0.3,
    }

    model = IntegratedNetwork(controller_params, params, dt, Np).to(device)

    # Load and process data
    train_data = pd.read_csv('Train Data.csv').values
    test_data = pd.read_csv('Test Data.csv').values
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)
    train_dataset = train_tensor
    test_dataset = test_tensor
    batch_size = 10000

    # Loss function and optimizer
    Qx_values = [0, 60, 500, 50, 10, 100]
    Qu_values = [5, 500]
    Qt_values = [0, 60, 1000, 70, 20, 200]
    loss_function = CustomLoss(model, Qx_values, Qu_values, Qt_values, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    train_losses = []
    test_losses = []

    # Training and testing
    for epoch in range(1, 1001):
        train(epoch)
        test()

    # Save model and results
    torch.save(model, 'DPC_complete_integrated_net.pth')
    torch.save(model.controller, 'DPC_controller_net.pth')

    # Save losses to file
    data_to_save = {
        'DPC_train_losses': train_losses,
        'DPC_test_losses': test_losses,
    }
    with open('DPC_losses.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

    # Save final training loss and total training time to a text file
    final_train_loss = train_losses[-1]
    total_time = time.time() - main_time_start
    with open('DPC_training_summary.txt', 'w') as summary_file:
        summary_file.write(f'Final Training Loss: {final_train_loss:.4f}\n')
        summary_file.write(f'Total Training Time: {total_time:.2f} seconds\n')

    print(f'Final Training Loss and Total Time saved to "DPC_training_summary.txt".')
    print(f'Whole Time: {total_time:.2f} seconds')

    # Save controller model weights to .mat file
    model = torch.load("DPC_controller_net.pth", map_location=torch.device('cpu'))  # Load controller model for CPU
    model.eval()

    # Print the model structure
    print("Model structure:")
    print(model)

    print("\nSaving model weights to MATLAB .mat file...")

    # Extract weights and convert variable names
    state_dict = model.state_dict()
    weights = {key.replace('.', '_'): param.cpu().numpy() for key, param in state_dict.items()}  # Replace '.' with '_'

    # Save weights to .mat file
    savemat("DPC_controller_net_weights.mat", weights, do_compression=True)

    print("\nWeights have been saved to 'DPC_controller_net_weights.mat'.")

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
