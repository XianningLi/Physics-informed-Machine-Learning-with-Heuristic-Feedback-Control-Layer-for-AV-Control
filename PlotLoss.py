import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the files
dpc_path = 'DPC_losses.pkl'
rpc_path = 'RPC_losses.pkl'
hfrpc_path = 'HFRPC_losses.pkl'

dpc_losses = pd.read_pickle(dpc_path)
rpc_losses = pd.read_pickle(rpc_path)
hfrpc_losses = pd.read_pickle(hfrpc_path)

# Extract training losses for each method
dpc_train_losses = dpc_losses['DPC_train_losses']
rpc_train_losses = rpc_losses['RPC_train_losses']
hfrpc_train_losses = hfrpc_losses['HFRPC_train_losses']

# Set the font to Times New Roman for all elements
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Times New Roman'

# Plot the training losses
plt.figure(figsize=(12, 3))  # Wider and shorter figure
plt.plot(dpc_train_losses, label='DPC', linewidth=2)
plt.plot(rpc_train_losses, label='RPC', linewidth=2)
plt.plot(hfrpc_train_losses, label='HFRPC', linewidth=2)

# Add labels, legend, and customize axes
plt.xlabel('Epochs', fontsize=20, labelpad=5)  # Reduce padding for x-axis label
plt.ylabel('Training Loss', fontsize=20, labelpad=5)  # Reduce padding for y-axis label
plt.xlim(0, 1000)
plt.ylim(2000, 6000)
plt.yticks(range(2000, 6001, 1000), fontsize=18)  # Larger font for y-axis ticks
plt.xticks(fontsize=18)  # Larger font for x-axis ticks
plt.legend(fontsize=18)  # Larger font for legend

# Adjust the plot to prevent labels from being cut off
plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95, top=0.9)

# Add grid and emphasize black borders
plt.grid(True, linewidth=1, color='black', alpha=0.5)

# Save the figure
plt.savefig('Training Loss over Epochs.png', dpi=600, bbox_inches='tight')

# Show the plot
plt.show()
