from simulation import simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Set the font in matplotlib to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 16
rcParams['legend.fontsize'] = 16

lane_width = 4  # m, lane width

# Initial conditions
t_0 = 0
X_0 = 0
Y_0 = lane_width / 2 * 1
psi_0 = 0
vx_0 = 80 / 3.6
vy_0 = 0
omega_r_0 = 0
acc_1 = 0
delta_f_1 = 0
X_CL_0 = 25
Y_CL_0 = lane_width / 2
v_CL_0 = 70 / 3.6
initial_conditions = np.array([t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_CL_0, v_CL_0])

vx_ref = 100 / 3.6
Y_ref = lane_width * 3 / 2

# Controller definitions
controllers = {
    0: "MPC",
    1: "AMPC",
    2: "HFAMPC",
    3: "DPC",
    4: "RPC",
    5: "HFRPC"
}
controller_colors = {
    "MPC": "blue",
    "AMPC": "green",
    "HFAMPC": "orange",
    "DPC": "red",
    "RPC": "purple",
    "HFRPC": "brown"
}

# Store results for each controller
results = {}
time_records = {}
performance_metrics = []  # Store performance metrics for all controllers

for flag, name in controllers.items():
    sim_results, avg_time, time_record = simulation(initial_conditions, 15, 0.01, 0.05, flag, vx_ref, Y_ref)
    results[name] = sim_results
    time_records[name] = time_record

    # Calculate max and average calculation time
    max_time = max(time_record)
    avg_time = np.mean(time_record)
    performance_metrics.append((name, max_time, avg_time))

    # Print metrics
    print(f"Controller: {name}, Max Time: {max_time:.4f} s, Avg Time: {avg_time:.4f} s")

# Write performance metrics to a txt file
with open("results.txt", "w") as file:
    file.write("Controller Performance Metrics\n")
    file.write("================================\n")
    for name, max_time, avg_time in performance_metrics:
        file.write(f"Controller: {name}, Max Time: {max_time:.4f} s, Avg Time: {avg_time:.4f} s\n")

# Create charts
# fig, axs = plt.subplots(3, 3, figsize=(18, 9))
fig, axs = plt.subplots(3, 3, figsize=(14, 9))
lines = []
labels = []

# First subplot: X-Y Trajectories
ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
ax1.set_facecolor('gray')  # Set background color
ax1.axhline(y=lane_width, color='white', linestyle=(0, (10, 5)))  # Draw lane lines
for name, sim_results in results.items():
    X = sim_results[:, 1]
    Y = sim_results[:, 2]
    line, = ax1.plot(X, Y, label=name, color=controller_colors[name])
    if name not in labels:
        lines.append(line)
        labels.append(name)
# ax1.set_title("X and Y Trajectories")
ax1.set_xlabel("X Position (m)")
ax1.set_ylabel("Y Position (m)")
ax1.set_xlim([0, 400])  # Set X-axis range
ax1.set_ylim([0, 8])  # Set Y-axis range to 8
ax1.grid(False)
axs[0, 0].get_xaxis().set_visible(False)
axs[0, 0].get_yaxis().set_visible(False)
axs[0, 1].get_xaxis().set_visible(False)
axs[0, 1].get_yaxis().set_visible(False)

# Second subplot: Calculation Time
# axs[0, 2].set_title("Calculation Time")
axs[0, 2].set_xlabel("Simulation Steps")
axs[0, 2].set_ylabel("Time (s)")
for name, time_record in time_records.items():
    axs[0, 2].plot(range(len(time_record)), time_record, label=name, color=controller_colors[name])
axs[0, 2].grid(True)

# Other subplots
variables = ["Yaw Angle", "Longitudinal\nVelocity", "Lateral Velocity", "Yaw Rate", "Acceleration", "Steering Angle"]
indices = [3, 4, 5, 6, 7, 8]
units = ["rad", "km/h", "km/h", "rad/s", "m/s$^2$", "rad"]

for idx, (var, i, unit) in enumerate(zip(variables, indices, units)):
    row = idx // 3 + 1
    col = idx % 3
    ax = axs[row, col]
    for name, sim_results in results.items():
        data = sim_results[:, i]
        if i in [4, 5]:  # Convert velocity to km/h
            data = data * 3.6
        ax.plot(sim_results[:, 0], data, label=name, color=controller_colors[name])
    # ax.set_title(var)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(f"{var} ({unit})")
    ax.grid(True)

# Add global legend
fig.legend(lines, labels, loc='upper right', ncol=len(controllers), bbox_to_anchor=(1, 1.015), frameon=False)
plt.tight_layout()
plt.savefig("Lane-Change Instance Within Training Set Boundaries.png", dpi=600, bbox_inches='tight')
plt.show()
