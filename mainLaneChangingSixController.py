from simulation import simulation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys

# Redirect stdout to a file
log_file = open("simulation_output_log.txt", "w")
sys.stdout = log_file

lane_width = 4  # m, lane width

# Simulation parameters
initial_speeds = [70, 85, 95, 110]  # km/h
reference_speeds = [70, 85, 95, 110]  # km/h
lane_changes = [(2, 3), (2, 4), (2, 5), (2, 6), (6, 5), (6, 4), (6, 3), (6, 2)]  # Lane change combinations

# Set font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 10
rcParams['legend.fontsize'] = 10

# Define a consistent color map for controllers
controller_colors = {
    "MPC": "blue",
    "AMPC": "green",
    "HFAMPC": "orange",
    "DPC": "red",
    "RPC": "purple",
    "HFRPC": "brown"
}

controllers = {
    0: "MPC",
    1: "AMPC",
    2: "HFAMPC",
    3: "DPC",
    4: "RPC",
    5: "HFRPC"
}

# Function to compute metrics
def compute_metrics(results, vx_ref, Y_ref):
    time = results[:, 0]
    vx = results[:, 4]
    Y = results[:, 2]
    acc = results[:, 7]
    delta_f = results[:, 8]
    calc_time = results[:, -1]

    speed_rmse = np.sqrt(np.mean((vx - vx_ref)**2))
    lateral_rmse = np.sqrt(np.mean((Y - Y_ref)**2))
    avg_calc_time = np.mean(calc_time)
    acc_variance = np.var(acc)
    delta_f_variance = np.var(delta_f)

    return speed_rmse, lateral_rmse, avg_calc_time, acc_variance, delta_f_variance

# Store metrics
controller_metrics = {name: [] for name in controllers.values()}

# Adjust the figsize to make each subplot flatter
fig, axes = plt.subplots(len(initial_speeds), len(reference_speeds), figsize=(18, 8))  # Wider and shorter figure

# Create a list to collect lines and labels for the global legend
lines = []
labels = []

# Iterate through initial and reference speed combinations
for i, vx_0_kmh in enumerate(initial_speeds):
    for j, vx_ref_kmh in enumerate(reference_speeds):
        ax = axes[i, j]
        for initial_lane, target_lane in lane_changes:
            # Initial conditions
            t_0 = 0
            X_0 = 0
            Y_0 = lane_width * initial_lane / 2
            psi_0 = 0
            vx_0 = vx_0_kmh / 3.6
            vy_0 = 0
            omega_r_0 = 0
            acc_1 = 0
            delta_f_1 = 0
            X_CL_0 = 25
            Y_CL_0 = lane_width * initial_lane / 2
            v_CL_0 = vx_0_kmh / 3.6
            initial_conditions = np.array(
                [t_0, X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0, acc_1, delta_f_1, X_CL_0, Y_CL_0, v_CL_0])

            vx_ref = vx_ref_kmh / 3.6
            Y_ref = lane_width * target_lane / 2

            # Plot trajectories for each controller and compute metrics
            for flag, controller_name in controllers.items():
                results, calc_time, _ = simulation(initial_conditions, 15, 0.01, 0.05, flag, vx_ref, Y_ref)
                X = results[:, 1]
                Y = results[:, 2]

                # Add calculation time column
                calc_time_column = np.full(results.shape[0], calc_time)
                results = np.column_stack((results, calc_time_column))

                # Compute metrics
                metrics = compute_metrics(results, vx_ref, Y_ref)
                controller_metrics[controller_name].append(metrics)

                # Plot
                line, = ax.plot(X, Y, label=controller_name if (initial_lane == 2 and target_lane == 3) else "",
                                color=controller_colors[controller_name])

                # Collect one instance of each controller's line for the global legend
                if (initial_lane == 2 and target_lane == 3) and controller_name not in labels:
                    lines.append(line)
                    labels.append(controller_name)

        # Set subplot title and labels
        ax.set_title(f'$v_{{\mathrm{{0}}}}={vx_0_kmh}$km/h, $v_{{\mathrm{{ref}}}}={vx_ref_kmh}$km/h')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_yticks([4, 6, 8, 10, 12])
        ax.set_ylim(3, 13)
        ax.grid(True)

# Add global legend
fig.legend(lines, labels, loc='upper right', ncol=len(controllers), bbox_to_anchor=(1, 1.015), frameon=False)

# Print metrics
def print_metrics(metrics_dict):
    for controller_name, metrics in metrics_dict.items():
        avg_metrics = np.mean(metrics, axis=0)
        print(f"{controller_name} Performance Metrics:")
        print(f"Speed Tracking RMSE: {avg_metrics[0]:.2f} m/s")
        print(f"Lateral Position Tracking RMSE: {avg_metrics[1]:.2f} m")
        print(f"Average Calculation Time: {avg_metrics[2]:.4f} s")
        print(f"Acceleration Variance: {avg_metrics[3]:.4f} (m/s^2)^2")
        print(f"Steering Angle Variance: {avg_metrics[4]:.4f} rad^2")
        print()

print_metrics(controller_metrics)

# Restore stdout and close the log file
sys.stdout = sys.__stdout__
log_file.close()

# Adjust overall layout
plt.tight_layout()
plt.savefig("Closed-loop Numerical Simulation Results.png", dpi=600, bbox_inches='tight')
plt.show()
