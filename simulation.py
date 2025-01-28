import numpy as np
from MPC_controller import mpc_controller
import time
import torch
from AMPC_train_GPU import AMPCNetwork
from HFAMPC_train_GPU import HFAMPCNetwork


# controller_flag = 0 (MPC)
# controller_flag = 1 (AMPC)
# controller_flag = 2 (HFAMPC)
# controller_flag = 3 (DPC)
# controller_flag = 4 (RPC)
# controller_flag = 5 (HFRPC)
# simulation_step: Ts
# controller_step: Tc
def simulation(initial_conditions, simulation_time, Ts, Tc, controller_flag, vx_ref, Y_ref):
    # Load the saved model state
    control_ranges = [(-3, 3), (-0.3, 0.3)]  # 根据实际控制范围定义
    AMPC_model_LC = AMPCNetwork(num_state=6, num_reference=2, num_control=2, control_ranges=control_ranges)
    # AMPC_model_LC = AMPCNetwork(num_state=6, num_reference=2, num_control=2)
    AMPC_model_LC.load_state_dict(torch.load('AMPC_controller_net.pth', map_location="cpu"))
    AMPC_model_LC.eval()
    AMPC_model_LC.to("cpu")

    HFAMPC_model_LC = HFAMPCNetwork(num_state=6, num_reference=2, num_control=2, acc_min=-3, acc_max=3, delta_f_min=-0.3, delta_f_max=0.3)
    HFAMPC_model_LC.load_state_dict(torch.load('HFAMPC_controller_net.pth', map_location="cpu"))
    HFAMPC_model_LC.eval()
    HFAMPC_model_LC.to("cpu")

    DPC_model_LC = torch.load('DPC_controller_net.pth')
    DPC_model_LC.eval()
    DPC_model_LC.to("cpu")

    RPC_model_LC = torch.load('RPC_controller_net.pth')
    RPC_model_LC.eval()
    RPC_model_LC.to("cpu")

    HFRPC_model_LC = torch.load('HFRPC_controller_net.pth')
    HFRPC_model_LC.eval()
    HFRPC_model_LC.to("cpu")

    # Initialize the vehicle state
    t_0 = initial_conditions[0]  # s, initial time
    X_0 = initial_conditions[1]  # m, initial X coordinate
    Y_0 = initial_conditions[2]  # m, initial Y coordinate
    psi_0 = initial_conditions[3]  # rad, initial yaw angle
    vx_0 = initial_conditions[4]  # m/s, initial longitudinal velocity
    vy_0 = initial_conditions[5]  # m/s, initial lateral velocity
    omega_r_0 = initial_conditions[6]  # rad/s, initial yaw rate
    acc_1 = initial_conditions[7]  # m/s^2, initial acceleration
    delta_f_1 = initial_conditions[8]  # rad, initial front wheel angle
    X_CL_0 = initial_conditions[9]  # m, initial current lane leading car X coordinate
    Y_CL_0 = initial_conditions[10]  # m, initial current lane leading car Y coordinate
    v_CL_0 = initial_conditions[11]  # m/s, initial current lane leading car velocity

    # Vehicle parameters
    m = 1270  # kg, vehicle mass
    Iz = 1536.7  # kg⋅m^2, moment of inertia about the z-axis
    a = 1.015  # m, distance from CoG to front axles
    b = 1.895  # m, distance from CoG to rear axles
    Cf = 1250  # N/rad, front axle cornering stiffness
    Cr = 755  # N/rad, rear axle cornering stiffness

    # Initialize the simulation
    simulation_results = np.array([initial_conditions])  # Record results
    num_steps = int(simulation_time / Ts)  # Calculate total number of simulation steps
    current_conditions = initial_conditions
    t_k = t_0
    X_k = X_0
    Y_k = Y_0
    psi_k = psi_0
    vx_k = vx_0
    vy_k = vy_0
    omega_r_k = omega_r_0
    acc_k_1 = acc_1
    delta_f_k_1 = delta_f_1
    X_CL_k = X_CL_0
    Y_CL_k = Y_CL_0
    v_CL_k = v_CL_0

    t_record = np.array([])

    for step in range(num_steps):
        # Run Controller
        acc_k = 0
        delta_f_k = 0
        if step % (Tc // Ts) == 0:
            if controller_flag == 0:
                # MPC
                start_time = time.time()
                [acc_k, delta_f_k] = mpc_controller(current_conditions, vx_ref, Y_ref)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
            elif controller_flag == 1:
                # AMPC
                features = current_conditions[1:7].copy()  # Extract relevant state variables
                features[0] = 0  # Reset X to 0
                features[1] -= Y_ref  # Normalize lateral position
                features = np.append(features, 0)  # Append Y_ref offset
                features = np.append(features, vx_ref)  # Append reference speed
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                start_time = time.time()
                controller_output = AMPC_model_LC(features)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
                acc_k = controller_output[0][0].item()
                delta_f_k = controller_output[0][1].item()
            elif controller_flag == 2:
                # HFAMPC
                features = current_conditions[1:7].copy()  # Extract relevant state variables
                features[0] = 0  # Reset X to 0
                features[1] -= Y_ref  # Normalize lateral position
                features = np.append(features, 0)  # Append Y_ref offset
                features = np.append(features, vx_ref)  # Append reference speed
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                start_time = time.time()
                _, controller_output, _ = HFAMPC_model_LC(features)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
                acc_k = controller_output[0][0].item()
                delta_f_k = controller_output[0][1].item()
            elif controller_flag == 3:
                # DPC
                features = current_conditions[1:7].copy()
                features[0] = 0
                features[1] -= Y_ref
                features = np.append(features, [0, vx_ref])
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                start_time = time.time()
                initial_state, controller_output, reference = DPC_model_LC(features)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
                acc_k = controller_output[0][0][0].item()
                delta_f_k = controller_output[0][0][1].item()
            elif controller_flag == 4:
                # RPC
                features = current_conditions[1:7].copy()  # Extract relevant state variables
                features[0] = 0  # Reset X to 0
                features[1] -= Y_ref  # Normalize lateral position
                features = np.append(features, 0)  # Append Y_ref offset
                features = np.append(features, vx_ref)  # Append reference speed
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                start_time = time.time()
                _, controller_output, _ = RPC_model_LC(features)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
                acc_k = controller_output[0][0].item()
                delta_f_k = controller_output[0][1].item()
            elif controller_flag == 5:
                # HFRPC
                features = current_conditions[1:7].copy()  # Extract relevant state variables
                features[0] = 0  # Reset X to 0
                features[1] -= Y_ref  # Normalize lateral position
                features = np.append(features, 0)  # Append Y_ref offset
                features = np.append(features, vx_ref)  # Append reference speed
                features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                start_time = time.time()
                _, controller_output, _ = HFRPC_model_LC(features)
                end_time = time.time()
                t_record = np.append(t_record, end_time - start_time)
                acc_k = controller_output[0][0].item()
                delta_f_k = controller_output[0][1].item()
            else:
                print("Controller not found!")
        else:
            acc_k = acc_k_1
            delta_f_k = delta_f_k_1

        # Update states
        deriv_X = vx_k * np.cos(psi_k) - vy_k * np.sin(psi_k)
        deriv_Y = vx_k * np.sin(psi_k) + vy_k * np.cos(psi_k)
        deriv_psi = omega_r_k
        deriv_vx = acc_k
        deriv_vy = -2 * (Cf + Cr) * vy_k / (m * vx_k) - (2 * (a * Cf - b * Cr) / (m * vx_k) + vx_k) * omega_r_k + 2 * Cf * delta_f_k / m
        deriv_omega_r = -2 * (a * Cf - b * Cr) * vy_k / (Iz * vx_k) - 2 * (a ** 2 * Cf + b ** 2 * Cr) * omega_r_k / (Iz * vx_k) + 2 * a * Cf * delta_f_k / Iz
        deriv_X_CL = v_CL_k

        t_k = current_conditions[0] + Ts
        X_k = current_conditions[1] + deriv_X * Ts
        Y_k = current_conditions[2] + deriv_Y * Ts
        psi_k = current_conditions[3] + deriv_psi * Ts
        vx_k = current_conditions[4] + deriv_vx * Ts
        vy_k = current_conditions[5] + deriv_vy * Ts
        omega_r_k = current_conditions[6] + deriv_omega_r * Ts
        acc_k_1 = acc_k
        delta_f_k_1 = delta_f_k
        X_CL_k = current_conditions[9] + deriv_X_CL * Ts
        Y_CL_k = Y_CL_k
        v_CL_k = v_CL_k

        current_conditions = np.array([t_k, X_k, Y_k, psi_k, vx_k, vy_k, omega_r_k, acc_k_1, delta_f_k_1, X_CL_k, Y_CL_k, v_CL_k])
        simulation_results = np.append(simulation_results, [current_conditions], axis=0)

    t_average = t_record.mean()
    return simulation_results, t_average, t_record


