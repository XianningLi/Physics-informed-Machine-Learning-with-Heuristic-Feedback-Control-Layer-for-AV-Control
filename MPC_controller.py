import casadi as ca
import numpy as np
import time

def mpc_controller(current_conditions, vx_ref, Y_ref):

    # Define time step and prediction horizon
    dt = 0.5
    Np = 10  # Prediction horizon length

    # Current vehicle state
    X_0 = 0  # m, initial X coordinate, normalized to 0, convenient for DPC training later
    Y_0 = current_conditions[2]  # m, initial Y coordinate
    psi_0 = current_conditions[3]  # rad, initial yaw angle
    vx_0 = current_conditions[4]  # m/s, initial longitudinal velocity
    vy_0 = current_conditions[5]  # m/s, initial lateral velocity
    omega_r_0 = current_conditions[6]  # rad/s, initial yaw rate
    acc_1 = current_conditions[7]  # m/s^2, initial acceleration
    delta_f_1 = current_conditions[8]  # rad, initial front wheel angle
    X_CL_0 = current_conditions[9] - current_conditions[1]  # m, initial leading car X coordinate in current lane
    Y_CL_0 = current_conditions[10]  # m, initial leading car Y coordinate in current lane
    v_CL_0 = current_conditions[11]  # m/s, initial velocity of leading car in current lane

    # Vehicle parameters
    m = 1270  # kg, vehicle mass
    Iz = 1536.7  # kg⋅m^2, vehicle moment of inertia about the z-axis
    a = 1.015  # m, distance from the center of gravity to the front axles
    b = 1.895  # m, distance from the center of gravity to the rear axles
    Cf = 1250  # N/rad, cornering stiffness of the front axles
    Cr = 755  # N/rad, cornering stiffness of the rear axles
    lane_width = 4  # m, lane width

    # Define state variables and control variables
    X = ca.SX.sym('X')
    Y = ca.SX.sym('Y')
    psi = ca.SX.sym('psi')
    vx = ca.SX.sym('vx')
    vy = ca.SX.sym('vy')
    omega_r = ca.SX.sym('omega_r')
    states = ca.vertcat(X, Y, psi, vx, vy, omega_r)  # State vector
    n_states = states.numel()

    acc = ca.SX.sym('acc')
    delta_f = ca.SX.sym('delta_f')
    controls = ca.vertcat(acc, delta_f)  # Control vector
    n_controls = controls.numel()

    # Define system dynamic equations
    X_dot = vx * ca.cos(psi) - vy * ca.sin(psi)
    Y_dot = vx * ca.sin(psi) + vy * ca.cos(psi)
    psi_dot = omega_r
    vx_dot = acc
    vy_dot = - 2 * (Cf + Cr) / (m * vx) * vy - (2 * (a * Cf - b * Cr) / (m * vx) + vx) * omega_r + 2 * Cf / m * delta_f
    wr_dot = - 2 * (a * Cf - b * Cr) / (Iz * vx) * vy - 2 * (a ** 2 * Cf + b ** 2 * Cr) / (Iz * vx) * omega_r + 2 * a * Cf / Iz * delta_f

    # Combine system dynamic equations into vector form
    f_dynamic = ca.vertcat(X_dot, Y_dot, psi_dot, vx_dot, vy_dot, wr_dot)

    # Package dynamic equations as a CasADi function
    system_dynamics = ca.Function('system_dynamics', [states, controls], [f_dynamic])

    # Define weight matrices for states and control inputs
    Qx = ca.diag([0, 60, 500, 50, 10, 100])  # State weight matrix
    Qu = ca.diag([5, 500])  # Control input weight matrix
    Qdelta = ca.diag([5, 500])  # Weight matrix for control input changes
    Qt = ca.diag([0, 60, 1000, 70, 20, 200])  # Terminal weight matrix

    # Velocity constraints for state variables and control inputs
    vx_min, vx_max = 40/3.6, 120/3.6  # Velocity range
    acc_min, acc_max = -3.0, 3.0  # Acceleration range
    delta_f_min, delta_f_max = -0.3, 0.3  # Front wheel angle range

    # Initial state and reference state
    x0 = ca.DM([X_0, Y_0, psi_0, vx_0, vy_0, omega_r_0])  # Initial state
    x_ref = ca.DM([0, Y_ref, 0, vx_ref, 0, 0])  # Reference state

    # Initialize decision variables and constraints
    U = ca.SX.sym('U', n_controls, Np)
    X = x0

    # Objective function
    J = 0

    # Constraint bounds
    g = []
    lbg = []
    ubg = []
    lbx = []
    ubx = []
    for k in range(Np):
        # Update state
        diff = system_dynamics(X, U[:, k])
        X_next = X + diff * dt

        # State deviation
        J += (X_next - x_ref).T @ Qx @ (X_next - x_ref)

        # Control input
        J += U[:, k].T @ Qu @ U[:, k]

        # Add velocity constraints
        g.append(X_next[3])  # X_next[3] is vx

        # Update state
        X = X_next

        # Add control input constraints
        lbg += [vx_min]
        ubg += [vx_max]
        lbx += [acc_min, delta_f_min]
        ubx += [acc_max, delta_f_max]

    # Add terminal penalty
    J = J + (X - x_ref).T @ Qt @ (X - x_ref)

    # Convert to NLP problem
    nlp = {'f': J, 'x': ca.vec(U), 'g': ca.vertcat(*g)}

    # Set solver options
    opts = {'ipopt.print_level': 0, 'print_time': 0}

    # Create solver
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve the problem
    sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    # Extract solution
    u_opt = np.array(sol['x']).reshape((Np, n_controls)).T

    print(f"{u_opt} \n")

    acc_k = u_opt[0][0]
    delta_f_k = u_opt[1][0]
    return [acc_k, delta_f_k]
