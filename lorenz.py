import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def lorenz(t, u, sigma=10.0, r=28.0, b=8.0 / 3.0):
    x, y, z = u

    dxdt = sigma * (y - x)
    dydt = r * x - y - x * z
    dzdt = x * y - b * z

    return np.array([dxdt, dydt, dzdt])


def rk4(f, u0, t0, tf, h):
    t_values = np.arange(t0, tf, h)
    t_values = np.append(t_values, tf)
    u_values = np.zeros((len(t_values), len(u0)))

    u_values[0] = u0

    for n in range(len(t_values) - 1):
        t = t_values[n]
        u = u_values[n]
        dt = t_values[n + 1] - t

        k1 = f(t, u)
        k2 = f(t + dt / 2, u + dt * k1 / 2)
        k3 = f(t + dt / 2, u + dt * k2 / 2)
        k4 = f(t + dt, u + dt * k3)

        u_values[n + 1] = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return t_values, u_values


def convergence_test():
    u0 = np.array([1.0, 1.0, 1.0])
    t0 = 0.0
    tf = 5.0   
    h_values = [0.02, 0.01, 0.005, 0.0025]
    h_ref = 0.0005

    # reference solution
    t_ref, u_ref = rk4(lorenz, u0, t0, tf, h_ref)

    # interpolate each component
    interp_x = interp1d(t_ref, u_ref[:, 0], kind='cubic')
    interp_y = interp1d(t_ref, u_ref[:, 1], kind='cubic')
    interp_z = interp1d(t_ref, u_ref[:, 2], kind='cubic')

    print("Convergence test (fixed):")
    print()
    print(f"{'h':<12}{'error':<20}")
    print("-" * 32)

    for h in h_values:
        t, u = rk4(lorenz, u0, t0, tf, h)

        # interpolate reference onto this grid
        u_ref_interp = np.zeros_like(u)
        u_ref_interp[:, 0] = interp_x(t)
        u_ref_interp[:, 1] = interp_y(t)
        u_ref_interp[:, 2] = interp_z(t)

        # compute max error over time (better than final-only)
        error = np.max(np.linalg.norm(u - u_ref_interp, axis=1))

        print(f"{h:<12}{error:<20.12e}")


def plot_lorenz_3d():
    u0 = np.array([1.0, 1.0, 1.0])
    t0 = 0.0
    tf = 40.0
    h = 0.01

    _, u = rk4(lorenz, u0, t0, tf, h)

    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x, y, z)
    ax.set_title("Lorenz System Trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.savefig("lorenz_3d.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_sensitivity():
    u0 = np.array([1.0, 1.0, 1.0])
    u0_close = np.array([1.0001, 1.0, 1.0])

    t0 = 0.0
    tf = 40.0
    h = 0.01

    t, u = rk4(lorenz, u0, t0, tf, h)
    _, u_close = rk4(lorenz, u0_close, t0, tf, h)

    distance = np.linalg.norm(u - u_close, axis=1)

    plt.figure()
    plt.plot(t, distance)
    plt.title("Sensitivity to Initial Conditions")
    plt.xlabel("Time")
    plt.ylabel("Distance between solutions")

    plt.savefig("lorenz_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    convergence_test()
    plot_lorenz_3d()
    plot_sensitivity()