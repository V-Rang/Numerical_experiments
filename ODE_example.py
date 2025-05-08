# simulating dy/dt = -\lambda y

import numpy as np
import matplotlib.pyplot as plt

def simulate_2point(a, b, dt, y0, nsteps, scale_val):
    """
    Two‑step multistep for dy/dt = -y:
      y_{n+1} = y_n + dt*(a * f(y_{n-1}) + b * f(y_n))
    with a + b = 1.

    Bootstrap step (n=0->1) uses forward Euler: y1 = y0 + dt * b * f(y0).

    Inputs:
      a, b       : scheme coefficients (a+b should = 1)
      dt         : time step
      y0         : initial scalar
      nsteps     : total number of steps to take

    Returns:
      y : array of length nsteps+1
    """
    # storage
    y = np.zeros(nsteps+1)
    y[0] = y0

    # first step: forward Euler variant (only b-term)
    f0 = -scale_val * y[0]
    y[1] = y[0] + dt * b * f0

    # multistep for n>=1
    for n in range(1, nsteps):
        f_nm1 = -scale_val * y[n-1]
        f_n   = -scale_val * y[n]
        y[n+1] = y[n] + dt * (a * f_nm1 + b * f_n)

    return y

def simulate_3point(m, n_coeff, p, dt, y0, nsteps, scale_val):
    """
    Three‑step multistep for dy/dt = -y:
      y_{n+1} = y_n + dt*(m * f(y_{n-2})
                          + n_coeff * f(y_{n-1})
                          + p * f(y_n))
    with m + n_coeff + p = 1.

    Bootstraps:
      - step 0->1: forward Euler with p‑term only
      - step 1->2: two‑step with n_coeff & p

    Inputs:
      m, n_coeff, p : scheme coefficients
      dt            : time step
      y0            : initial scalar
      nsteps        : total number of steps

    Returns:
      y : array of length nsteps+1
    """
    y = np.zeros(nsteps+1)
    y[0] = y0

    # step 1: forward Euler (p only)
    f0 = -scale_val * y[0]
    y[1] = y[0] + dt * p * f0

    if nsteps >= 2:
        # step 2: two‑step with n_coeff & p
        f0 = -scale_val * y[0]
        f1 = -scale_val * y[1]
        y[2] = y[1] + dt * (n_coeff * f0 + p * f1)

    # full three‑step for n>=2
    for k in range(2, nsteps):
        f_km2 = -scale_val * y[k-2]
        f_km1 = -scale_val * y[k-1]
        f_k = -scale_val * y[k]
        y[k+1] = y[k] + dt * (m * f_km2 + n_coeff * f_km1 + p * f_k)

    return y


if __name__ == "__main__":
    # example parameters
    dt = 1.4
    y0 = 1.0
    nsteps = 50
    scale_val = 0.3

    # 2‑step: e.g. trapezoidal rule (a=0.5, b=0.5)
    y2 = simulate_2point(a=-0.5, b=1.5, dt=dt, y0=y0, nsteps=nsteps, scale_val = scale_val)

    # 3‑step: pick e.g. Adams–Bashforth 3 coefficients
    # (m, n_coeff, p) = (0, -1/2, 3/2) but note these sum to 1
    y3 = simulate_3point(m=0., n_coeff= 0., p=1., dt=dt, y0=y0, nsteps=nsteps, scale_val=scale_val)

    # analytic solution
    t = np.arange(0, nsteps+1)*dt
    y_true = y0 * np.exp(-scale_val * t)

    print(np.linalg.norm(y_true - y2))
    # plot
    plt.plot(t, y_true, 'k-', label='exact')
    plt.plot(t, y2,    'ro--', label='2‑step (trapezoidal)')
    plt.plot(t, y3,    'bs--', label='3‑step AB3')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.show()
