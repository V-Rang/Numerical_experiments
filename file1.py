import numpy as np

# def D_upwind(u, dx):
#     """First-order upwind derivative (c=+1) with periodic BCs."""
#     return (u - np.roll(u, 1)) / dx


def D_upwind_periodic_matrix(N):
    P = np.eye(N);  
    P = np.roll(P, 1, axis=0)

    # D = (I − P) / dx
    D = (np.eye(N) - P)
    return D


def simulate(a, b, dx, dt, u0, nsteps, verbose = True, tol = 1e-2):
    """
    Simulate u_{n+1} = u_n - (dt/dx)[a*D u_{n-1} + b*D u_n].

    Inputs:
      a, b       : scheme coefficients
      dx, dt     : space/time steps
      u0         : 1D array, initial state at n=0
      nsteps     : number of time steps to take
      scheme     : 'upwind' or 'central'

    Returns:
      us : array of shape (nsteps+1, len(u0)), containing u^0,…,u^nsteps
    """

    D = D_upwind_periodic_matrix(u0.shape[0])

    M0 = np.max(np.abs(u0))
    threshold = (1 + tol)*M0

    N = u0.size
    us = np.zeros((nsteps+1, N))
    us[0] = u0.copy()

    # first step: use a 1‑step method (e.g. forward‐Euler => a=0 term only)
    us[1] = u0 - (dt/dx) * b * D @ u0

    # now apply the two‑step formula
    for n in range(1, nsteps):
        # Du_nm1 = D(us[n-1], dx)
        # Du_n   = D(us[n  ], dx)
        
        Du_nm1 = D @ us[n-1]
        Du_n   = D @ us[n  ]
        
        u_next = us[n] - (dt/dx)*(a*Du_nm1 + b*Du_n)

        us[n+1] = u_next

    return us

def simulate_3point(m, n_coeff, p, dx, dt, u0, nsteps, verbose=True, tol=1e-6):
    """
    Simulate the 3‑point stencil
      u_{n+1} = u_n - (dt/dx)[ m D u_{n-2} + n_coeff D u_{n-1} + p D u_n ]
    with:
      * forward‑Euler for step 1  (only the p‑term)
      * the 2‑step formula for step 2 (n_coeff & p terms)
      * the full 3‑point formula thereafter

    Inputs:
      m, n_coeff, p : stencil coefficients
      dx, dt        : space/time steps
      u0            : 1D array, initial state at n=0
      nsteps        : how many steps to take
      scheme        : 'upwind' or 'central'
      tol           : relative tol for max‑norm blow‑up check
      verbose       : if True, prints blow‑up step

    Returns:
      us      : array of shape (Nsteps+1, len(u0)), with u^0…u^Nsteps
      blew_up : bool, True if we stopped early due to blow‑up
    """
    D = D_upwind_periodic_matrix(u0.shape[0])
    N = u0.size
    us = np.zeros((nsteps+1, N))
    us[0] = u0.copy()

    # compute threshold from initial max‐norm
    M0 = np.max(np.abs(u0))
    threshold = (1 + tol) * M0

    # step 1: only the p‐term (like a forward‐Euler bootstrap)
    # us[1] = u0 - (dt/dx) * p * D(u0, dx)
    us[1] = u0 - (dt/dx) * p * D @ u0

    # if np.max(np.abs(us[1])) > threshold:
    #     if verbose:
    #         print("Blow‑up at step 1")
    #     return us[:2], True

    if nsteps >= 2:
        # step 2: use n_coeff and p (m term is zero)
        # us[2] = us[1] - (dt/dx)*(n_coeff*D(us[0],dx) + p*D(us[1],dx))
        us[2] = us[1] - (dt/dx)*(n_coeff*D @ us[0] + p*D @ us[1])
        
        
        # if np.max(np.abs(us[2])) > threshold:
        #     if verbose:
        #         print("Blow‑up at step 2")
        #     return us[:3], True

    # general 3‑point stencil for n>=2
    for k in range(2, nsteps):
        # Du_km2 = D(us[k-2], dx)
        # Du_km1 = D(us[k-1], dx)
        # Du_k    = D(us[k  ], dx)

        Du_km2 = D @ us[k-2]
        Du_km1 = D @ us[k-1]
        Du_k    = D @ us[k  ]

        u_next = us[k] - (dt/dx)*(m*Du_km2 + n_coeff*Du_km1 + p*Du_k)

        # if np.max(np.abs(u_next)) > threshold or np.isnan(u_next).any():
        #     if verbose:
        #         print(f"Blow‑up at step {k+1}")
        #     us[k+1] = u_next
        #     return us[:k+2], True

        us[k+1] = u_next

    return us

def check_blowup(us, u0, tol=1e-6):
    """
    Checks if max|u| ever exceeds (1+tol)*max|u0| in a trajectory us.

    Returns:
      blew_up (bool), step_of_failure (int or None)
    """
    M0 = np.max(np.abs(u0))
    threshold = (1 + tol) * M0
    for n, u in enumerate(us):
        if np.max(np.abs(u)) > threshold or np.isnan(u).any():
            return True, n
    return False, None

def random_fourier_ic(x, n_mode, seed=None):
    """
    Build u0(x) = sum_{i=1}^{n_mode} [alpha_i sin(2π i x) + beta_i cos(2π i x)]
    with alpha_i, beta_i ~ N(0,1) (or uniform if you prefer).
    """
    if seed is not None:
        np.random.seed(seed)
    alpha = np.random.randn(n_mode)
    beta  = np.random.randn(n_mode)

    u0 = np.zeros_like(x)
    for i in range(1, n_mode+1):
        u0 += alpha[i-1]*np.sin(2*np.pi*i*x) + beta[i-1]*np.cos(2*np.pi*i*x)
    return u0

# example usage
if __name__ == "__main__":
    # parameters
    m_val, n_val, p_val = 0.65, 0.05, 0.3  
    
    # a_val, b_val = 1.5 , -0.5
    scale_factor = 0.1
    dx    = 0.005
    dt = scale_factor * dx
    x = np.arange(0,1+dx, dx)
    n_mode = 3
    u0 = random_fourier_ic(x, n_mode, seed=42)
    nsteps = 200

    # us = simulate(a_val, b_val, dx, dt, u0, nsteps, verbose = True, tol = 1e-2)
    
    us = simulate_3point(m_val, n_val, p_val, dx, dt, u0, nsteps)

    print(np.linalg.norm(us[0]), ":", np.linalg.norm(us[-1]))


import matplotlib.pyplot as plt
plt.plot(x, u0)
plt.plot(x, us[0],label='t=0')
plt.plot(x, us[-1],label='t=-1')
plt.legend()
plt.show()
