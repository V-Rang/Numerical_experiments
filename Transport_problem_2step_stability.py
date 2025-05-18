# stability analysis for del{y}/del{t} + del{y}/del{x} = 0 using 2-time step discretization.
# y_{n+1} = y_{n} + del{t}[ a( -del{y}/del{x}  )_{n}  + b(  -del{y}/del{x}  )_{n-1}    ]
# y_{n+1} = y_{n} - del{t}/del{x}[  aDy_{n} + bDy_{n-1}], where D is upwind matrix.
# substituting y_{n} = r^{n}v, where v is eigenvector of D corresponding to the largest eigenvalue \mu, Dv = \mu{v}. 

--------------------------------------------------------------------------------
      Coefficients        |   Max dt   | Stable dt (0.9*max)  | Unstable dt (1.1*max)
--------------------------------------------------------------------------------      
      a=0.5, b=0.5        |  0.009999  |       0.008999       |       0.010999      
     a=0.55, b=0.45       |  0.011110  |       0.009999       |       0.012221      
      a=0.6, b=0.4        |  0.012500  |       0.011250       |       0.013750      
      a=0.7, b=0.3        |  0.016666  |       0.015000       |       0.018333      
     a=0.75, b=0.25       |  0.020000  |       0.018000       |       0.021999      
      a=0.8, b=0.2        |  0.016666  |       0.015000       |       0.018333      
     a=0.85, b=0.15       |  0.014285  |       0.012857       |       0.015714      
      a=0.9, b=0.1        |  0.012500  |       0.011250       |       0.013750      
     a=0.95, b=0.05       |  0.011110  |       0.009999       |       0.012221      
       a=1., b=0.         |  0.009999  |       0.008999       |       0.010999      
     a=1.05, b=-0.05      |  0.009090  |       0.008181       |       0.009999      
      a=1.1, b=-0.1       |  0.008333  |       0.007500       |       0.009167      
     a=1.15, b=-0.15      |  0.007691  |       0.006922       |       0.008461      
      a=1.2, b=-0.2       |  0.007142  |       0.006428       |       0.007856      
     a=1.25, b=-0.25      |  0.006666  |       0.006000       |       0.007333      
      a=1.3, b=-0.3       |  0.006249  |       0.005624       |       0.006874      
     a=1.35, b=-0.35      |  0.005882  |       0.005294       |       0.006470      
      a=1.4, b=-0.4       |  0.005555  |       0.005000       |       0.006111      
     a=1.45, b=-0.45      |  0.005262  |       0.004736       |       0.005789      
      a=1.5, b=-0.5       |  0.004999  |       0.004499       |       0.005499   

import numpy as np
import matplotlib.pyplot as plt

def adams_bashforth_2step_roots(h, dx, mu, a, b):
    """Calculate the roots of the characteristic polynomial for Adams-Bashforth 2-step method."""
    # z = lambda_val * h
    z = (h/dx)*mu
    # Characteristic polynomial: r^2 - (1 + a*z)*r - b*z = 0
    coeffs = [1, -(1 - a*z), b*z]
    return np.roots(coeffs)

def is_stable(h, dx, mu, a, b):
    """Check if the Adams-Bashforth method is stable for given h and lambda."""
    roots = adams_bashforth_2step_roots(h, dx, mu, a, b)
    return np.all(np.abs(roots) < 1)

def manual_find_max_dt(dx, mu, a, b, precision=1e-6):
    """Find maximum stable step size using binary search."""
    dt_low = 0.0
    dt_high = 1.0
    
    # First, find an unstable h_high
    while is_stable(dt_high, dx, mu, a, b):
        dt_high *= 2
        if dt_high > 100:  # Safety check
            return float('inf')  # Method might be A-stable
    
    # Binary search for the threshold
    while dt_high - dt_low > precision:
        dt_mid = (dt_low + dt_high) / 2
        if is_stable(dt_mid, dx, mu, a, b):
            dt_low = dt_mid
        else:
            dt_high = dt_mid
    
    return dt_low

def plot_stability_region(a, b, dx, mu, ax=None, color='blue', label=None):
    """Plot the stability region for Adams-Bashforth 2-step method with coefficients a and b."""
    # Find maximum stable step size
    max_dt = manual_find_max_dt(dx, mu, a, b)
    critical_z = (max_dt / dx) * mu
    
    # Create a grid of points in the complex plane
    real = np.linspace(-4.5, 0.5, 200)
    imag = np.linspace(-1.5, 1.5, 200)
    re_grid, im_grid = np.meshgrid(real, imag)
    
    # Initialize stability matrix
    stability = np.zeros_like(re_grid, dtype=bool)
    
    # For each point, check if the method is stable
    for i in range(len(real)):
        for j in range(len(imag)):
            z = complex(real[i], imag[j])
            # For Adams-Bashforth 2-step
            coeffs = [1, -(1 - a*z), b*z]
            roots = np.roots(coeffs)
            stability[j, i] = np.all(np.abs(roots) < 1)
    
    # Plot the result
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    contour = ax.contourf(re_grid, im_grid, stability, levels=[0, 0.5, 1], colors=['white', color], alpha=0.3)
    ax.contour(re_grid, im_grid, stability, levels=[0.5], colors=[color], linewidths=2)
    
    # Mark the point lambda*h_max on the real axis
    if not np.isinf(max_dt):
        ax.plot([critical_z], [0], 'o', color=color, markersize=6)
    
    return max_dt, contour

def compare_stability_regions():
    """Compare stability regions for different coefficient pairs."""
    
    N  = 100         # number of grid points
    L  = 1.0        # domain length
    dx = L / N

    # 1) Build D and get its eigenvalues (no 1/dx here)
    D = D_upwind_periodic_matrix(N)
    mu_vals = np.linalg.eigvals(D)
    
    # Define the four coefficient pairs
    coefficient_pairs = [
        (0.5, 0.5, "a=0.5, b=0.5"), 
        (0.55, 0.45, "a=0.55, b=0.45"),
        (0.6, 0.4, "a=0.6, b=0.4"),
        (0.7, 0.3, "a=0.7, b=0.3"),
        (0.75, 0.25, "a=0.75, b=0.25"),
        (0.8, 0.2, "a=0.8, b=0.2"),
        (0.85, 0.15, "a=0.85, b=0.15"),
        (0.9, 0.1, "a=0.9, b=0.1"),
        (0.95, 0.05, "a=0.95, b=0.05"),
        (1., 0., "a=1., b=0."),
        (1.05, -0.05, "a=1.05, b=-0.05"),
        (1.1, -0.1, "a=1.1, b=-0.1"),
        (1.15, -0.15, "a=1.15, b=-0.15"),
        (1.2, -0.2, "a=1.2, b=-0.2"),
        (1.25, -0.25, "a=1.25, b=-0.25"),
        (1.3, -0.3, "a=1.3, b=-0.3"),
        (1.35, -0.35, "a=1.35, b=-0.35"),
        (1.4, -0.4, "a=1.4, b=-0.4"),
        (1.45, -0.45, "a=1.45, b=-0.45"),
        (1.5, -0.5, "a=1.5, b=-0.5")
    ]
    
    # Colors for different regions
    # colors = ['blue', 'red', 'green', 'purple']
    cmap = plt.get_cmap("tab20", 20)
    colors = list(cmap.colors)   # a list of 21 (r,g,b,a) tuples

    # Create a figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 7))
    
    # Lambda value for test equation
    # lambda_val = -10
  
    # Plot all stability regions in one plot
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.grid(alpha=0.3)
    ax1.set_xlabel('Re(z)', fontsize=12)
    ax1.set_ylabel('Im(z)', fontsize=12)
    ax1.set_title('Stability Regions Comparison', fontsize=14)
    
    # Plot each region and store max_h values
    max_dt_values = []
    for i, (a, b, label) in enumerate(coefficient_pairs):
        max_dt, contour = plot_stability_region(a, b, dx, mu_vals[0], ax1, colors[i])
        # max_dt, contour = plot_stability_region(a, b, dx, mu_vals[0], ax1)
        max_dt_values.append(max_dt)
    
    # Add a legend
    legend_elements = [plt.Line2D([0], [0], color=color, lw=2, label=f"{label} (Max dt={max_h:.4f})") 
                      for (a, b, label), color, max_h in zip(coefficient_pairs, colors, max_dt_values)]

    # legend_elements = [plt.Line2D([0], [0], lw=2, label=f"{label} (Max dt={max_h:.4f})") 
    #                   for (a, b, label), max_h in zip(coefficient_pairs, max_dt_values)]

    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
    # Create a table of maximum step sizes
    print(f"Maximum stable step sizes:")
    print("-" * 60)
    print(f"{'Coefficients':^30} | {'Max dt':^12} | {'z':^12}")
    print("-" * 60)
    for (a, b, label), max_dt in zip(coefficient_pairs, max_dt_values):
        critical_z = (max_dt/dx)*mu_vals[0]
        print(f"{label:^30} | {max_dt:^12.6f} | {critical_z:^12.6f}")
    print("-" * 60)

# Run the comparison
compare_stability_regions()


########################################################################################

import numpy as np
import matplotlib.pyplot as plt

def D_upwind_periodic_matrix(N):
    P = np.eye(N)
    P = np.roll(P, 1, axis=0)
    return np.eye(N) - P

def adams_bashforth_2step_roots(dt ,dx, mu, a, b):
    """Calculate the roots of the characteristic polynomial for Adams-Bashforth 2-step method."""
    z = (dt/dx)*mu
    coeffs = [1, -(1 - a*z), b*z]
    return np.roots(coeffs)

def is_stable(dt, dx, mu, a, b):
    """Check if the Adams-Bashforth method is stable for given h and lambda."""
    roots = adams_bashforth_2step_roots(dt, dx, mu, a, b)
    return np.all(np.abs(roots) < 1)

def manual_find_max_dt(dx, mu, a, b, precision=1e-6):
    """Find maximum stable step size using binary search."""
    dt_low = 0.0
    dt_high = 1.0
    
    while is_stable(dt_high, dx, mu, a, b):
        dt_high *= 2
        if dt_high > 100:  # Safety check
            return float('inf')
    
    while dt_high - dt_low > precision:
        dt_mid = (dt_low + dt_high) / 2
        if is_stable(dt_mid, dx, mu, a, b):
            dt_low = dt_mid
        else:
            dt_high = dt_mid
    
    return dt_low

def simulate_ode(y0, D_mat, x_vals, n_modes, seed, mu, h, steps, a, b):
    """Simulate the ODE y' + del{y}/del{x} = 0 with Adams-Bashforth 2-step method."""

    t = np.zeros(steps+1)
    y = np.zeros((steps+1, len(x_vals)) )
    exact = np.zeros((steps+1, len(x_vals)))
    dx = x_vals[1] - x_vals[0]

    # Initial conditions
    t[0] = 0
    y[0] = y0
    exact[0] = y0
    
    # Use exact solution for first step
    t[1] = h
    y[1] = random_fourier_solution(x_vals, n_modes, t[1], seed)
    exact[1] = y[1]
    
    # Apply Adams-Bashforth 2-step method
    for i in range(1, steps):
        f_n = D_mat @ y[i]
        f_nm1 = D_mat @ y[i-1]
        y[i+1] = y[i] - (h/dx) * (a * f_n + b * f_nm1)
        t[i+1] = t[i] + h
        exact[i + 1] = random_fourier_solution(x_vals, n_modes, t[i+1], seed)
    
    return t, y, exact


def random_fourier_solution(x, n_mode, t=0.0, seed=None):
    """
    Return the exact solution at time t of
       ∂y/∂t + ∂y/∂x = 0
    with y(x,0) = sum_{i=1}^n_mode [alpha_i sin(2π i x) + beta_i cos(2π i x)].
    
    u(x,t) = u0(x - t) with periodicity on [0,1).
    
    Parameters
    ----------
    x : array_like
        Points in [0,1) at which to evaluate y.
    n_mode : int
        Number of Fourier modes.
    t : float, optional
        Time at which to evaluate (default 0.0).
    seed : int or None, optional
        RNG seed for reproducibility (only used the first time).
    """

    # reproducible random coefficients
    if seed is not None:
        np.random.seed(seed)
    alpha = np.random.randn(n_mode)
    beta  = np.random.randn(n_mode)
    
    # characteristic shift and periodic wrap
    xi = (x - t) % 1.0
    
    # build the solution
    y = np.zeros_like(xi)
    for i in range(1, n_mode+1):
        y += alpha[i-1] * np.sin(2*np.pi*i*xi) \
           + beta[i-1]  * np.cos(2*np.pi*i*xi)

    return y

def comprehensive_stability_analysis():
    # Define the coefficient pairs
    coefficient_pairs = [
        (0.5, 0.5, "a=0.5, b=0.5"), 
        (0.55, 0.45, "a=0.55, b=0.45"),
        (0.6, 0.4, "a=0.6, b=0.4"),
        (0.7, 0.3, "a=0.7, b=0.3"),
        (0.75, 0.25, "a=0.75, b=0.25"),
        (0.8, 0.2, "a=0.8, b=0.2"),
        (0.85, 0.15, "a=0.85, b=0.15"),
        (0.9, 0.1, "a=0.9, b=0.1"),
        (0.95, 0.05, "a=0.95, b=0.05"),
        (1., 0., "a=1., b=0."),
        (1.05, -0.05, "a=1.05, b=-0.05"),
        (1.1, -0.1, "a=1.1, b=-0.1"),
        (1.15, -0.15, "a=1.15, b=-0.15"),
        (1.2, -0.2, "a=1.2, b=-0.2"),
        (1.25, -0.25, "a=1.25, b=-0.25"),
        (1.3, -0.3, "a=1.3, b=-0.3"),
        (1.35, -0.35, "a=1.35, b=-0.35"),
        (1.4, -0.4, "a=1.4, b=-0.4"),
        (1.45, -0.45, "a=1.45, b=-0.45"),
        (1.5, -0.5, "a=1.5, b=-0.5")
    ]
    
    # Colors for different methods
    cmap = plt.get_cmap("tab20", 20)
    colors = list(cmap.colors)   # a list of 21 (r,g,b,a) tuples
    
    # Test equation parameter
    N  = 100         # number of grid points
    L  = 1.0        # domain length
    dx = L / N
    x_vals = np.linspace(0, L, N)

    # 1) Build D and get its eigenvalues (no 1/dx here)
    D_mat = D_upwind_periodic_matrix(N)
    mu_vals = np.linalg.eigvals(D_mat)
    
    # Initial condition and simulation parameters
    # y0 = 1.0

    n_modes = 3
    seed = 42
    y0 = random_fourier_solution(x_vals, n_modes, 0., seed)
    # print(y0)

    t_end_stable = 10.0  # Longer time for stable comparison
    t_end_demo = 10.0     # Shorter time for stability demo
    
    # Calculate max_h for each coefficient pair
    max_dt_values = []
    for a, b, _ in coefficient_pairs:
        max_dt = manual_find_max_dt(dx, mu_vals[0], a, b)
        max_dt_values.append(max_dt)
    
    # Print table of results
    print(f"Stability Analysis for:")
    print("-" * 80)
    print(f"{'Coefficients':^25} | {'Max dt':^10} | {'Stable dt (0.9*max)':^20} | {'Unstable dt (1.1*max)':^20}")
    print("-" * 80)
    
    for (a, b, label), max_dt in zip(coefficient_pairs, max_dt_values):
        dt_stable = 0.9 * max_dt
        dt_unstable = 1.1 * max_dt
        print(f"{label:^25} | {max_dt:^10.6f} | {dt_stable:^20.6f} | {dt_unstable:^20.6f}")
    
    print("-" * 80)
    
    # FIGURE 1: Comparison of all methods at 0.9 * max_dt
    plt.figure(figsize=(12, 8))
    
    # Run stable simulations for comparison
    for i, ((a, b, label), max_dt) in enumerate(zip(coefficient_pairs, max_dt_values)):
        dt_stable = 0.9 * max_dt
        steps_stable = int(t_end_stable / dt_stable)
        t_stable, y_stable, exact_stable = simulate_ode(y0, D_mat, x_vals, n_modes, seed, mu_vals[0], dt_stable, steps_stable, a, b)

        plt.plot(x_vals, y_stable[-1], color=colors[i], label=f'{label} (h = {dt_stable:.4f})')
    
    # Add exact solution to comparison plot
    plt.plot(x_vals, exact_stable[-1], 'k--', label='Exact solution')
    
    # Finalize comparison plot
    plt.xlabel('Time t', fontsize=12)
    plt.ylabel('y(t)', fontsize=12)
    plt.title('Comparison of Different Adams-Bashforth 2-Step Methods (0.9 * max dt)', fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.ylim(-1.5, 1.5)
    plt.tight_layout()
    plt.savefig('adams_bashforth_comparison.png', dpi=300)
    
    # FIGURE 2: Individual stability demonstrations
    fig, axes = plt.subplots(21,1, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, ((a, b, label), max_dt) in enumerate(zip(coefficient_pairs, max_dt_values)):
        # Set up stable and unstable step sizes
        dt_stable = 0.9 * max_dt
        dt_unstable = 1.1 * max_dt
        
        # Calculate number of steps
        steps_stable = int(t_end_demo / dt_stable)
        steps_unstable = int(t_end_demo / dt_unstable)
        
        # # Run stable and unstable simulations        
        t_stable, y_stable, exact_stable = simulate_ode(y0, D_mat, x_vals, n_modes, seed, mu_vals[0], dt_stable, steps_stable, a, b)
        t_unstable, y_unstable, exact_unstable = simulate_ode(y0, D_mat, x_vals, n_modes, seed, mu_vals[0], dt_unstable, steps_unstable, a, b)

        # Plot on the corresponding subplot
        ax = axes[i]
        
        # Plot exact solution
        ax.plot(x_vals, exact_stable[-1], 'k--', label='Exact solution')
        
        # Plot stable solution
        ax.plot(x_vals, y_stable[-1], color=colors[i], label=f'Stable (h = {dt_stable:.4f})')


        row_norms = np.linalg.norm(y_unstable, axis=1)    # shape (m,)
        bad_rows = np.where(row_norms > 100)[0]  # indices of rows
            
        # Plot unstable solution (may need to limit if it blows up)
        if (len(bad_rows) > 0):  
          cutoff = bad_rows[0]
          ax.plot(x_vals, y_unstable[cutoff], linestyle='-.', color='r', label=f'Unstable (h = {dt_unstable:.4f})')            
        else:
          ax.plot(t_unstable, y_unstable[-1], linestyle='-.', color='r', label=f'Unstable (h = {dt_unstable:.4f})')
        
        # Set title and labels
        ax.set_title(label, fontsize=14)
        ax.set_xlabel('Time t', fontsize=12)
        ax.set_ylabel('y(t)', fontsize=12)
        ax.grid(True)
        ax.legend()
        
        # Set reasonable y-axis limits
        ax.set_ylim(-5, 5)
    
    # Finalize demonstration plot
    plt.suptitle('Stability Demonstration for Different Adams-Bashforth 2-Step Methods', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('adams_bashforth_stability_demonstration.png', dpi=300)
    
    # Show all figures
    plt.show()

# Run the analysis
comprehensive_stability_analysis()
