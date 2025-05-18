# stability analysis for del{y}/del{t} + del{y}/del{x} = 0 using 2-time step discretization.
# y_{n+1} = y_{n} + del{t}[ a( -del{y}/del{x}  )_{n}  + b(  -del{y}/del{x}  )_{n-1}    ]
# y_{n+1} = y_{n} - del{t}/del{x}[  aDy_{n} + bDy_{n-1}], where D is upwind matrix.
# substituting y_{n} = r^{n}v, where v is eigenvector of D corresponding to the largest eigenvalue \mu, Dv = \mu{v}. 

--------------------------------------------------------------------------------
      Coefficients        |   Max dt   | Stable dt (0.9*max)  | Unstable dt (1.1*max)
--------------------------------------------------------------------------------
      a=1.5, b=-0.5       |  0.004999  |       0.004499       |       0.005499      
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
        (1.5, -0.5, "a=1.5, b=-0.5"),
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
    cmap = plt.get_cmap("tab20", 21)
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
