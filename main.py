import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import solve_ivp


# System constants
rho = 1.0    # Outer radius
delta = 0.1  # Inner radius
tau = 1.0    # Time scale
a = (rho**2 - delta**2)/tau  
b = delta**2                 
V = 1 * a/(2 * np.sqrt(b))      

def r(t):
    """Radius function"""
    return np.sqrt((t/tau)*(rho**2 - delta**2) + delta**2)

def phi_integrand(u):
    """Integrand for φ(t)"""
    denominator = a*u + b
    return np.sqrt((V**2/denominator) - (a**2)/(4*denominator**2))

# def compute_phi(t):
#     """Compute φ(t) by numerical integration"""
#     result, _ = integrate.quad(phi_integrand, 0, t, epsrel=1e-9, epsabs=1e-12)
#     return 


def compute_phi(t):
    """Compute φ(t) using solve_ivp for higher precision"""
    def dphi_dt(u, phi):
        denominator = a*u + b
        return np.sqrt((V**2/denominator) - (a**2)/(4*denominator**2))
    
    sol = solve_ivp(dphi_dt, [0, t], [0], method='RK45', rtol=1e-10, atol=1e-12)
    return sol.y[0][-1]  # Return the last value of φ(t)

def compute_speed(x, y, dt):
    """
    Compute speed using finite differences:
    speed = sqrt((dx/dt)² + (dy/dt)²)
    """
    dx = np.diff(x)
    dy = np.diff(y)
    speeds = np.sqrt((dx/dt)**2 + (dy/dt)**2)
    return speeds

def generate_single_trajectory(theta_k, points=100):
    """Generate a single trajectory with starting angle theta_k"""
    t_values = np.linspace(0, tau, points)
    r_values = [r(t) for t in t_values]
    phi_values = [compute_phi(t) for t in t_values]
    
    # Add theta_k to all phi values
    total_angles = [phi + theta_k for phi in phi_values]
    
    # Convert to Cartesian coordinates
    x_values = [r * np.cos(theta) for r, theta in zip(r_values, total_angles)]
    y_values = [r * np.sin(theta) for r, theta in zip(r_values, total_angles)]
    
    return x_values, y_values, t_values

def check_angular_coverage(x_all, y_all, num_sectors=6):
    """Check coverage uniformity by counting points in angular sectors"""
    r = np.sqrt(x_all**2 + y_all**2)
    theta = np.arctan2(y_all, x_all) % (2*np.pi)
    
    sector_edges = np.linspace(0, 2*np.pi, num_sectors + 1)
    sector_counts = []
    sector_areas = []
    
    for i in range(num_sectors):
        mask = (theta >= sector_edges[i]) & (theta < sector_edges[i+1])
        count = np.sum(mask)
        sector_counts.append(count)
        
        sector_area = (sector_edges[i+1] - sector_edges[i]) * (rho**2 - delta**2) / (2*np.pi)
        sector_areas.append(sector_area)
    
    total_points = len(x_all)
    point_fractions = np.array(sector_counts) / total_points
    area_fractions = np.array(sector_areas) / np.sum(sector_areas)
    
    return point_fractions, area_fractions

def plot_comprehensive_results(trajectories_x, trajectories_y, point_fractions, area_fractions):
    """Create comprehensive visualization"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15))
    
    # Trajectory plot
    for x, y in zip(trajectories_x, trajectories_y):
        ax1.plot(x, y, 'b-', linewidth=0.5, alpha=0.5)
    ax1.add_artist(plt.Circle((0, 0), rho, fill=False, color='red'))
    ax1.add_artist(plt.Circle((0, 0), delta, fill=False, color='red'))
    ax1.set_title('UAV Trajectories')
    ax1.set_aspect('equal')
    ax1.grid(True)
    
    # Position density plot
    x_all = np.concatenate(trajectories_x)
    y_all = np.concatenate(trajectories_y)
    heatmap, xedges, yedges = np.histogram2d(x_all, y_all, bins=50,
                                            range=[[-rho, rho], [-rho, rho]])
    im = ax2.imshow(heatmap.T, extent=[-rho, rho, -rho, rho], origin='lower',
                    cmap='hot', norm=LogNorm())
    ax2.add_artist(plt.Circle((0, 0), rho, fill=False, color='white'))
    ax2.add_artist(plt.Circle((0, 0), delta, fill=False, color='white'))
    ax2.set_title('Position Density')
    plt.colorbar(im, ax=ax2)
    
    # Speed analysis plot
# Speed analysis plot
    dt = 1 * tau/len(trajectories_x[0])
    first_traj_speed = compute_speed(np.array(trajectories_x[0]), 
                                   np.array(trajectories_y[0]), dt)
    times = np.linspace(0, 100*tau, len(first_traj_speed))
    ax3.plot(times/tau, first_traj_speed, 'g-', alpha=0.7)
    ax3.axvline(x=2, color='r', linestyle='--', label='t = 2τ')
    ax3.set_title('Speed over 200τ (First Trajectory)')
    ax3.set_xlabel('Time (t/τ)')
    ax3.set_ylabel('Speed')
    ax3.grid(True)
    ax3.legend()
    
    # Coverage comparison plot
    sectors = range(len(point_fractions))
    width = 0.35
    ax4.bar(np.array(sectors) - width/2, point_fractions, width, 
            label='Actual Coverage')
    ax4.bar(np.array(sectors) + width/2, area_fractions, width, 
            label='Theoretical Coverage')
    ax4.set_title('Angular Coverage Distribution')
    ax4.set_xlabel('Sector')
    ax4.set_ylabel('Fraction')
    ax4.legend()
    
    plt.tight_layout()
    return fig

def main():
    # Parameters
    num_trajectories = 1000
    points_per_traj = 1000
    num_sectors = 6
    
    # Generate trajectories
    thetas = np.random.uniform(0, 2*np.pi, num_trajectories)
    trajectories_x = []
    trajectories_y = []
    
    # Generate trajectories
    for theta_k in thetas:
        x_values, y_values, _ = generate_single_trajectory(theta_k, points_per_traj)
        trajectories_x.append(np.array(x_values))
        trajectories_y.append(np.array(y_values))
    
    # Analyze coverage
    x_all = np.concatenate(trajectories_x)
    y_all = np.concatenate(trajectories_y)
    point_fractions, area_fractions = check_angular_coverage(x_all, y_all, num_sectors)
    
    # Plot results
    fig = plot_comprehensive_results(trajectories_x, trajectories_y, 
                                   point_fractions, area_fractions)
    
    # Print speed analysis for first trajectory
    dt = tau/points_per_traj
    first_traj_speed = compute_speed(np.array(trajectories_x[0]), 
                                   np.array(trajectories_y[0]), dt)
    
    print("\nSpeed Analysis (First Trajectory):")
    print("-" * 50)
    print(f"Mean speed: {np.mean(first_traj_speed):.6f}")
    print(f"Std dev: {np.std(first_traj_speed):.6f}")
    print(f"Max difference: {np.max(first_traj_speed) - np.min(first_traj_speed):.6f}")
    
    print("\nAngular Coverage Analysis:")
    print("-" * 50)
    for i in range(len(point_fractions)):
        print(f"Sector {i}:")
        print(f"  Actual coverage: {point_fractions[i]:.3f}")
        print(f"  Theoretical coverage: {area_fractions[i]:.3f}")
        print(f"  Difference: {abs(point_fractions[i] - area_fractions[i]):.3f}")
    
    plt.show()

if __name__ == "__main__":
    main()
