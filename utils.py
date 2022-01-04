import taichi as ti

quality = 6
real = ti.f32
n_particles = 1 << (quality * 2 - 1)
#n_particles = 1
n_grid = 1 << (quality - 1)
dx = 1 / n_grid
dx_inv = 1 / dx
dt = 5e-5
n_subframe = int(1e-3 // dt)
rho_p = 1
vol_p = (dx * 0.5) ** 2
m_p = vol_p * rho_p
rigid_m_inv = 1 / (n_particles * m_p * 0.05)
E, nu = 5e3, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
x_mesh_0, y_mesh_0 = 0.3, 0.9
dx_mesh = dx
bound = 3
#gravity = [0, -9.8 * 10]
k_penetrate = 1
