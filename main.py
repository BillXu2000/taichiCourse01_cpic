import taichi as ti
import time

ti.init(debug = False, arch = ti.cpu)

from utils import *
from rigid import *

rigid = [Rigid(200)]
material = 'jelly'
boundary_condition = 'separate'

Jp = ti.field(dtype = real, shape = n_particles)
a = ti.field(dtype = ti.i32, shape = n_particles)
t = ti.field(dtype = ti.i32, shape = n_particles)
x = ti.Vector.field(2, dtype = real, shape = n_particles)
v = ti.Vector.field(2, dtype = real, shape = n_particles)
dist = ti.Vector.field(2, dtype = real, shape = n_particles)
norm = ti.Vector.field(2, dtype = real, shape = n_particles)
v_penetrate = ti.Vector.field(2, dtype = real, shape = n_particles)
C = ti.Matrix.field(2, 2, dtype = real, shape = n_particles)
F = ti.Matrix.field(2, 2, dtype = real, shape = n_particles)
m_grid = ti.field(dtype = real, shape = (n_grid, n_grid))
a_grid = ti.field(dtype = ti.i32, shape = (n_grid, n_grid))
t_grid = ti.field(dtype = ti.i32, shape = (n_grid, n_grid))
v_grid = ti.Vector.field(2, dtype = real, shape = (n_grid, n_grid))
delta_w = ti.field(dtype = real, shape = ())
gravity = ti.Vector.field(2, dtype = real, shape = ())

@ti.kernel
def reset_k():
    gravity[None] = ti.Vector([0, -9.8 * 10])
    for p in x:
        x[p] = [ti.random() * 0.4 + 0.3, ti.random() * 0.4 + 0.3]
        v[p] = ti.Vector.zero(real, 2)
        F[p] = ti.Matrix.identity(real, 2)
        C[p] = ti.Matrix.zero(real, 2, 2)
        Jp[p] = 1

def reset():
    global rigid
    #rigid[0].set_line((-0.9, 0), (0.9, 0))
    rigid[0].set_line((0, -0.5), (0, 0.5))
    #rigid[0].init(rigid[0].draw_line, 0, [0.5, 0.5], -10, [0, 0])
    rigid[0].init(rigid[0].draw_line, 0, [0.5, 1.2], 0, [0, 0])
    a.fill(0)
    t.fill(0)
    reset_k()

@ti.kernel
def surface_2_grid(num: ti.i32, x_s: ti.template(), p_s: ti.template()):
    for p in x_s:
        x_tmp = x_s[p] + 0.5 * p_s[p]
        base = (x_tmp * dx_inv - 0.5).cast(int)
        local = x_tmp * dx_inv - base
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - local) * dx
            x_prime = dpos.dot(p_s[p].normalized())
            if -0.5 * p_s[p].norm() <= x_prime < 0.5 * p_s[p].norm():
                if p_s[p].cross(dpos) < 0: t_grid[base + offset] |= 2 ** num
                a_grid[base + offset] |= 2 ** num

def grid_cdf():
    a_grid.fill(0)
    t_grid.fill(0)
    for i, r in enumerate(rigid):
        surface_2_grid(i, r.o, r.p)

@ti.kernel
def surface_2_particle(num: ti.i32, x_s_: ti.template(), p_s_: ti.template(), n_s: ti.i32): # only work for straight lines
    x_s = x_s_[0]
    p_s = p_s_[0] * n_s
    for p in x:
        dpos = x[p] - x_s
        x_prime = dpos.dot(p_s.normalized())
        dist[p] = dpos - x_prime * p_s.normalized()
        norm[p] = dist[p].normalized()
        #if 0 <= x_prime < p_s.norm() and dist[p].norm() < dx:
        if 0 <= x_prime < p_s.norm():
            if not (a[p] & (2 ** num)):
                a[p] |= 2 ** num
                if p_s.cross(dpos) < 0: 
                    t[p] |= 2 ** num
                    norm[p] = ti.Matrix([[0, 1], [-1, 0]]) @ p_s.normalized()
                else: 
                    t[p] &= ~(2 ** num)
                    norm[p] = ti.Matrix([[0, -1], [1, 0]]) @ p_s.normalized()
            elif (p_s.cross(dpos) < 0) ^ ((t[p] & (2 ** num)) > 0): v_penetrate[p] -= dist[p] * dt * k_penetrate
        else: a[p] &= ~(2 ** num)

def particle_cdf():
    for i, r in enumerate(rigid):
        surface_2_particle(i, r.o, r.p, r.n)

@ti.kernel
def p2g():
    for I in ti.grouped(v_grid):
        v_grid[I] = ti.Vector.zero(real, 2)
        m_grid[I] = 0
    for p in x:
        base = (x[p] * dx_inv - 0.5).cast(int)
        local = x[p] * dx_inv - base
        w = [0.5 * (1.5 - local) ** 2, 0.75 - (local - 1) ** 2, 0.5 * (local - 0.5) ** 2]
        affine = m_p * C[p]
        F[p] = (ti.Matrix.identity(real, 2) + dt * C[p]) @ F[p]
        U, sig, V = ti.svd(F[p])
        if ti.static(material == 'water'):
            J = sig[0, 0] * sig[1, 1]
            F[p] = ti.Matrix.identity(real, 2) * ti.sqrt(J)
            affine += -4 * dt * (dx_inv ** 2) * vol_p * E * (J - 1) * ti.Matrix.identity(real, 2)
        if ti.static(material == 'jelly'):
            J = sig[0, 0] * sig[1, 1]
            h = 0.3
            mu, la = mu_0 * h, lambda_0 * h
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + la * (J - 1) * J * ti.Matrix.identity(real, 2)
            affine += -4 * dt * (dx_inv ** 2) * vol_p * stress
        if ti.static(material == 'snow'):
            h = max(0.1, min(5, ti.exp(10 * (1.0 - Jp[p]))))  # Hardening coefficient: snow gets harder when compressed
            mu, la = mu_0 * h, lambda_0 * h
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = min(max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + la * (J - 1) * J * ti.Matrix.identity(real, 2)
            affine += -4 * dt * (dx_inv ** 2) * vol_p * stress
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            k = w[i].x * w[j].y
            if a_grid[base + offset] & a[p] & (t_grid[base + offset] ^ t[p]):
                v_r = rigid[0].v[None] + rigid[0].w[None] * ti.Matrix([[0, -1], [1, 0]]) @ ((base + offset + 1).cast(real) * dx - rigid[0].pos[None]) # TODO: rigid number hack
                v_tmp = v[p] - v_r
                if ti.static(boundary_condition == 'sticky'): v_tmp = v_r
                if ti.static(boundary_condition == 'slip'): v_tmp += v_r - norm[p] * v_tmp.dot(norm[p])
                if ti.static(boundary_condition == 'separate'):
                    v_dist = v_tmp.dot(norm[p])
                    v_tmp += v_r
                    if v_dist < 0: v_tmp -= norm[p] * v_dist
                impulse = k * m_p * (v[p] - v_tmp)
                delta_w[None] += impulse.dot(ti.Matrix([[0, -1], [1, 0]]) @ ((base + offset + 1).cast(real) * dx - rigid[0].pos[None])) * rigid_m_inv # TODO: rigid impulse hack
                #print(v_r, v[p], v_tmp, impulse * rigid_m_inv)
            else:
                dpos = (offset.cast(float) - local) * dx
                v_grid[base + offset] += k * (m_p * v[p] + affine @ dpos)
                m_grid[base + offset] += k * m_p

@ti.kernel
def grid_op():
    for i, j in v_grid:
        I = ti.Vector([i, j])
        if m_grid[I] <= 0:
            continue
        v_grid[I] = v_grid[I] / m_grid[I] + gravity[None] * dt
        if ti.static(boundary_condition == 'sticky'):
            if i < bound or i >= n_grid - bound or j < bound or j >= n_grid - bound: v_grid[I] = [0, 0]
        if ti.static(boundary_condition == 'slip'):
            if i < bound or i >= n_grid - bound: v_grid[I].x = 0
            if j < bound or j >= n_grid - bound: v_grid[I].y = 0
        if ti.static(boundary_condition == 'separate'):
            if i < bound and v_grid[I].x < 0: v_grid[I].x = 0
            if i >= n_grid - bound and v_grid[I].x > 0: v_grid[I].x = 0
            if j < bound and v_grid[I].y < 0: v_grid[I].y = 0
            if j >= n_grid - bound and v_grid[I].y > 0: v_grid[I].y = 0

@ti.kernel
def g2p():
    for p in x:
        base = (x[p] * dx_inv - 0.5).cast(int)
        local = x[p] * dx_inv - base
        w = [0.5 * (1.5 - local) ** 2, 0.75 - (local - 1) ** 2, 0.5 * (local - 0.5) ** 2]
        #new_v = v_penetrate[p]
        new_v = ti.Vector.zero(real, 2)
        v_penetrate[p] = ti.Vector.zero(real, 2)
        new_C = ti.Matrix.zero(real, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            k = w[i].x * w[j].y
            dpos = offset.cast(float) - local # actually dpos / dx
            v_tmp = v_grid[base + offset]
            if a_grid[base + offset] & a[p] & (t_grid[base + offset] ^ t[p]):
                v_r = rigid[0].v[None] + ti.Matrix([[0, -1], [1, 0]]) @ (rigid[0].w[None] * ((base + offset + 1).cast(real) * dx - rigid[0].pos[None])) # TODO: rigid number hack
                v_tmp = v[p] - v_r
                if ti.static(boundary_condition == 'sticky'): v_tmp = v_r
                if ti.static(boundary_condition == 'slip'): v_tmp += v_r - norm[p] * v_tmp.dot(norm[p])
                if ti.static(boundary_condition == 'separate'):
                    v_dist = v_tmp.dot(norm[p])
                    v_tmp += v_r
                    if v_dist < 0: v_tmp -= norm[p] * v_dist
                #v_tmp += k_penetrate * norm[p]
            new_v += k * v_tmp
            new_C += k * 4 * dx_inv * v_tmp.outer_product(dpos)
        x[p] += dt * new_v
        v[p] = new_v
        C[p] = new_C
    #rigid[0].w[None] += delta_w[None]
    delta_w[None] = 0

def substep():
    grid_cdf()
    particle_cdf()
    p2g()
    grid_op()
    g2p()
    for r in rigid:
        r.advect()

gui = ti.GUI('Windmill Charge', res = (512, 512))
reset()
pause = False
start = time.perf_counter()
frame = 0
reset()
while gui.running:
    user_v = 1.0
    for e in gui.get_events(gui.PRESS):
        if e.key in [gui.ESCAPE, gui.EXIT, 'q']: gui.running = False
        elif e.key == 'r': reset()
        elif e.key == ' ': pause = not pause
        elif e.key == 'g': substep()
    if gui.is_pressed('w'): rigid[0].pos[None][1] += user_v / 60
    if gui.is_pressed('s'): rigid[0].pos[None][1] -= user_v / 60
    rigid[0].v[None][0] = 0
    if gui.is_pressed('d'): rigid[0].v[None][0] = user_v * 8
    if gui.is_pressed('a'): rigid[0].v[None][0] = -user_v * 8
    if gui.is_pressed(gui.SHIFT): rigid[0].v[None][0] *= 0.5
    if not pause:
        for subframe in range(n_subframe):
            substep()
    for r in rigid:
        r.draw(gui)
    gui.circles(x.to_numpy(), color = 0x068587, radius=2)
    gui.show()
    frame += 1
