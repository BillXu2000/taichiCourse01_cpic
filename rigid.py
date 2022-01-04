import taichi as ti

from utils import *

@ti.data_oriented
class Rigid:
    def __init__(self, n):
        self.pos = ti.Vector.field(2, dtype = real, shape = ())
        self.v = ti.Vector.field(2, dtype = real, shape = ())
        self.theta = ti.field(dtype = real, shape = ())
        self.w = ti.field(dtype = real, shape = ())
        self.o_r = ti.Vector.field(2, dtype = real, shape = n)
        self.p_r = ti.Vector.field(2, dtype = real, shape = n)
        self.o = ti.Vector.field(2, dtype = real, shape = n)
        self.p = ti.Vector.field(2, dtype = real, shape = n)
        self.n = n

    def init(self, draw, theta, pos, w, v, fixed = False):
        self.theta[None] = theta
        self.pos[None] = pos
        self.w[None] = w
        self.v[None] = v
        self.fixed = fixed
        self.draw = draw
        self.advect()

    @ti.kernel
    def set_line(self, a: ti.template(), b: ti.template()):
        o = ti.Vector(a) # TODO: Atomic add (float32 to int32) may lose precision.
        p = (ti.Vector(b) - o) / self.n
        for i in self.o:
            self.o_r[i] = o
            self.p_r[i] = p
            o += p

    @ti.kernel
    def advect(self):
        self.pos[None] += self.v[None] * dt
        self.theta[None] += self.w[None] * dt
        theta = self.theta[None]
        rotate = ti.Matrix([[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]])
        for i in self.o:
            self.o[i] = rotate @ self.o_r[i] + self.pos[None]
            self.p[i] = rotate @ self.p_r[i]

    def draw_line(self, gui):
        gui.line(self.o[0], [self.o[self.n - 1].x + self.p[self.n - 1].x, self.o[self.n - 1].y + self.p[self.n - 1].y])
