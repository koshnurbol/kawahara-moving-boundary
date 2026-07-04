"""Standard quartic (degree-4) B-spline on the reference interval [0,5]
and its first three derivatives, vectorised with numpy.

A quartic B-spline associated with the left knot s on a uniform grid of size h
is obtained as  N_s(x) = B4((x - s*h)/h),  and its d-th derivative as
[B4, dB4, ddB4, dddB4][d]((x - s*h)/h) / h**d.
"""
import numpy as np


def B4(z):
    z = np.asarray(z, float)
    c = [(z >= 0) & (z < 1), (z >= 1) & (z < 2), (z >= 2) & (z < 3),
         (z >= 3) & (z < 4), (z >= 4) & (z <= 5)]
    f = [z**4 / 24,
         (z**4 - 5*(z-1)**4) / 24,
         (z**4 - 5*(z-1)**4 + 10*(z-2)**4) / 24,
         ((5-z)**4 - 5*(4-z)**4) / 24,
         (5-z)**4 / 24]
    return np.select(c, f, 0.0)


def dB4(z):
    z = np.asarray(z, float)
    c = [(z >= 0) & (z < 1), (z >= 1) & (z < 2), (z >= 2) & (z < 3),
         (z >= 3) & (z < 4), (z >= 4) & (z <= 5)]
    f = [4*z**3 / 24,
         (4*z**3 - 20*(z-1)**3) / 24,
         (4*z**3 - 20*(z-1)**3 + 40*(z-2)**3) / 24,
         (-4*(5-z)**3 + 20*(4-z)**3) / 24,
         -4*(5-z)**3 / 24]
    return np.select(c, f, 0.0)


def ddB4(z):
    z = np.asarray(z, float)
    c = [(z >= 0) & (z < 1), (z >= 1) & (z < 2), (z >= 2) & (z < 3),
         (z >= 3) & (z < 4), (z >= 4) & (z <= 5)]
    f = [12*z**2 / 24,
         (12*z**2 - 60*(z-1)**2) / 24,
         (12*z**2 - 60*(z-1)**2 + 120*(z-2)**2) / 24,
         (12*(5-z)**2 - 60*(4-z)**2) / 24,
         12*(5-z)**2 / 24]
    return np.select(c, f, 0.0)


def dddB4(z):
    z = np.asarray(z, float)
    c = [(z >= 0) & (z < 1), (z >= 1) & (z < 2), (z >= 2) & (z < 3),
         (z >= 3) & (z < 4), (z >= 4) & (z <= 5)]
    f = [24*z / 24,
         (24*z - 120*(z-1)) / 24,
         (24*z - 120*(z-1) + 240*(z-2)) / 24,
         (-24*(5-z) + 120*(4-z)) / 24,
         -24*(5-z) / 24]
    return np.select(c, f, 0.0)
