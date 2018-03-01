import sympy as sp
from collections import namedtuple

Vec2 = sp.Matrix
Segment = namedtuple('Segment', 'a b')
Real = sp.Rational


def norm(v):
    return sp.sqrt(v[0] ** 2 + v[1] ** 2)


def normalize(v):
    return v / norm(v)


def atan2(v):
    a = sp.atan2(v[1], v[0])
    return a if a >= 0 else a + 2 * sp.pi


def cross(v, w):
    return v[0] * w[1] - v[1] * w[0]


def are_collinear(p, q, r):
    v = q - p
    w = r - p
    return cross(v, w) == 0


def parse(s):
    points = [Vec2(list(map(Real, l.split(',')))) for l in s.split()]
    return [Segment(points[i], points[i + 1]) for i in range(0, len(points), 2)]
