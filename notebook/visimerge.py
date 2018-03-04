import math
import sympy as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mplp
import matplotlib.collections as mplc

from stuff import *

# ----------------------------------------------------------------------------------------------------------------------

origin = Vec2([0, 0])

ViewRay = namedtuple('ViewRay', 'a v r l')


# ----------------------------------------------------------------------------------------------------------------------

def draw(segments=[], region=[], xlim=12, ylim=12, figsize=8):

    rlim = math.ceil(max(xlim, ylim) * math.sqrt(2.0))

    fig, ax = plt.subplots(figsize=(figsize, figsize))

    # Draw line segments

    lc = mplc.LineCollection(segments, linewidths=2)
    ax.add_collection(lc)

    # Draw triangles and wedges

    patches = []
    unbounded_regions = []

    iv = Vec2([1, 0])

    for i, vec in enumerate(region):

        pvec = region[i - 1] if i > 0 else ViewRay(0, iv, -1, -1)

        if vec.r < 0 and vec.v != pvec.v:
            unbounded_regions.append((float(pvec.a), float(vec.a)))

        if vec.r > 0 and pvec.l > 0:
            patches.append(mplp.Polygon([origin, pvec.v * pvec.l, vec.v * vec.r], True))

    if len(region) > 0 and region[-1].v != iv:
        unbounded_regions.append((float(region[-1].a), 0))

    for (t0, t1) in unbounded_regions:
        patches.append(mplp.Wedge([0, 0], rlim, t0 * 180 / math.pi, t1 * 180 / math.pi))

    pc = mplc.PatchCollection(patches, alpha=0.4)
    ax.add_collection(pc)

    # Draw origin

    ax.plot([origin], marker='o')

    # mpl stuff

    ax.margins(0.1)
    ax.set_aspect('equal', 'box')

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    plt.show(block=True)


# ----------------------------------------------------------------------------------------------------------------------

def viewrays_for_segment(p):
    a = p.a
    b = p.b

    if cross(a, b) < 0:
        a, b = b, a

    atan2b = atan2(b) if atan2(b) != 0 else 2 * sp.pi

    return [ViewRay(atan2(a), normalize(a), -1, norm(a)), ViewRay(atan2b, normalize(b), norm(b), -1)]


# ----------------------------------------------------------------------------------------------------------------------

def visible_region_for_segment(p):
    a = p.a
    b = p.b

    if cross(a, b) < 0:
        a, b = b, a

    s = b - a

    if s[1] == 0:
        return viewrays_for_segment(p)

    i = -a[1] / s[1]

    if 0 < i < 1 and a[0] + s[0] * i > 0:
        c = a + s * i

        return viewrays_for_segment(Segment(c, b)) + viewrays_for_segment(Segment(a, c))

    return viewrays_for_segment(p)


# ----------------------------------------------------------------------------------------------------------------------

def ray_segment_intersection(p, v):
    u = p.b - p.a

    w = p.a

    s = cross(u, v)

    if s == 0:
        return None

    r = w / s
    i = cross(v, r)

    if 0 <= i <= 1:
        return p[0] + u * i

    return None


def cut_segment(s, v, w):
    a = ray_segment_intersection(s, v)
    b = ray_segment_intersection(s, w)

    if a is None or b is None:
        return None

    return Segment(a, b)


# ----------------------------------------------------------------------------------------------------------------------

def merge(a, b):
    c = [a, b]
    n = [len(a), len(b)]
    i = [0, 0]

    s = []

    while i[0] < n[0] or i[1] < n[1]:

        if i[0] == n[0] or i[1] == n[1]:
            sel = 0 if i[0] < n[0] else 1
        else:
            sel = 0 if c[0][i[0]].a < c[1][i[1]].a else 1

        s.append(c[sel][i[sel]])

        if len(s) == 1:
            i[sel] = i[sel] + 1
            continue

        p = []

        for j in range(2):
            if 0 < i[j] < n[j] and c[j][i[j]].r > 0 and c[j][i[j] - 1].l > 0:
                v = c[j][i[j]]
                w = c[j][i[j] - 1]
                p.append(Segment(w.v * w.l, v.v * v.r))

        p = sorted(filter(None, [cut_segment(q, s[-2].v, s[-1].v) for q in p]), key=lambda q: tuple(map(norm, q)))

        if p:
            s[-2] = s[-2]._replace(l=norm(p[0][0]))
            s[-1] = s[-1]._replace(r=norm(p[0][1]))

        i[sel] = i[sel] + 1

    return s


# ----------------------------------------------------------------------------------------------------------------------

def visimerge(s):
    n = len(s)
    m = n // 2

    if n == 0:
        return []

    if n == 1:
        return visible_region_for_segment(s[0])

    s0 = visimerge(s[0:m])
    s1 = visimerge(s[m:n])

    return merge(s0, s1)


def visible_region(s):
    r = [p for p in s if not are_collinear(origin, *p)]
    return visimerge(r)


# ----------------------------------------------------------------------------------------------------------------------

