#!/usr/bin/env python3

import sys
import visimerge as vm


def parse_viewray(l):
    v = list(map(vm.Real, l.split()))
    return vm.ViewRay(v[0], vm.Vec2([v[1], v[2]]), v[3], v[4])


def parse_visimerge(s):
    return list(map(parse_viewray, filter(lambda l: l, s.split('\n'))))


def main():
    if len(sys.argv) != 3:
        print('{0}: missing file operands'.format(sys.argv[0]), file=sys.stderr)
        exit(1)

    vm.Real = float

    with open(sys.argv[1]) as f:
        segments = vm.parse(f.read())

    xlim = 0
    ylim = 0

    for s in segments:
        xlim = max(xlim, abs(s[0][0]), abs(s[1][0]))
        ylim = max(ylim, abs(s[0][1]), abs(s[1][1]))

    xlim = float(xlim) * 1.05
    ylim = float(ylim) * 1.05

    with open(sys.argv[2]) as f:
        vis = parse_visimerge(f.read())

    vm.draw(segments, vis, xlim=xlim, ylim=ylim)


if __name__ == "__main__":
    main()
