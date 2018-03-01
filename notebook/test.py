#!/usr/bin/env python3

import sys
import visimerge as vm


def main():
    if len(sys.argv) != 2:
        print('{0}: missing file operand'.format(sys.argv[0]), file=sys.stderr)
        exit(1)

    with open(sys.argv[1]) as f:
        segments = vm.parse(f.read())

    xlim = 0
    ylim = 0

    for s in segments:
        xlim = max(xlim, abs(s[0][0]), abs(s[1][0]))
        ylim = max(ylim, abs(s[0][1]), abs(s[1][1]))

    xlim = float(xlim) * 1.05
    ylim = float(ylim) * 1.05

    vis = vm.visible_region(segments)

    vm.draw(segments, vis, xlim=xlim, ylim=ylim)


if __name__ == "__main__":
    main()
