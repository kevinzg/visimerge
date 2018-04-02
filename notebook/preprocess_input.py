#!/usr/bin/env python3

import sys
import visimerge as vm


def main():
    if len(sys.argv) != 2:
        print('{0}: missing file operand'.format(sys.argv[0]), file=sys.stderr)
        exit(1)

    vm.Real = float

    with open(sys.argv[1]) as f:
        segments = vm.parse(f.read())

    xlim = 0
    ylim = 0

    count_collinear = 0
    count_splitted = 0

    for s in segments:
        if vm.are_collinear(vm.origin, *s):
            count_collinear += 1
            continue

        vrs = vm.visible_region_for_segment(s)

        for i in range(0, len(vrs), 2):
            a = map(float, vrs[i].v * max(vrs[i].l, vrs[i].r))
            b = map(float, vrs[i + 1].v * max(vrs[i + 1].l, vrs[i + 1].r))

            print('{0:.3f},{1:.3f} {2:.3f},{3:.3f}'.format(*a, *b))

        count_splitted += len(vrs) // 2 - 1

    print('{0} collinear segments removed. {1} segments splitted.'.format(count_collinear, count_splitted),
        file=sys.stderr)

if __name__ == '__main__':
    main()
