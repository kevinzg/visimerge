#!/usr/bin/env python3

import sys
import itertools


def main():
    if len(sys.argv) < 2:
        print('{0}: missing k parameter'.format(sys.argv[0]), file=sys.stderr)
        exit(1)

    k = int(sys.argv[1])

    if k >= 24:
        print('{0}: k is too big'.format(sys.argv[0]), file=sys.stderr)
        exit(1)

    n = 2 ** k
    s = n // 4

    if s * 4 != n:
        s += 1

    m = 0

    for c, i in itertools.product(range(4), range(s)):
        if m >= n:
            break

        a = (1 + i, s - i)
        b = (1 + s, s - i)

        mx = -1 if c & 0b01 else 1
        my = -1 if c & 0b10 else 1

        print('{0:.3f},{1:.3f} {2:.3f},{3:.3f}'.format(mx * a[0], my * a[1], mx * b[0], my * b[1]))

        m += 1


if __name__ == "__main__":
    main()
