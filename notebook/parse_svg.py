#!/usr/bin/env python3

import sys
import xml.etree.ElementTree as ET

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('{0}: missing file operand'.format(sys.argv[0]))
        exit(1)

    filename = sys.argv[1]

    it = ET.iterparse(filename)
    for _, el in it:
        if '}' in el.tag:
            el.tag = el.tag.split('}', 1)[1]

    g = it.root.find('g')

    segments = []
    origin = (0, 0)

    for e in g:
        if e.tag == 'path':
            d = e.attrib['d'].split()

            if len(d) != 4 or d[0] != 'M' or d[2] != 'L':
                print('error: please use paths with absolute coordinates e.g. "M 0,10 L 10,20"\n(id: {0})'.format(e.attrib['id']),
                      file=sys.stderr)
                exit(1)

            points = [tuple(map(float, p.split(','))) for p in d[1::2]]
            segments.append(points)

        elif e.tag == 'circle':
            origin = (float(e.attrib['cx']), float(e.attrib['cy']))

    for s in segments:
        print('{0:.3f},{1:.3f} {2:.3f},{3:.3f}'.format(s[0][0] - origin[0], -(s[0][1] - origin[1]),
                                                       s[1][0] - origin[0], -(s[1][1] - origin[1])))
