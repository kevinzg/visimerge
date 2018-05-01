#!/usr/bin/env python3

import os, sys, pathlib, subprocess

visimerge_dir = pathlib.Path(os.path.realpath(__file__)).parents[1]
benchmark_script = './scripts/run_benchmark_test.py'

K_BEGIN = 15
K_END = 23


def print_results(results):
    for i in range(0, K_END - K_BEGIN + 2):
        r = [r.split('\n')[i] for r in results]
        print(r[0].split(',')[0], end='')

        for item in r:
            print(',{0}'.format(item.split(',')[-1]), end='')
        print('')


def main():
    options = [
        ['-s'],
        ['-s', '-d'],
        [],
        ['-d']
    ]

    print("running all tests between 2**{0} and 2**{1} segments".format(K_BEGIN, K_END),
        file=sys.stderr)

    print("=" * 60, file=sys.stderr)

    results = []

    for opt in options:
        proc = subprocess.run([benchmark_script, str(K_BEGIN), str(K_END)] + opt, stdout=subprocess.PIPE,
            cwd=str(visimerge_dir))

        results.append(proc.stdout.decode('utf-8'))

    print_results(results)


if __name__ == '__main__':
    main()
