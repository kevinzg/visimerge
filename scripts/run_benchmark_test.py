#!/usr/bin/env python3

import os, sys, argparse, pathlib, subprocess

MAX_K = 23

visimerge_dir = pathlib.Path(os.path.realpath(__file__)).parents[1]
cuda_exec = './bin/test_kernel_visimergesort'
cpu_exec = './bin/test_serial_visimergesort'
generator_exec = './scripts/generator.py'
input_file_template = '/tmp/gen_{0}_input'


def parse_output(output):
    output = [l for l in output.decode('utf-8').split('\n') if l]
    time = [t for t in output[-1].split() if t and t.endswith('ms')]
    time = time[0][0:-2]

    return float(time)


def print_results(results, time_col_name='time'):
    print('k,{0}'.format(time_col_name))
    for k, time in sorted(results.items()):
        print('{0},{1:0.5f}'.format(k, results[k]))


def main(args):
    a = min(args.a, MAX_K)
    b = a if args.b is None else min(args.b, MAX_K)

    a, b = min(a, b), max(a, b)

    processor = 'cpu' if args.serial else 'gpu'

    results = {}

    for k in range(a, b + 1):
        input_file = pathlib.Path(input_file_template.format(k))
        if not input_file.is_file():
            print("generating input file with 2**{0} segments".format(k),
                file=sys.stderr)

            input_file.touch()
            with input_file.open(mode='w') as f:
                subprocess.run([generator_exec, str(k)], stdout=f, cwd=str(visimerge_dir))

        print("computing visibility region of 2**{0} segments using {1}".format(k, processor),
            file=sys.stderr)

        data_type = '--double' if args.double else '--float'
        binary_exec = cpu_exec if args.serial else cuda_exec

        proc = subprocess.run([binary_exec, str(input_file), '--profile', data_type], stderr=subprocess.PIPE,
            cwd=str(visimerge_dir))

        results[k] = parse_output(proc.stderr)

        print("\t{0:.5f}ms".format(results[k]), file=sys.stderr)

    print('-' * 60, file=sys.stderr)
    print_results(results, '{0}_{1}'.format(processor, data_type[2:]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('a', type=int, help="range begin")
    parser.add_argument('b', type=int, help="range end", nargs='?')
    parser.add_argument('-d', '--double', action='store_true', help="use double instead of floats")
    parser.add_argument('-s', '--serial', action='store_true', help="use cpu instead of gpu")

    args = parser.parse_args()

    main(args)
