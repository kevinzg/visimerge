#!/usr/bin/env bash

function print_equals {
    printf '=%.0s' {1..120}
    printf '\n'
}

function run_kernel_test {
    ./bin/test_kernel_visimergesort /tmp/gen_$1_input --profile > /dev/null
    print_equals
}

function test_kernel {
    if [ ! -f /tmp/gen_$1_input ]; then
        ./notebook/generator.py $1 > /tmp/gen_$1_input
    fi

    echo "    nvprof ./bin/test_kernel_visimergesort /tmp/gen_$1_input --profile > /dev/null"

    print_equals

    nvprof ./bin/test_kernel_visimergesort /tmp/gen_$1_input --profile > /dev/null

    print_equals

    echo "    ./bin/test_kernel_visimergesort /tmp/gen_$1_input --profile > /dev/null"

    print_equals

    run_kernel_test $1
    run_kernel_test $1
    run_kernel_test $1
}

test_kernel $1
