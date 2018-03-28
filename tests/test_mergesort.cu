#include <vector>
#include <algorithm>
#include <cstdio>
#include <moderngpu/kernel_mergesort.hxx>

int main(int argc, char** argv)
{
    mgpu::standard_context_t context;

    int count = 1000000;

    mgpu::mem_t<int> data = mgpu::fill_random(0, 100000, count, false, context);

    mgpu::mergesort(data.data(), count, mgpu::less_t<int>(), context);

    std::vector<int> ref = mgpu::from_mem(data);
    std::sort(ref.begin(), ref.end());
    std::vector<int> sorted = mgpu::from_mem(data);

    bool success = ref == sorted;

    printf("%7d: %s\n", count, success ? "SUCCESS" : "FAILURE");

    if(!success)
        return 1;

    return 0;
}
