#include "tests/common.h"

MTL_KERNEL void add_kernel(
    uint index [[thread_position_in_grid]],
    float MTL_CONSTANT * buf1 [[buffer(0)]],
    float MTL_CONSTANT * buf2 [[buffer(1)]],
    float MTL_DEVICE * result [[buffer(2)]])
{
    result[index] = buf1[index] + buf2[index];
}
