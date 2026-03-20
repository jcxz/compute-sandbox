#include "tests/common.h"


MTL_KERNEL void invert_kernel2(
	const uint index [[thread_position_in_grid]],
	uint8_t MTL_DEVICE * data [[buffer(0)]])
{
	data[index] = 255 - data[index];
}
