#include "tests/common.h"


struct InvertArgs
{
	uint dstStride;
	uint srcStride;
	uint8_t MTL_DEVICE * dst [[buffer(0)]];
	uint8_t MTL_CONSTANT * src [[buffer(1)]];
};

MTL_KERNEL void invert_kernel(
	const uint2 index [[thread_position_in_grid]],
	InvertArgs MTL_CONSTANT & args [[buffer(0)]])
{
	uint8_t MTL_DEVICE * dst = args.dst + index.y * args.dstStride + index.x;
	uint8_t MTL_CONSTANT * src = args.src + index.y * args.srcStride + index.x;
	*dst = 255 - *src;
}
