#include "core/kernel.h"

DECL_KERNEL_ARGS_BEGIN(Args)
	DECL_KERNEL_ARGS_FIELD(uint32_t, dstStride)
	DECL_KERNEL_ARGS_FIELD(uint32_t, srcStride)
	DECL_KERNEL_ARGS_FIELD(uint8_t MTL_DEVICE *, dst)
	DECL_KERNEL_ARGS_FIELD(uint8_t MTL_CONSTANT *, src)
DECL_KERNEL_ARGS_END(Args)

DECL_KERNEL_2D(Invert, Args)
{
	uint8_t MTL_DEVICE * dst = args.dst + index.y * args.dstStride + index.x;
	uint8_t MTL_CONSTANT * src = args.src + index.y * args.srcStride + index.x;
	*dst = 255 - *src;
}
