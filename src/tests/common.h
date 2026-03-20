#pragma once

#if defined(__METAL_VERSION__)
#define MTL_CONSTANT constant
#define MTL_DEVICE device
#define MTL_KERNEL kernel
#include <metal_stdlib>
#else
#define MTL_CONSTANT const
#define MTL_DEVICE
#define MTL_KERNEL static inline
#include <cstdint>
typedef uint32_t uint;
struct uint2
{
	uint x;
	uint y;
};
#endif
