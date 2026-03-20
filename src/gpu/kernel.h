#pragma once

#if defined(__SLANG__)
#include "gpu/kernel_slang.h"
#elif defined(__METAL_VERSION__)
#include "gpu/kernel_metal.h"
#else
#include "gpu/kernel_cpu.h"
#endif

#define DECL_KERNEL_1D(NAME, ARGS) DECL_KERNEL(uint, NAME, ARGS)
#define DECL_KERNEL_2D(NAME, ARGS) DECL_KERNEL(uint2, NAME, ARGS)
#define DECL_KERNEL_3D(NAME, ARGS) DECL_KERNEL(uint3, NAME, ARGS)