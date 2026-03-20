#pragma once

#include "core/preprocessor.h"

#include <metal_stdlib>

// metal has its own standard library prefixed with metal namespace
// This makes an alias for it, so that we can compile regular c++ code with less changes
namespace std = metal;

#define MTL_CONSTANT constant
#define MTL_DEVICE device
#define DECL_KERNEL(ITYPE, NAME, ARGS) \
	kernel void NAME(const ITYPE index [[thread_position_in_grid]], constant ARGS& args [[buffer(0)]])

#define DECL_KERNEL_ARGS_BEGIN(NAME) struct NAME {
#define DECL_KERNEL_ARGS_FIELD_(TYPE, NAME, ARRAY) TYPE NAME ARRAY;
#define DECL_KERNEL_ARGS_FIELD2(TYPE, NAME) DECL_KERNEL_ARGS_FIELD_(TYPE, NAME, )
#define DECL_KERNEL_ARGS_FIELD3(TYPE, NAME, ARRAY) DECL_KERNEL_ARGS_FIELD_(TYPE, NAME, ARRAY)
#define DECL_KERNEL_ARGS_FIELD(...) EM_CONCAT(DECL_KERNEL_ARGS_FIELD, EM_NARGS(__VA_ARGS__))(__VA_ARGS__)
#define DECL_KERNEL_ARGS_END(NAME) };