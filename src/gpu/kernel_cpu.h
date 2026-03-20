#pragma once

#include "core/reflection.h"
#include "gpu/gpu.h"

struct uint2
{
	uint32_t x;
	uint32_t y;
};

struct uint3
{
	uint32_t x;
	uint32_t y;
	uint32_t z;
};

#define MTL_CONSTANT const
#define MTL_DEVICE
#define DECL_KERNEL(ITYPE, NAME, ARGS) \
	struct NAME \
	{ \
		typedef ARGS ArgsType; \
		typedef ITYPE IndexType; \
		static inline const std::string kName = #NAME; \
		static inline const uint32_t kID = RegisterKernel(#NAME); \
		static inline void Run(const IndexType index, const ArgsType& args); \
	}; \
	inline void NAME::Run(const IndexType index, const ArgsType& args)

#define DECL_KERNEL_ARGS_BEGIN(NAME) REFL_DECL_STRUCT_BEGIN(NAME)
#define DECL_KERNEL_ARGS_FIELD(...) REFL_DECL_STRUCT_FIELD(__VA_ARGS__)
#define DECL_KERNEL_ARGS_END(NAME) REFL_DECL_STRUCT_END(NAME)

// GPU Kernel Execution /////////////////////////////////////////////////////////////////////////////

template <typename K>
bool ExecuteGPUKernel(const uint32_t n, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint32_t>, "Mismatching kernel dimensionality");
	return ExecuteGPUKernel(K::kID, n, static_cast<const void*>(&args), K::ArgsType::kMetaInfo);
}

template <typename K>
bool ExecuteGPUKernel(const uint32_t nx, const uint32_t ny, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint2>, "Mismatching kernel dimensionality");
	return ExecuteGPUKernel(K::kID, nx, ny, static_cast<const void*>(&args), K::ArgsType::kMetaInfo);
}

template <typename K>
bool ExecuteGPUKernel(const uint32_t nx, const uint32_t ny, const uint32_t nz, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint3>, "Mismatching kernel dimensionality");
	return ExecuteGPUKernel(K::kID, nx, ny, nz, static_cast<const void*>(&args), K::ArgsType::kMetaInfo);
}

// CPU Kernel Execution /////////////////////////////////////////////////////////////////////////////

template <typename K>
void ExecuteCPUKernel(const uint32_t n, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint32_t>, "Mismatching kernel dimensionality");
	for (uint32_t i = 0; i < n; ++i)
	{
		K::Run(i, args);
	}
}

template <typename K>
void ExecuteCPUKernel(const uint32_t nx, const uint32_t ny, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint2>, "Mismatching kernel dimensionality");
	for (uint32_t y = 0; y < ny; ++y)
	{
		for (uint32_t x = 0; x < nx; ++x)
		{
			K::Run({ x, y }, args);
		}
	}
}

template <typename K>
void ExecuteCPUKernel(const uint32_t nx, const uint32_t ny, const uint32_t nz, const typename K::ArgsType& args)
{
	static_assert(std::is_same_v<typename K::IndexType, uint3>, "Mismatching kernel dimensionality");
	for (uint32_t z = 0; z < nz; ++z)
	{
		for (uint32_t y = 0; y < ny; ++y)
		{
			for (uint32_t x = 0; x < nx; ++x)
			{
				K::Run({ x, y, z }, args);
			}
		}
	}
}

// Any Kernel Execution /////////////////////////////////////////////////////////////////////////////

template <typename K>
void ExecuteKernel(const uint32_t n, const typename K::ArgsType& args, const bool useCPU = false)
{
	if (useCPU || !ExecuteGPUKernel<K>(n, args))
		ExecuteCPUKernel<K>(n, args);
}

template <typename K>
void ExecuteKernel(const uint32_t nx, const uint32_t ny, const typename K::ArgsType& args, const bool useCPU = false)
{
	if (useCPU || !ExecuteGPUKernel<K>(nx, ny, args))
		ExecuteCPUKernel<K>(nx, ny, args);
}

template <typename K>
void ExecuteKernel(const uint32_t nx, const uint32_t ny, const uint32_t nz, const typename K::ArgsType& args, const bool useCPU = false)
{
	if (useCPU || !ExecuteGPUKernel<K>(nx, ny, nz, args))
		ExecuteCPUKernel<K>(nx, ny, nz, args);
}