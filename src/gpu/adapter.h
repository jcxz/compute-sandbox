#pragma once

#include "gpu/gpu.h"


namespace gpu
{

class IAdapter
{
public:
	virtual ~IAdapter() = default;

	virtual void* Alloc(const size_t size, const AllocationMode mode) = 0;
	virtual void Free(void* const ptr) = 0;

	virtual bool ExecuteKernel(
		const uint32_t id,
		const uint32_t nx,
		const uint32_t ny,
		const uint32_t nz,
		const void* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo) = 0;
};

} // End of namespace gpu