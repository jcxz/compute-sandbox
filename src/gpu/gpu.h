#pragma once

#include "core/reflection.h"

#include <string>


enum AllocationMode
{
	Shared = (1 << 0),
	Device = (1 << 1),
};

enum class GpuAdapterType
{
	Default = (1 << 0),
	Metal   = (1 << 1),
	Vulkan  = (1 << 2),
};

// GPU initialization and termination
extern bool InitializeGpu(const GpuAdapterType type = GpuAdapterType::Default);
extern void TerminateGpu();

//! Allocates GPU memory of a given size
//! optional flags specify whether the allocation is exclusive to GPU and not visible to CPU
//! or if the allocation is Shared, i.e. visible to CPU and can be mapped for reading/writing by CPU.
//! This is the default mode
extern void* GpuAlloc(const size_t size, const AllocationMode mode = AllocationMode::Shared);

//! deallocates the memory allocated with GpuAlloc
extern void GpuFree(void* const ptr);

//! registers a kernel function, so that ExecuteGPUKernel can execute it
extern uint32_t RegisterKernel(const std::string& name);

//! executes a kernel on the GPU with given arguments
//! \param id the id of the kernel to execute (this must have been registered before hand), otherwise the call will fail
//! \param nx specifies the execution domain (the number of work items) in the x direction
//! \param pArgs arguments to be passed to a kernel (this is a struct declared via reflection macros, i.e REFL_DECL_STRUCT_*)
//! \param pArgsInfo meta information about the args structure, so that the function knows how to pass it to the GPU)
extern bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo);

//! executes a kernel on the GPU with given arguments
//! \param id the id of the kernel to execute (this must have been registered before hand), otherwise the call will fail
//! \param nx specifies the execution domain (the number of work items) in the x direction
//! \param ny specifies the execution domain (the number of work items) in the y direction
//! \param pArgs arguments to be passed to a kernel (this is a struct declared via reflection macros, i.e REFL_DECL_STRUCT_*)
//! \param pArgsInfo meta information about the args structure, so that the function knows how to pass it to the GPU)
extern bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo);

//! executes a kernel on the GPU with given arguments
//! \param id the id of the kernel to execute (this must have been registered before hand), otherwise the call will fail
//! \param nx specifies the execution domain (the number of work items) in the x direction
//! \param ny specifies the execution domain (the number of work items) in the y direction
//! \param nz specifies the execution domain (the number of work items) in the z direction
//! \param pArgs arguments to be passed to a kernel (this is a struct declared via reflection macros, i.e REFL_DECL_STRUCT_*)
//! \param pArgsInfo meta information about the args structure, so that the function knows how to pass it to the GPU)
extern bool ExecuteGPUKernel(
	const uint32_t id,
	const uint32_t nx,
	const uint32_t ny,
	const uint32_t nz,
	const void* const pArgs,
	const refl::TypeMetaInfo* const pArgsInfo);
