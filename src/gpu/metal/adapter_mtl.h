#include "gpu/gpu.h"
#include "gpu/adapter.h"
#include "gpu/metal/mtl.h"

#include <memory>
#include <unordered_map>



namespace gpu
{

class AdapterMtl final : public IAdapter
{
	static_assert(sizeof(MTL::GPUAddress) == sizeof(uint64_t), "MTL::GPUAddress not the same size as uint64_t");
	static_assert(sizeof(void*) == sizeof(uint64_t), "void* not the same size as uint64_t");

private:
	struct KernelInfo
	{
		NS::SharedPtr<MTL::Function> pFunc;
		NS::SharedPtr<MTL::ComputePipelineState> pPSO;
	};

public:
	virtual void* Alloc(const size_t size, const AllocationMode mode) override final;

	virtual void Free(void* const ptr) override final;

	virtual bool ExecuteKernel(
		const uint32_t id,
		const uint32_t nx,
		const uint32_t ny,
		const uint32_t nz,
		const void* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo) override final;

	static AdapterMtl* CreateMetalAdapter(const uint32_t flags = 0)
	{
		std::unique_ptr<AdapterMtl> pAdapter(new AdapterMtl);
		return pAdapter->Init(flags) ? pAdapter.release() : nullptr;
	}

private:
	bool IsInitialized() const;
	bool Init(const uint32_t flags);

	const KernelInfo* RequestKernel(const uint32_t id);

	bool EncodeKernelArguments(
		MTL::ComputeCommandEncoder* pEncoder,
		const KernelInfo* pKernel,
		const uint8_t* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo);

	bool EncodeKernelArguments(
		MTL::ComputeCommandEncoder* pEncoder,
		MTL::ArgumentEncoder* pArgEncoder,
		const uint8_t* const pArgs,
		const refl::TypeMetaInfo* const pArgsInfo);

	MTL::Buffer* GetAllocationBuffer(const void* const ptr) const;

	MTL::Buffer* GetAllocationBuffer(const uint64_t ptr) const;

private:
	AdapterMtl() = default;
	AdapterMtl(AdapterMtl&& ) = delete;
	AdapterMtl(const AdapterMtl& ) = delete;
	AdapterMtl& operator=(AdapterMtl&& ) = delete;
	AdapterMtl& operator=(const AdapterMtl& ) = delete;

private:
	//! a global autorelease pool (just in case there is no other autorelease pool this one will make sure that no temporary ObjectiveC
	//! objects created by the metal adapter will leak)
	NS::SharedPtr<NS::AutoreleasePool> mpAutoReleasePool;
	//! the GPU device that will be used to execute kernels
	NS::SharedPtr<MTL::Device> mpDevice;
	//! shader library with the compile kernels
	// (allows us on demand compute pipeline creation)
	NS::SharedPtr<MTL::Library> mpLibrary;
	//! a command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> mpCommandQueue;
	//! a cache of precompiled pipelines ready to be used for starting a kernel
	std::vector<KernelInfo> mKernels;
	//! argument buffer allocator (here we will store the arguments that we pass to kernels)
	//ArgumentBufferAllocator mAllocator; // ... TODO
	//! Let's keep it simple for now, stores references to allocated GPU buffers
	std::unordered_map<uint64_t, NS::SharedPtr<MTL::Buffer>> mAllocations;
};

extern IAdapter* CreateMetalAdapter(const uint32_t flags = 0);

} // End of namespace gpu