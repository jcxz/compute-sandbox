#pragma once

#include <string>
#include <vector>
#include <cstdint>


class KernelRegistry
{
public:
	uint32_t GetKernelCount() const
	{
		return static_cast<uint32_t>(mKernels.size());
	}

	bool IsValidKernelID(const uint32_t id) const
	{
		return id < mKernels.size();
	}

	const std::string& GetKernelName(const uint32_t id) const
	{
		static const std::string kEmpty;
		return id < mKernels.size() ? mKernels[id] : kEmpty;
	}

	uint32_t New(const std::string& name)
	{
		const uint32_t id = static_cast<uint32_t>(mKernels.size());
		mKernels.emplace_back(name);
		return id;
	}

	static KernelRegistry* GetInstance()
	{
		static KernelRegistry registry;
		return &registry;
	}

private:
	KernelRegistry() = default;
	KernelRegistry(KernelRegistry&& ) = delete;
	KernelRegistry(const KernelRegistry& ) = delete;
	KernelRegistry& operator=(KernelRegistry&& ) = delete;
	KernelRegistry& operator=(const KernelRegistry& ) = delete;

private:
	std::vector<std::string> mKernels;
};