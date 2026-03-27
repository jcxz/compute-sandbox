#include "gpu/gpu.h"
#include <iostream>

extern bool testInvert();


int main()
{
	const uint32_t flags = GpuFlags::Debug | GpuFlags::RDocCapture | GpuFlags::PreloadKernels;
	if (!InitializeGpu(flags))
	{
		std::cerr << "Failed to initialize GPU" << std::endl;
		return 1;
	}

	std::cout << "========== Test Invert ==========" << std::endl;
	testInvert();

	TerminateGpu();

	return 0;
}
