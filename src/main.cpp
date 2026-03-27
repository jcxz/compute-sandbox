#include "gpu/gpu.h"
#include <iostream>

extern bool testInvert();


int main()
{
	if (!InitializeGpu(GpuFlags::Debug | GpuFlags::RDocCapture))
	{
		std::cerr << "Failed to initialize GPU" << std::endl;
		return 1;
	}

	std::cout << "========== Test Invert ==========" << std::endl;
	testInvert();

	TerminateGpu();

	return 0;
}
