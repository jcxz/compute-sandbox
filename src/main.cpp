#include "gpu/gpu.h"
#include <iostream>
#include <string_view>

extern bool testInvert();


int main(const int argc, char* argv[])
{
	uint32_t flags = GpuFlags::Debug | GpuFlags::RDocCapture | GpuFlags::PreloadKernels;
	for (int i = 1; i < argc; ++i)
	{
		const std::string_view arg = argv[i];
		if (arg == "--enable-debug")
		{
			flags |= GpuFlags::Debug;
		}
		else if (arg == "--disable-capture")
		{
			flags &= ~GpuFlags::RDocCapture;
		}
		else if (arg == "--disable-preload")
		{
			flags &= ~GpuFlags::PreloadKernels;
		}
	}

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
