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
		if (arg == "--help")
		{
			std::cout << "Usage: compute-sandbox [options]\n"
					  << "Options:\n"
					  << "  --help             Display this help message\n"
					  << "  --enable-debug     Turns on GpuFlags::Debug\n"
					  << "  --disable-capture  Turns off GpuFlags::RDocCapture\n"
					  << "  --disable-preload  Turns off GpuFlags::PreloadKernels\n";
			return 0;
		}
		else if (arg == "--enable-debug")
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
