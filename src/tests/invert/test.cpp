#include "gpu/gpu.h"
#include "invert.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>



[[maybe_unused]] static bool WritePPM(
	const char* const path,
	const uint8_t* const pData,
	const size_t strideBytes,
	const uint32_t width,
	const uint32_t height)
{
	FILE* f = fopen(path, "w");
	if (f == nullptr)
	{
		std::cerr << "Failed to open " << path << " for writing" << std::endl;
		return false;
	}

	fprintf(f, "%s\n%d %d\n255\n", "P2", width, height);

	for (uint32_t y = 0; y < height; y++)
	{
		const uint8_t* pLine = pData + strideBytes * y;
		for (uint32_t x = 0; x < width; ++x)
		{
			fprintf(f, "%d\n", pLine[x]);
		}
	}

	fclose(f);

	return true;
}

[[maybe_unused]] static void GenerateRandom(uint8_t* data, const uint32_t w, const uint32_t h)
{
	static std::random_device sRngSeed;
	static std::mt19937 sRng(sRngSeed());
	static std::uniform_int_distribution<uint32_t> sDist(0, 255);

	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			data[x + y * w] = static_cast<uint8_t>(sDist(sRng));
		}
	}
}

[[maybe_unused]] static void GenerateCheckerboard(uint8_t* data, const uint32_t w, const uint32_t h)
{
	static constexpr uint32_t patternSize = 256; //16;
	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			data[x + y * w] = (x / patternSize + y / patternSize) & 1u ? 0 : 255;
		}
	}
}


bool testInvert()
{
	// Create buffers
	const size_t W = 8192;
	const size_t H = 8192;

	uint8_t* src = (uint8_t*) GpuAlloc   (W * H * sizeof(uint8_t));
	uint8_t* cpu = (uint8_t*) std::malloc(W * H * sizeof(uint8_t));
	uint8_t* gpu = (uint8_t*) GpuAlloc   (W * H * sizeof(uint8_t));

	// Create input data
	//GenerateRandom(src, W, H);
	GenerateCheckerboard(src, W, H);
	//WritePPM("src.ppm", src, W * sizeof(uint8_t), W, H);

	// compute on CPU (reference)
	auto cpu_t0 = std::chrono::high_resolution_clock::now();
	{
		Invert::ArgsType args;
		args.dstStride = W * sizeof(uint8_t);
		args.srcStride = W * sizeof(uint8_t);
		args.src = src;
		args.dst = cpu;
		ExecuteCPUKernel<Invert>(W, H, args);
	}
	auto cpu_t1 = std::chrono::high_resolution_clock::now();

	// compute on GPU
	BeginGpuCapture();

	auto gpu_t0 = std::chrono::high_resolution_clock::now();
	{
		Invert::ArgsType args;
		args.dstStride = W * sizeof(uint8_t);
		args.srcStride = W * sizeof(uint8_t);
		args.src = src;
		args.dst = gpu;
		ExecuteGPUKernel<Invert>(W, H, args);
	}
	auto gpu_t1 = std::chrono::high_resolution_clock::now();

	EndGpuCapture();

	// write outputs as images
	//WritePPM("cpu.ppm", cpu, W * sizeof(uint8_t), W, H);
	//WritePPM("gpu.ppm", gpu, W * sizeof(uint8_t), W, H);

	// validate results
	for (uint32_t y = 0; y < H; ++y)
	{
		for (uint32_t x = 0; x < W; ++x)
		{
			const uint32_t i = x + y * W;
			if (cpu[i] != gpu[i])
			{
				std::cerr << "Invalid results, cpu and gpu differ at element [" << x << ", " << y << "]: " << (uint32_t)cpu[i] << " vs " << (uint32_t)gpu[i] << std::endl;
				GpuFree  (src);
				std::free(cpu);
				GpuFree  (gpu);
				return false;
			}
		}
	}

	std::cout << "CPU and GPU produces equivalent outputs" << std::endl;
	std::cout << "CPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_t1 - cpu_t0).count() << " ms" << std::endl;
	std::cout << "GPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(gpu_t1 - gpu_t0).count() << " ms" << std::endl;

	GpuFree  (src);
	std::free(cpu);
	GpuFree  (gpu);

	return true;
}
