#include "core/mtl.h"
#include "kernels/invert.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>


static bool WritePPM(
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

static void GenerateRandom(uint8_t* data, const uint32_t w, const uint32_t h)
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

static void GenerateCheckerboard(uint8_t* data, const uint32_t w, const uint32_t h)
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

static void InvertCPU(
	const uint8_t* src,
	const uint32_t srcStride,
	uint8_t* dst,
	const uint32_t dstStride,
	const uint32_t w,
	const uint32_t h)
{
	InvertArgs args;
	args.src = src;
	args.dst = dst;
	args.srcStride = srcStride;
	args.dstStride = dstStride;
	for (uint32_t y = 0; y < h; ++y)
	{
		for (uint32_t x = 0; x < w; ++x)
		{
			invert_kernel({ x, y }, args);
		}
	}
}

static void InvertGPU(
	MTL::CommandQueue* pCommandQueue,
	MTL::ComputePipelineState* pPSO,
	MTL::Buffer* src,
	const uint32_t srcStride,
	MTL::Buffer* dst,
	const uint32_t dstStride,
	const uint32_t w,
	const uint32_t h)
{
	// Enqueue GPU commands
	MTL::CommandBuffer* pCommandBuffer = pCommandQueue->commandBuffer();
	MTL::ComputeCommandEncoder* pEncoder = pCommandBuffer->computeCommandEncoder();

	// bind the compute pipeline
	pEncoder->setComputePipelineState(pPSO);

	// set up an argument buffer for the InvertArgs struct
	InvertArgs args;
	args.src = reinterpret_cast<uint8_t*>(src->gpuAddress());
	args.dst = reinterpret_cast<uint8_t*>(dst->gpuAddress());
	args.srcStride = srcStride;
	args.dstStride = dstStride;

	pEncoder->setBytes(&args, sizeof(args), 0);
	// this is needed so that Metal knows that these buffers are actually
	// used by the current command buffer (it does not know what is inside
	// the blob passed on the line above). I imagine this is like the Resource State
	// flag that I tracked in my Vulkan renderer so that I could do automatic barrier transitions.
	pEncoder->useResource(src, MTL::ResourceUsageRead);
	pEncoder->useResource(dst, MTL::ResourceUsageWrite);

	MTL::Size gridSize = MTL::Size(w, h, 1);
	MTL::Size threadsPerThreadgroup = MTL::Size(pPSO->maxTotalThreadsPerThreadgroup(), 1, 1);

	pEncoder->dispatchThreads(gridSize, threadsPerThreadgroup);
	pEncoder->endEncoding();

	// execute computation on the GPU and wait for it
	pCommandBuffer->commit();
	pCommandBuffer->waitUntilCompleted();

	// how long the GPU took to execute the command buffer (i.e. the computation kernel)
	const double kernelStartTime = pCommandBuffer->kernelStartTime();
	const double kernelEndTime = pCommandBuffer->kernelEndTime();
	const double kerneTimelMs = (kernelEndTime - kernelStartTime) * 1000.0;
	const double gpuStartTime = pCommandBuffer->GPUStartTime();
	const double gpuEndTime = pCommandBuffer->GPUEndTime();
	const double gpuTimeMs = (gpuEndTime - gpuStartTime) * 1000.0;
	std::cout << "GPU scheduling time: " << kerneTimelMs << " ms" << std::endl;
	std::cout << "GPU execution time: " << gpuTimeMs << " ms" << std::endl;
}


bool test3()
{
	// a global autorelease pool for all temporary objects released with autorelease (otherwise they would leak)
	NS::SharedPtr<NS::AutoreleasePool> pAutoReleasePool = TransferPtr(NS::AutoreleasePool::alloc()->init());

	// create default system device
	NS::SharedPtr<MTL::Device> pDevice = TransferPtr(MTL::CreateSystemDefaultDevice());
	if (!pDevice)
	{
		std::cerr << "Failed to create system default device." << std::endl;;
		return false;
	}

	// load shaders
	NS::SharedPtr<MTL::Library> pLibrary = TransferPtr(pDevice->newDefaultLibrary());
	if (!pLibrary)
	{
		std::cerr << "Failed to load default library." << std::endl;;
		return false;
	}

	// create compute pipeline for the add_kernel shader
	NS::Error* pError = nullptr;
	NS::SharedPtr<MTL::Function> pInvertFn = TransferPtr(pLibrary->newFunction(NS::String::string("invert_kernel", NS::UTF8StringEncoding)));
	NS::SharedPtr<MTL::ComputePipelineState> pPSO = TransferPtr(pDevice->newComputePipelineState(pInvertFn.get(), &pError));
	if (!pPSO)
	{
		std::cerr << "Failed to load create compute pipeline: " << pError->localizedDescription()->utf8String() << std::endl;;
		return false;
	}

	// create command queue for submitting commands to the GPU
	NS::SharedPtr<MTL::CommandQueue> pCommandQueue = TransferPtr(pDevice->newCommandQueue());
	if (!pCommandQueue)
	{
		std::cerr << "Failed to create command queue" << std::endl;
		return false;
	}

	// Create buffers
	const size_t W = 8192;
	const size_t H = 8192;

	NS::SharedPtr<MTL::Buffer> bufSrc = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resCPU = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));
	NS::SharedPtr<MTL::Buffer> resGPU = TransferPtr(pDevice->newBuffer(W * H * sizeof(uint8_t), MTL::ResourceStorageModeShared));

	// Create input data
	//GenerateRandom(static_cast<uint8_t*>(bufSrc->contents()), W, H);
	GenerateCheckerboard(static_cast<uint8_t*>(bufSrc->contents()), W, H);
	//WritePPM("src.ppm", static_cast<uint8_t*>(bufSrc->contents()), W * sizeof(uint8_t), W, H);

	// compute on CPU (reference)
	auto cpu_t0 = std::chrono::high_resolution_clock::now();
	InvertCPU(static_cast<const uint8_t*>(bufSrc->contents()), W * sizeof(uint8_t), static_cast<uint8_t*>(resCPU->contents()), W * sizeof(uint8_t), W, H);
	auto cpu_t1 = std::chrono::high_resolution_clock::now();

	// compute on GPU
	auto gpu_t0 = std::chrono::high_resolution_clock::now();
	InvertGPU(pCommandQueue.get(), pPSO.get(), bufSrc.get(), W * sizeof(uint8_t), resGPU.get(), W * sizeof(uint8_t), W, H);
	auto gpu_t1 = std::chrono::high_resolution_clock::now();

	// write outputs as images
	const uint8_t* cpu = static_cast<const uint8_t*>(resCPU->contents());
	//WritePPM("cpu.ppm", cpu, W * sizeof(uint8_t), W, H);

	const uint8_t* gpu = static_cast<const uint8_t*>(resGPU->contents());
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
				return false;
			}
		}
	}

	std::cout << "CPU and GPU produces equivalent outputs" << std::endl;
	std::cout << "CPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpu_t1 - cpu_t0).count() << " ms" << std::endl;
	std::cout << "GPU time " << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(gpu_t1 - gpu_t0).count() << " ms" << std::endl;

	return true;
}
