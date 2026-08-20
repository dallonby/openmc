//! \file backend_metal.mm
//! Apple Metal implementation of the GPU backend ABI (backend.h).
//! Device kernels are runtime-compiled from embedded MSL source so no
//! offline metal toolchain is required.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstring>
#include <string>

#include "backend.h"

namespace {

struct MetalCtx {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLLibrary> library = nil;
  NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* psos = nil;
  id<MTLBuffer> buffers[OMG_SLOT_COUNT] = {};
  double last_time = 0.0;
  std::string device_name;
};

void set_err(char* err, int errcap, NSString* msg)
{
  if (err && errcap > 0) {
    std::strncpy(err, msg.UTF8String ? msg.UTF8String : "unknown error",
      (size_t)errcap - 1);
    err[errcap - 1] = '\0';
  }
}

} // namespace

extern "C" {

int omg_metal_available(void)
{
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    return dev != nil ? 1 : 0;
  }
}

void* omg_metal_create(void)
{
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev)
      return nullptr;
    auto* ctx = new MetalCtx();
    ctx->device = dev;
    ctx->queue = [dev newCommandQueue];
    ctx->psos = [NSMutableDictionary new];
    ctx->device_name = dev.name.UTF8String;
    return ctx;
  }
}

void omg_metal_destroy(void* vctx)
{
  auto* ctx = static_cast<MetalCtx*>(vctx);
  if (!ctx)
    return;
  @autoreleasepool {
    for (auto& b : ctx->buffers)
      b = nil;
    ctx->psos = nil;
    ctx->library = nil;
    ctx->queue = nil;
    ctx->device = nil;
  }
  delete ctx;
}

int omg_metal_compile(void* vctx, const char* src, char* err, int errcap)
{
  auto* ctx = static_cast<MetalCtx*>(vctx);
  @autoreleasepool {
    NSError* nserr = nil;
    MTLCompileOptions* opts = [MTLCompileOptions new];
    // Precise fp32 math. Fast math's approximate divide/sqrt/log introduce
    // ~1e-6-level systematic biases that compound over long collision
    // chains and visibly bias k-eff (observed ~-0.5% on Godiva);
    // correctness beats the modest ALU win.
    opts.mathMode = MTLMathModeSafe;
    id<MTLLibrary> lib =
      [ctx->device newLibraryWithSource:@(src) options:opts error:&nserr];
    if (!lib) {
      set_err(err, errcap, nserr.localizedDescription);
      return 1;
    }
    ctx->library = lib;
    [ctx->psos removeAllObjects];
    return 0;
  }
}

int omg_metal_buffer(void* vctx, int slot, size_t bytes)
{
  auto* ctx = static_cast<MetalCtx*>(vctx);
  if (slot < 0 || slot >= OMG_SLOT_COUNT)
    return 1;
  if (bytes == 0)
    bytes = 4; // Metal requires non-empty bindings
  @autoreleasepool {
    id<MTLBuffer> cur = ctx->buffers[slot];
    if (cur && cur.length >= bytes)
      return 0;
    id<MTLBuffer> buf =
      [ctx->device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
    if (!buf)
      return 1;
    ctx->buffers[slot] = buf;
    return 0;
  }
}

void* omg_metal_contents(void* vctx, int slot)
{
  auto* ctx = static_cast<MetalCtx*>(vctx);
  if (slot < 0 || slot >= OMG_SLOT_COUNT || !ctx->buffers[slot])
    return nullptr;
  return ctx->buffers[slot].contents;
}

int omg_metal_dispatch(
  void* vctx, const char* fn, unsigned int nthreads, char* err, int errcap)
{
  auto* ctx = static_cast<MetalCtx*>(vctx);
  if (nthreads == 0)
    return 0;
  @autoreleasepool {
    NSString* name = @(fn);
    id<MTLComputePipelineState> pso = ctx->psos[name];
    if (!pso) {
      id<MTLFunction> f = [ctx->library newFunctionWithName:name];
      if (!f) {
        set_err(err, errcap,
          [NSString stringWithFormat:@"kernel '%s' not found", fn]);
        return 1;
      }
      NSError* nserr = nil;
      pso = [ctx->device newComputePipelineStateWithFunction:f error:&nserr];
      if (!pso) {
        set_err(err, errcap, nserr.localizedDescription);
        return 1;
      }
      ctx->psos[name] = pso;
    }

    id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (int i = 0; i < OMG_SLOT_COUNT; ++i) {
      if (ctx->buffers[i])
        [enc setBuffer:ctx->buffers[i] offset:0 atIndex:(NSUInteger)i];
    }
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > 256)
      tg = 256;
    [enc dispatchThreads:MTLSizeMake(nthreads, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.status == MTLCommandBufferStatusError) {
      set_err(err, errcap, cb.error.localizedDescription);
      return 1;
    }
    ctx->last_time = cb.GPUEndTime - cb.GPUStartTime;
    return 0;
  }
}

double omg_metal_last_time(void* vctx)
{
  return static_cast<MetalCtx*>(vctx)->last_time;
}

const char* omg_metal_device_name(void* vctx)
{
  return static_cast<MetalCtx*>(vctx)->device_name.c_str();
}

} // extern "C"
