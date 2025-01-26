#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

__global__ void project_gaussians_2d_backward_kernel(
    const int num_gaussians,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float2* __restrict__ v_mean2d,
    float3* __restrict__ v_L_elements
);

__global__ void project_gaussians_2d_scale_rot_backward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const int* __restrict__ radii,
    const float3* __restrict__ conics,
    const float2* __restrict__ v_xy,
    const float* __restrict__ v_depth,
    const float3* __restrict__ v_conic,
    float3* __restrict__ v_cov2d,
    float2* __restrict__ v_mean2d,
    float2* __restrict__ v_scale,
    float* __restrict__ v_rot
);