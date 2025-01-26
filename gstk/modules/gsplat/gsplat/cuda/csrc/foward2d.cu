#include "forward2d.cuh"
#include "helpers.cuh"
#include <algorithm>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <iostream>
namespace cg = cooperative_groups;

// kernel function for projecting each gaussian on device
// each thread processes one gaussian
// 这个函数应该包含的过程：计算2d cov，num_tiles_hit tile_bounds blabla 然后接入rasterize_forward？
__global__ void project_gaussians_2d_forward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float3* __restrict__ L_elements,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 2D Gaussian parameters
    // printf("means2d %d, %.2f %.2f \n", idx, means2d[idx].x, means2d[idx].y);
    // float clamped_x = max(-1.0f, min(1.0f, means2d[idx].x)); // Clamp x between -1 and 1
    // float clamped_y = max(-1.0f, min(1.0f, means2d[idx].y)); // Clamp y between -1 and 1

    float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
                     0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};
    // Assuming L is packed row-wise in a 1D array: [l11, l21, l22]
    float l11 = L_elements[idx].x; // scale_x
    float l21 = L_elements[idx].y; // covariance_xy
    float l22 = L_elements[idx].z; // scale_y

    // Construct the 2x2 covariance matrix from L
    // float2x2 Cov2D = make_float2x2(l11*l11, l11*l21,
                                //    l11*l21, l21*l21 + l22*l22);
    float3 cov2d = make_float3(l11*l11, l11*l21, l21*l21 + l22*l22);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
    // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;

}

__global__ void project_gaussians_2d_scale_rot_forward_kernel(
    const int num_points,
    const float2* __restrict__ means2d,
    const float2* __restrict__ scales2d,
    const float* __restrict__ rotation,
    const dim3 img_size,
    const dim3 tile_bounds,
    const float clip_thresh,
    float2* __restrict__ xys,
    float* __restrict__ depths,
    int* __restrict__ radii,
    float3* __restrict__ conics,
    int32_t* __restrict__ num_tiles_hit
) {
    unsigned idx = cg::this_grid().thread_rank(); // idx of thread within grid
    if (idx >= num_points) {
        return;
    }
    radii[idx] = 0;
    num_tiles_hit[idx] = 0;

    // Retrieve the 2D Gaussian parameters
    float2 center = {0.5f * img_size.x * means2d[idx].x + 0.5f * img_size.x,
                     0.5f * img_size.y * means2d[idx].y + 0.5f * img_size.y};

    glm::mat2 R = rotmat2d(rotation[idx]);
    glm::mat2 S = scale_to_mat2d(scales2d[idx]);
    glm::mat2 M = R * S;
    glm::mat2 tmp = M * glm::transpose(M);
    // glm::mat2 tmp = R * S * glm::transpose(R);

    float3 cov2d = make_float3(tmp[0][0], tmp[0][1], tmp[1][1]);
    // printf("cov2d %d, %.2f %.2f %.2f\n", idx, cov2d.x, cov2d.y, cov2d.z);
    float3 conic;
    float radius;
    bool ok = compute_cov2d_bounds(cov2d, conic, radius);
    if (!ok)
        return; // zero determinant
    // printf("conic %d %.2f %.2f %.2f\n", idx, conic.x, conic.y, conic.z);
    conics[idx] = conic;
    xys[idx] = center;
    radii[idx] = (int)radius;
    uint2 tile_min, tile_max;
    get_tile_bbox(center, radius, tile_bounds, tile_min, tile_max);
    int32_t tile_area = (tile_max.x - tile_min.x) * (tile_max.y - tile_min.y);
    if (tile_area <= 0) {
        // printf("%d point bbox outside of bounds\n", idx);
        return;
    }
    num_tiles_hit[idx] = tile_area;
    // 先给一个固定的depth，为了后面的函数调用方便
    depths[idx] = 0.0f;

}