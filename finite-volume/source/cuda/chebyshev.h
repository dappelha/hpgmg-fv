/*
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#if BLOCKCOPY_TILE_I == 10000
#error GPU code only supports 3D tiling, please specify BLOCKCOPY_TILE_I
#endif

// enforce min 50% occupancy on Kepler/Maxwell
#define CHEBY_MIN_BLOCKS_PER_SM		(2048 / 2 / (BLOCKCOPY_TILE_I * BLOCKCOPY_TILE_J))

// select stencil implementation
#ifndef CUDA_STENCIL_OPT_TEX
/*
 ------------------------------------------------------------------------------
 Baseline implementation
 ------------------------------------------------------------------------------
 This implementation requires minimum changes to the original CPU version by
 preserving the same code structure and stencil macros. The key difference is
 using CUDA thread block instead of single OMP thread for each block/tile, thus
 enabling finer granularity parallelism. The 2D thread block size is BLOCK_I x
 BLOCK_J and it computes BLOCK_K elements of 3D tile BLOCK_I x BLOCK_J x
 BLOCK_K. LOG_DIM_I is used to detect MG level depth at compile time for easier
 profiling of different levels.
 ------------------------------------------------------------------------------
*/
template<int LOG_DIM_I, int BLOCK_I, int BLOCK_J, int BLOCK_K> 
#ifdef CHEBY_MIN_BLOCKS_PER_SM
__launch_bounds__((BLOCKCOPY_TILE_I * BLOCKCOPY_TILE_J), CHEBY_MIN_BLOCKS_PER_SM)
#endif
__global__ void cheby_smooth_kernel(level_type level, int x_id, int rhs_id, double a, double b, int s, double *chebyshev_c1, double *chebyshev_c2)
{
  // one CUDA thread block works on one tile/block
  int block = blockIdx.x;

  const int box = level.my_blocks[block].read.box;
  const int ilo = level.my_blocks[block].read.i;
  const int jlo = level.my_blocks[block].read.j;
  const int klo = level.my_blocks[block].read.k + BLOCK_K * blockIdx.y;
  const int ihi = level.my_blocks[block].dim.i + ilo;
  const int jhi = level.my_blocks[block].dim.j + jlo;
  const int khi = min(level.my_blocks[block].dim.k + klo, BLOCK_K + klo);
  const int ghosts = level.box_ghosts;
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  const double h2inv = 1.0/(level.h*level.h);

  const double * __restrict__ rhs      = level.my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
#ifdef USE_HELMHOLTZ
  const double * __restrict__ alpha    = level.my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
#endif
  const double * __restrict__ beta_i   = level.my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
  const double * __restrict__ beta_j   = level.my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
  const double * __restrict__ beta_k   = level.my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
  const double * __restrict__ Dinv     = level.my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
#ifdef STENCIL_FUSE_BC
  const double * __restrict__ valid    = level.my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); // cell is inside the domain
#endif

        double * __restrict__ x_np1;
  const double * __restrict__ x_n;
  const double * __restrict__ x_nm1;
                   if((s&1)==0){x_n    = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                x_nm1  = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                x_np1  = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                           else{x_n    = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                x_nm1  = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                x_np1  = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}

  const double c1 = chebyshev_c1[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.
  const double c2 = chebyshev_c2[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.

  int i = ilo + threadIdx.x;
  int j = jlo + threadIdx.y;
  if (i >= ihi || j >= jhi) return;
  
  // each thread works on multiple grid points along k dimension
  for (int k = klo; k < khi; k++) {
    const int ijk = i + j*jStride + k*kStride;
    // According to Saad... but his was missing a Dinv[ijk] == D^{-1} !!!
    //  x_{n+1} = x_{n} + rho_{n} [ rho_{n-1}(x_{n} - x_{n-1}) + (2/delta)(b-Ax_{n}) ]
    //  x_temp[ijk] = x_n[ijk] + c1*(x_n[ijk]-x_temp[ijk]) + c2*Dinv[ijk]*(rhs[ijk]-Ax_n);
    const double Ax_n   = apply_op_ijk(x_n);
    const double lambda =     Dinv_ijk();
    x_np1[ijk] = x_n[ijk] + c1*(x_n[ijk]-x_nm1[ijk]) + c2*lambda*(rhs[ijk]-Ax_n);
  }
}

#else
/*
 ------------------------------------------------------------------------------
 Optimized implementation
 ------------------------------------------------------------------------------
 This implementation optimizes stencil computations by 1) explicitly storing
 current K slice in registers for reuse in the next K+1 iteration, 2) using
 read-only cache for loading x values through LDG instruction. Otherwise the
 thread scheduling strategy is the same as in the baseline implementation.
------------------------------------------------------------------------------
*/
template<int LOG_DIM_I, int BLOCK_I, int BLOCK_J, int BLOCK_K>
#ifdef CHEBY_MIN_BLOCKS_PER_SM
__launch_bounds__((BLOCKCOPY_TILE_I * BLOCKCOPY_TILE_J), CHEBY_MIN_BLOCKS_PER_SM)
#endif
__global__ void cheby_smooth_kernel(level_type level, int x_id, int rhs_id, double a, double b, int s, double *chebyshev_c1, double *chebyshev_c2)
{
  int block = blockIdx.x;

  const int box = level.my_blocks[block].read.box;
  const int ilo = level.my_blocks[block].read.i;
  const int jlo = level.my_blocks[block].read.j;
  const int klo = level.my_blocks[block].read.k + BLOCK_K * blockIdx.y;
  const int ihi = level.my_blocks[block].dim.i + ilo;
  const int jhi = level.my_blocks[block].dim.j + jlo;
  const int khi = min(level.my_blocks[block].dim.k + klo, BLOCK_K + klo);
  const int ghosts = level.box_ghosts;
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  const double h2inv = 1.0/(level.h*level.h);

  const double * __restrict__ rhs      = level.my_boxes[box].vectors[       rhs_id] + ghosts*(1+jStride+kStride);
#ifdef USE_HELMHOLTZ
  const double * __restrict__ alpha    = level.my_boxes[box].vectors[VECTOR_ALPHA ] + ghosts*(1+jStride+kStride);
#endif
  const double * __restrict__ beta_i   = level.my_boxes[box].vectors[VECTOR_BETA_I] + ghosts*(1+jStride+kStride);
  const double * __restrict__ beta_j   = level.my_boxes[box].vectors[VECTOR_BETA_J] + ghosts*(1+jStride+kStride);
  const double * __restrict__ beta_k   = level.my_boxes[box].vectors[VECTOR_BETA_K] + ghosts*(1+jStride+kStride);
  const double * __restrict__ Dinv     = level.my_boxes[box].vectors[VECTOR_DINV  ] + ghosts*(1+jStride+kStride);
#ifdef STENCIL_FUSE_BC
  const double * __restrict__ valid    = level.my_boxes[box].vectors[VECTOR_VALID ] + ghosts*(1+jStride+kStride); // cell is inside the domain
#endif

        double * __restrict__ x_np1;
  const double * __restrict__ x_n;
  const double * __restrict__ x_nm1;
                   if((s&1)==0){x_n    = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                x_nm1  = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                x_np1  = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                           else{x_n    = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                                x_nm1  = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                                x_np1  = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}

  const double c1 = chebyshev_c1[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.
  const double c2 = chebyshev_c2[s%CHEBYSHEV_DEGREE]; // limit polynomial to degree CHEBYSHEV_DEGREE.

  int i = ilo + threadIdx.x;
  int j = jlo + threadIdx.y;
  if (i >= ihi || j >= jhi) return;

  // load current/previous x, beta_k and valid into regisers
  double x_kprev = __ldg(x_n + i + j*jStride + klo*kStride - kStride);
  double x_cur = __ldg(x_n + i + j*jStride + klo*kStride);
  double x_knext;
  double beta_k_cur = beta_k[i + j*jStride + klo*kStride];
  double beta_k_next;
#ifdef STENCIL_FUSE_BC
  double valid_kprev = __ldg(valid + i + j*jStride + klo*kStride - kStride);
  double valid_cur = __ldg(valid + i + j*jStride + klo*kStride);
  double valid_knext;
#endif
#if __CUDA_ARCH__ == 350
  // helps Kepler but not Maxwell
  #pragma unroll 2
#endif
  // each thread works on multiple grid points along k dimension
  for (int k = klo; k < khi; k++) {
    const int ijk = i + j*jStride + k*kStride;

    // load next x, beta_k and valid into registers
    x_knext = __ldg(x_n + ijk + kStride);
    beta_k_next = beta_k[ijk + kStride];
#ifdef STENCIL_FUSE_BC
    valid_knext = __ldg(valid + ijk + kStride);
#endif
    const double Ax_n   = apply_op_ijk(x_n);
    const double lambda =     Dinv_ijk();
    x_np1[ijk] = x_cur + c1*(x_cur-x_nm1[ijk]) + c2*lambda*(rhs[ijk]-Ax_n);

    // update x, beta_k and valid in registers
    x_kprev = x_cur;
    x_cur = x_knext;
    beta_k_cur = beta_k_next;
#ifdef STENCIL_FUSE_BC
    valid_kprev = valid_cur;
    valid_cur = valid_knext;
#endif
  }
}
#endif
// TODO: add more stencil implementations here

// run kernel with arguments
#define CHEBY_KERNEL_TILE(log_dim_i, block_i, block_j, block_k) cheby_smooth_kernel<log_dim_i, block_i, block_j, block_k><<<dim3(num_blocks, (block_dim_k+block_k-1)/block_k), dim3(block_i, block_j)>>>(d_level, x_id, rhs_id, a, b, s, d_chebyshev_c1, d_chebyshev_c2);

// select appropriate block size
#define CHEBY_KERNEL(log_dim_i) \
	if (block_dim_i == BLOCKCOPY_TILE_I) { \
	  CHEBY_KERNEL_TILE(log_dim_i, BLOCKCOPY_TILE_I, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
        } else { \
               if (block_dim_i <= 1)   CHEBY_KERNEL_TILE(log_dim_i,   1, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 2)   CHEBY_KERNEL_TILE(log_dim_i,   2, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 4)   CHEBY_KERNEL_TILE(log_dim_i,   4, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 8)   CHEBY_KERNEL_TILE(log_dim_i,   8, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 16)  CHEBY_KERNEL_TILE(log_dim_i,  16, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 32)  CHEBY_KERNEL_TILE(log_dim_i,  32, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 64)  CHEBY_KERNEL_TILE(log_dim_i,  64, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 128) CHEBY_KERNEL_TILE(log_dim_i, 128, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 256) CHEBY_KERNEL_TILE(log_dim_i, 256, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
          else printf("ERROR: this tile dimension is not supported in the GPU path, please update the macros!\n"); \
        }

// maximum supported level can have 2^10 dimension
#define CHEBY_KERNEL_DETECT_LEVEL(log_dim_i) \
	switch (log_dim_i) { \
        case 0: { CHEBY_KERNEL(0) break; } \
        case 1: { CHEBY_KERNEL(1) break; } \
        case 2: { CHEBY_KERNEL(2) break; } \
        case 3: { CHEBY_KERNEL(3) break; } \
        case 4: { CHEBY_KERNEL(4) break; } \
        case 5: { CHEBY_KERNEL(5) break; } \
        case 6: { CHEBY_KERNEL(6) break; } \
        case 7: { CHEBY_KERNEL(7) break; } \
        case 8: { CHEBY_KERNEL(8) break; } \
        case 9: { CHEBY_KERNEL(9) break; } \
        case 10: { CHEBY_KERNEL(10) break; } \
	default: { printf("ERROR: this level size is not supported in the GPU path, please update the macros!\n"); } \
        }

extern "C"
void cuda_cheby_smooth(level_type d_level, int x_id, int rhs_id, double a, double b, int s, double *d_chebyshev_c1, double *d_chebyshev_c2)
{
  int num_blocks = d_level.num_my_blocks;
  int log_dim_i = (int)log2((double)d_level.dim.i);
  int block_dim_i = min(d_level.box_dim, BLOCKCOPY_TILE_I);
  int block_dim_k = min(d_level.box_dim, BLOCKCOPY_TILE_K);

  CHEBY_KERNEL_DETECT_LEVEL(log_dim_i)
}
