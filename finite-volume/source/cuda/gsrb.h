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
#define GSRB_MIN_BLOCKS_PER_SM         (2048 / 2 / (BLOCKCOPY_TILE_I * BLOCKCOPY_TILE_J))

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
#ifdef GSRB_MIN_BLOCKS_PER_SM
__launch_bounds__((BLOCKCOPY_TILE_I * BLOCKCOPY_TILE_J), GSRB_MIN_BLOCKS_PER_SM)
#endif
__global__ void gsrb_smooth_kernel(level_type level, int x_id, int rhs_id, double a, double b, int s)
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

  const int color000 = (level.my_boxes[box].low.i^level.my_boxes[box].low.j^level.my_boxes[box].low.k^s)&1;  // is element 000 red or black on *THIS* sweep

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
  #ifdef GSRB_OOP
  const double * __restrict__ x_n;
        double * __restrict__ x_np1;
                 if((s&1)==0){x_n      = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);
                              x_np1    = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);}
                         else{x_n      = level.my_boxes[box].vectors[VECTOR_TEMP  ] + ghosts*(1+jStride+kStride);
                              x_np1    = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride);}
  #else
  const double * __restrict__ x_n      = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
        double * __restrict__ x_np1    = level.my_boxes[box].vectors[         x_id] + ghosts*(1+jStride+kStride); // i.e. [0] = first non ghost zone point
  #endif

  int j = jlo + threadIdx.y;
  if (j >= jhi) return;
#ifndef GSRB_STRIDE2
  int i = ilo + threadIdx.x;
  if (i >= ihi) return;
#endif

      #if defined(GSRB_FP)
      #warning GSRB using pre-computed 1.0/0.0 FP array for Red-Black to facilitate vectorization...
      for(int k=klo;k<khi;k++){const double * __restrict__ RedBlack = level.RedBlack_FP + ghosts*(1+jStride) + kStride*((k^color000)&0x1);
            int ij  = i + j*jStride;
            int ijk = i + j*jStride + k*kStride;
            double Ax     = apply_op_ijk(x_n);
            double lambda =     Dinv_ijk();
            x_np1[ijk] = x_n[ijk] + RedBlack[ij]*lambda*(rhs[ijk]-Ax);
            //x_np1[ijk] = ((i^j^k^color000)&1) ? x_n[ijk] : x_n[ijk] + lambda*(rhs[ijk]-Ax);
      }
      #elif defined(GSRB_STRIDE2)
      for(int k=klo;k<khi;k++){
        #ifdef GSRB_OOP
        #warning GSRB using out-of-place and stride-2 accesses to minimize the number of flops
        {
          // out-of-place must copy old value...
          int i = ilo+!((ilo^j^k^color000)&1) + threadIdx.x*2; // stride-2 GSRB
          if (i < ihi) {
            int ijk = i + j*jStride + k*kStride;
            x_np1[ijk] = x_n[ijk];
          }
        }
	#else
        #warning GSRB using stride-2 accesses to minimize the number of flops
        #endif
          int i = ilo+((ilo^j^k^color000)&1) + threadIdx.x*2; // stride-2 GSRB
          if (i < ihi) {
            int ijk = i + j*jStride + k*kStride;
            double Ax     = apply_op_ijk(x_n);
            double lambda =     Dinv_ijk();
            x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);
          }
      }
      #elif defined(GSRB_OOP)
      #warning GSRB using out-of-place implementation with an if-then-else on loop indices...
      for(int k=klo;k<khi;k++){
        int ijk = i + j*jStride + k*kStride;
        if((i^j^k^color000^1)&1){ // looks very clean when [0] is i,j,k=0,0,0 
          double Ax     = apply_op_ijk(x_n);
          double lambda =     Dinv_ijk();
          x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);
        }else{
          x_np1[ijk] = x_n[ijk]; // copy old value when sweep color != cell color
      }}
      #else
      #warning GSRB using if-then-else on loop indices...
      for(int k=klo;k<khi;k++){
      if((i^j^k^color000^1)&1){ // looks very clean when [0] is i,j,k=0,0,0 
            int ijk = i + j*jStride + k*kStride;
            double Ax     = apply_op_ijk(x_n);
            double lambda =     Dinv_ijk();
            x_np1[ijk] = x_n[ijk] + lambda*(rhs[ijk]-Ax);
      }}
      #endif
}

// kernel declaration with specified level info and block sizes
#ifndef GSRB_STRIDE2
#define GSRB_KERNEL_TILE(log_dim_i, block_i, block_j, block_k) \
	gsrb_smooth_kernel<log_dim_i, block_i, block_j, block_k><<<dim3(num_blocks, (block_dim_k+block_k-1)/block_k), dim3(block_i, block_j)>>>(d_level, phi_id, rhs_id, a, b, s);
#else 
#define GSRB_KERNEL_TILE(log_dim_i, block_i, block_j, block_k) \
	gsrb_smooth_kernel<log_dim_i, block_i/2, block_j, block_k><<<dim3(num_blocks, (block_dim_k+block_k-1)/block_k), dim3(block_i/2, block_j)>>>(d_level, phi_id, rhs_id, a, b, s);
#endif 

// select appropriate block size
#define GSRB_KERNEL(log_dim_i) \
	if (block_dim_i == BLOCKCOPY_TILE_I) { \
	  GSRB_KERNEL_TILE(log_dim_i, BLOCKCOPY_TILE_I, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
        } else { \
               if (block_dim_i <= 1)   GSRB_KERNEL_TILE(log_dim_i,   1, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 2)   GSRB_KERNEL_TILE(log_dim_i,   2, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 4)   GSRB_KERNEL_TILE(log_dim_i,   4, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 8)   GSRB_KERNEL_TILE(log_dim_i,   8, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 16)  GSRB_KERNEL_TILE(log_dim_i,  16, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 32)  GSRB_KERNEL_TILE(log_dim_i,  32, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 64)  GSRB_KERNEL_TILE(log_dim_i,  64, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 128) GSRB_KERNEL_TILE(log_dim_i, 128, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
	  else if (block_dim_i <= 256) GSRB_KERNEL_TILE(log_dim_i, 256, BLOCKCOPY_TILE_J, BLOCKCOPY_TILE_K) \
          else printf("ERROR: this tile dimension is not supported in the GPU path, please update the macros!\n"); \
        }

// maximum supported level can have 2^10 dimension
#define GSRB_KERNEL_DETECT_LEVEL(log_dim_i) \
	switch (log_dim_i) { \
        case 0: { GSRB_KERNEL(0) break; } \
        case 1: { GSRB_KERNEL(1) break; } \
        case 2: { GSRB_KERNEL(2) break; } \
        case 3: { GSRB_KERNEL(3) break; } \
        case 4: { GSRB_KERNEL(4) break; } \
        case 5: { GSRB_KERNEL(5) break; } \
        case 6: { GSRB_KERNEL(6) break; } \
        case 7: { GSRB_KERNEL(7) break; } \
        case 8: { GSRB_KERNEL(8) break; } \
        case 9: { GSRB_KERNEL(9) break; } \
        case 10: { GSRB_KERNEL(10) break; } \
	default: { printf("ERROR: this level size is not supported in the GPU path, please update the macros!\n"); } \
        }

extern "C"
void cuda_gsrb_smooth(level_type d_level, int phi_id, int rhs_id, double a, double b, int s)
{
  int num_blocks = d_level.num_my_blocks;
  if (num_blocks <= 0) return;

  int log_dim_i = (int)log2((double)d_level.dim.i);
  int block_dim_i = min(d_level.box_dim, BLOCKCOPY_TILE_I);
  int block_dim_k = min(d_level.box_dim, BLOCKCOPY_TILE_K);

  GSRB_KERNEL_DETECT_LEVEL(log_dim_i)
}
