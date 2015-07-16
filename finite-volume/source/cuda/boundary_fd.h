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

// Divide CUDA block into groups of threads (e.g. quads), each operating on an individual b.c. block.
#define APPLY_BCS_LINEAR_BLOCK_SIZE	128								// number of threads per block
#define APPLY_BCS_LINEAR_GROUP_SIZE	16								// number of threads per group
#define APPLY_BCS_LINEAR_NUM_GROUPS	(APPLY_BCS_LINEAR_BLOCK_SIZE / APPLY_BCS_LINEAR_GROUP_SIZE)	// number of groups per block


__constant__ int   faces[27] = {0,0,0,0,1,0,0,0,0,  0,1,0,1,0,1,0,1,0,  0,0,0,0,1,0,0,0,0};
__constant__ int   edges[27] = {0,1,0,1,0,1,0,1,0,  1,0,1,0,0,0,1,0,1,  0,1,0,1,0,1,0,1,0};
__constant__ int corners[27] = {1,0,1,0,0,0,1,0,1,  0,0,0,0,0,0,0,0,0,  1,0,1,0,0,0,1,0,1};


__global__ void apply_BCs_v1_kernel(level_type level, int x_id, int shape){
  // For cell-centered, we need to fill in the ghost zones to apply any BC's
  // This code does a simple piecewise linear interpolation for homogeneous dirichlet (0 on boundary)
  // Nominally, this is first performed across faces, then to edges, then to corners.  
  // In this implementation, these three steps are fused
  //
  //   . . . . . . . . . .        . . . . . . . . . .
  //   .       .       .          .       .       .
  //   .   ?   .   ?   .          .+x(0,0).-x(0,0).
  //   .       .       .          .       .       .
  //   . . . . +---0---+--        . . . . +-------+--
  //   .       |       |          .       |       |
  //   .   ?   0 x(0,0)|          .-x(0,0)| x(0,0)|
  //   .       |       |          .       |       |
  //   . . . . +-------+--        . . . . +-------+--
  //   .       |       |          .       |       |
  //
  //

  int bid = blockIdx.x*APPLY_BCS_LINEAR_NUM_GROUPS + threadIdx.x/APPLY_BCS_LINEAR_GROUP_SIZE;
  //if(blockIdx.x<2){printf("%d\t%d\t%d\n",blockIdx.x,threadIdx.x,bid);}
  if(bid >= level.boundary_condition.num_blocks[shape]) return;

  // load current block
  blockCopy_type block = level.boundary_condition.blocks[shape][bid];

  double scale = 1.0;
  if(  faces[block.subtype])scale=-1.0;
  if(  edges[block.subtype])scale= 1.0;
  if(corners[block.subtype])scale=-1.0;

  int i,j,k;
  const int       box = block.read.box;
  const int     dim_i = block.dim.i;
  const int     dim_j = block.dim.j;
  const int     dim_k = block.dim.k;
  const int       ilo = block.read.i;
  const int       jlo = block.read.j;
  const int       klo = block.read.k;
  const int normal = 26-block.subtype; // invert the normal vector
 
  // hard code for box to box BC's 
  const int jStride = level.my_boxes[box].jStride;
  const int kStride = level.my_boxes[box].kStride;
  double * __restrict__  x = level.my_boxes[box].vectors[x_id] + level.my_boxes[box].ghosts*(1+jStride+kStride);

  // convert normal vector into pointer offsets...
  const int di = (((normal % 3)  )-1);
  const int dj = (((normal % 9)/3)-1);
  const int dk = (((normal / 9)  )-1);
  const int stride = di + dj*jStride + dk*kStride;

/*
  if(dim_i==1){
    for(int gid=threadIdx.x%APPLY_BCS_LINEAR_GROUP_SIZE; gid<dim_j*dim_k; gid+=APPLY_BCS_LINEAR_GROUP_SIZE){
      k=gid/dim_j;
      j=gid%dim_j;
      int ijk = (  ilo) + (j+jlo)*jStride + (k+klo)*kStride;
      x[ijk] = scale*x[ijk+stride]; // homogeneous linear = 1pt stencil
    }
  }else if(dim_j==1){
    for(int gid=threadIdx.x%APPLY_BCS_LINEAR_GROUP_SIZE; gid<dim_i*dim_k; gid+=APPLY_BCS_LINEAR_GROUP_SIZE){
      k=gid/dim_i;
      i=gid%dim_i;
      int ijk = (i+ilo) + (  jlo)*jStride + (k+klo)*kStride;
      x[ijk] = scale*x[ijk+stride]; // homogeneous linear = 1pt stencil
    }
  }else if(dim_k==1){
    for(int gid=threadIdx.x%APPLY_BCS_LINEAR_GROUP_SIZE; gid<dim_i*dim_j; gid+=APPLY_BCS_LINEAR_GROUP_SIZE){
      j=gid/dim_i;
      i=gid%dim_i;
      int ijk = (i+ilo) + (j+jlo)*jStride + (  klo)*kStride;
      x[ijk] = scale*x[ijk+stride]; // homogeneous linear = 1pt stencil
    }
  }else{
*/
    for(int gid=threadIdx.x%APPLY_BCS_LINEAR_GROUP_SIZE; gid<dim_i*dim_j*dim_k; gid+=APPLY_BCS_LINEAR_GROUP_SIZE){
      k=(gid/dim_i)/dim_j;
      j=(gid/dim_i)%dim_j;
      i=gid%dim_i;
      int ijk = (i+ilo) + (j+jlo)*jStride + (k+klo)*kStride;
      x[ijk] = scale*x[ijk+stride]; // homogeneous linear = 1pt stencil
      //x[ijk] = scale * __ldg(x + ijk + stride);
    }
  //}
}

extern "C"
void cuda_apply_BCs_v1(level_type level, int x_id, int shape)
{
  int block = APPLY_BCS_LINEAR_BLOCK_SIZE;
  int grid = (level.boundary_condition.num_blocks[shape]+APPLY_BCS_LINEAR_NUM_GROUPS-1)/APPLY_BCS_LINEAR_NUM_GROUPS;

  apply_BCs_v1_kernel<<<grid, block>>>(level, x_id, shape);
}
