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

#define RESTRICTION_THREAD_BLOCK_SIZE		256

template<int block_type, int restrictionType>
__global__ void restriction_kernel(level_type level_c, int id_c, level_type level_f, int id_f, communicator_type restriction)
{
  // load current block
  blockCopy_type block = restriction.blocks[block_type][blockIdx.x];

  // restrict 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i; // calculate the dimensions of the resultant coarse block
  int   dim_j       = block.dim.j;
  int   dim_k       = block.dim.k;

  int  read_i       = block.read.i;
  int  read_j       = block.read.j;
  int  read_k       = block.read.k;
  int  read_jStride = block.read.jStride;
  int  read_kStride = block.read.kStride;

  int write_i       = block.write.i;
  int write_j       = block.write.j;
  int write_k       = block.write.k;
  int write_jStride = block.write.jStride;
  int write_kStride = block.write.kStride;

  double * __restrict__  read = block.read.ptr;
  double * __restrict__ write = block.write.ptr;

  if(block.read.box >=0){
     read = level_f.my_boxes[ block.read.box].vectors[id_f] + level_f.my_boxes[ block.read.box].ghosts*(1+level_f.my_boxes[ block.read.box].jStride+level_f.my_boxes[ block.read.box].kStride);
     read_jStride = level_f.my_boxes[block.read.box ].jStride;
     read_kStride = level_f.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level_c.my_boxes[block.write.box].vectors[id_c] + level_c.my_boxes[block.write.box].ghosts*(1+level_c.my_boxes[block.write.box].jStride+level_c.my_boxes[block.write.box].kStride);
    write_jStride = level_c.my_boxes[block.write.box].jStride;
    write_kStride = level_c.my_boxes[block.write.box].kStride;
  }

  int i = threadIdx.x % dim_i;
  int j_block_stride = RESTRICTION_THREAD_BLOCK_SIZE / dim_i;

  switch(restrictionType){
    case RESTRICT_CELL:
         for (int j = threadIdx.x / dim_i; j < dim_j; j += j_block_stride)
         for (int k = 0; k < dim_k; k++) {
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk                            ]+read[read_ijk+1                          ] +
                                read[read_ijk  +read_jStride             ]+read[read_ijk+1+read_jStride             ] +
                                read[read_ijk               +read_kStride]+read[read_ijk+1             +read_kStride] +
                                read[read_ijk  +read_jStride+read_kStride]+read[read_ijk+1+read_jStride+read_kStride] ) * 0.125;
         }break;
    case RESTRICT_FACE_I:
         for (int j = threadIdx.x / dim_i; j < dim_j; j += j_block_stride)
         for (int k = 0; k < dim_k; k++) {
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk                          ] +
                                read[read_ijk+read_jStride             ] +
                                read[read_ijk             +read_kStride] +
                                read[read_ijk+read_jStride+read_kStride] ) * 0.25;
         }break;
    case RESTRICT_FACE_J:
         for (int j = threadIdx.x / dim_i; j < dim_j; j += j_block_stride)
         for (int k = 0; k < dim_k; k++) {
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk               ] +
                                read[read_ijk+1             ] +
                                read[read_ijk  +read_kStride] +
                                read[read_ijk+1+read_kStride] ) * 0.25;
         }break;
    case RESTRICT_FACE_K:
         for (int j = threadIdx.x / dim_i; j < dim_j; j += j_block_stride)
         for (int k = 0; k < dim_k; k++) {
           int write_ijk = ((i   )+write_i) + ((j   )+write_j)*write_jStride + ((k   )+write_k)*write_kStride;
           int  read_ijk = ((i<<1)+ read_i) + ((j<<1)+ read_j)* read_jStride + ((k<<1)+ read_k)* read_kStride;
           write[write_ijk] = ( read[read_ijk               ] +
                                read[read_ijk+1             ] +
                                read[read_ijk  +read_jStride] +
                                read[read_ijk+1+read_jStride] ) * 0.25;
         }break;
  }
}

extern "C"
void cuda_restriction(level_type d_level_c, int id_c, level_type d_level_f, int id_f, communicator_type restriction, int restrictionType, int block_type)
{
  int block = RESTRICTION_THREAD_BLOCK_SIZE;
  int grid = restriction.num_blocks[block_type];

  if (grid > 0) {
    switch (block_type) {
      case 0: {
	switch(restrictionType) {
	case RESTRICT_CELL: restriction_kernel<0, RESTRICT_CELL><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
	case RESTRICT_FACE_I: restriction_kernel<0, RESTRICT_FACE_I><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
	case RESTRICT_FACE_J: restriction_kernel<0, RESTRICT_FACE_J><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
	case RESTRICT_FACE_K: restriction_kernel<0, RESTRICT_FACE_K><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
        default: printf("CUDA restriction error: incorrect restrictionType = %i\n", block_type);
        }
        break;
      }
      case 1: {
	switch(restrictionType) {
	case RESTRICT_CELL: restriction_kernel<1, RESTRICT_CELL><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
	case RESTRICT_FACE_I: restriction_kernel<1, RESTRICT_FACE_I><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
	case RESTRICT_FACE_J: restriction_kernel<1, RESTRICT_FACE_J><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
	case RESTRICT_FACE_K: restriction_kernel<1, RESTRICT_FACE_K><<<grid, block>>>(d_level_c, id_c, d_level_f, id_f, restriction); break;
        default: printf("CUDA restriction error: incorrect restrictionType = %i\n", block_type);
        }
        break;
      }
      default: printf("CUDA restriction error: incorrect block_type = %i\n", block_type);
    }
  }
} 
