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

#define INTERPOLATION_THREAD_BLOCK_SIZE		256

template<int interpolation_type, int block_type>
__global__ void interpolation_kernel(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation)
{
  // load current block
  blockCopy_type block = interpolation.blocks[block_type][blockIdx.x];

  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int   dim_i       = block.dim.i<<1; // calculate the dimensions of the resultant fine block
  int   dim_j       = block.dim.j<<1;
  int   dim_k       = block.dim.k<<1;

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
     read = level_c.my_boxes[ block.read.box].vectors[id_c] + level_c.my_boxes[ block.read.box].ghosts*(1+level_c.my_boxes[ block.read.box].jStride+level_c.my_boxes[ block.read.box].kStride);
     read_jStride = level_c.my_boxes[block.read.box ].jStride;
     read_kStride = level_c.my_boxes[block.read.box ].kStride;
  }
  if(block.write.box>=0){
    write = level_f.my_boxes[block.write.box].vectors[id_f] + level_f.my_boxes[block.write.box].ghosts*(1+level_f.my_boxes[block.write.box].jStride+level_f.my_boxes[block.write.box].kStride);
    write_jStride = level_f.my_boxes[block.write.box].jStride;
    write_kStride = level_f.my_boxes[block.write.box].kStride;
  }

  int i = threadIdx.x % dim_i;
  int j_block_stride = INTERPOLATION_THREAD_BLOCK_SIZE / dim_i;

  for (int j = threadIdx.x / dim_i; j < dim_j; j += j_block_stride)
  for (int k = 0; k < dim_k; k++) {
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    switch (interpolation_type) {
    case 0:
      // piece-wise
      write[write_ijk] = prescale_f*write[write_ijk] + read[read_ijk]; // CAREFUL !!!  you must guarantee you zero'd the MPI buffers(write[]) and destination boxes at some point to avoid 0.0*NaN or 0.0*inf
      break;
    case 1:
      // linear
      //
      // |   o   |   o   |
      // +---+---+---+---+
      // |   | x | x |   |
      //
      // CAREFUL !!!  you must guarantee you zero'd the MPI buffers(write[]) and destination boxes at some point to avoid 0.0*NaN or 0.0*inf
      // piecewise linear interpolation... NOTE, BC's must have been previously applied
      int delta_i=           -1;if(i&0x1)delta_i=           1; // i.e. even points look backwards while odd points look forward
      int delta_j=-read_jStride;if(j&0x1)delta_j=read_jStride;
      int delta_k=-read_kStride;if(k&0x1)delta_k=read_kStride;
      write[write_ijk] = prescale_f*write[write_ijk] +
          0.421875*read[read_ijk                        ] +
          0.140625*read[read_ijk                +delta_k] +
          0.140625*read[read_ijk        +delta_j        ] +
          0.046875*read[read_ijk        +delta_j+delta_k] +
          0.140625*read[read_ijk+delta_i                ] +
          0.046875*read[read_ijk+delta_i        +delta_k] +
          0.046875*read[read_ijk+delta_i+delta_j        ] +
          0.015625*read[read_ijk+delta_i+delta_j+delta_k];
      break;
    }
  }
}

extern "C"
void cuda_interpolation_pc(level_type d_level_f, int id_f, double prescale_f, level_type d_level_c, int id_c, communicator_type interpolation, int block_type)
{
  int block = INTERPOLATION_THREAD_BLOCK_SIZE;
  int grid = interpolation.num_blocks[block_type];

  if (grid > 0) {
    switch (block_type) {
      case 0: interpolation_kernel<0,0><<<grid, block>>>(d_level_f, id_f, prescale_f, d_level_c, id_c, interpolation); break;
      case 1: interpolation_kernel<0,1><<<grid, block>>>(d_level_f, id_f, prescale_f, d_level_c, id_c, interpolation); break;
    }
  }
} 

extern "C"
void cuda_interpolation_pl(level_type d_level_f, int id_f, double prescale_f, level_type d_level_c, int id_c, communicator_type interpolation, int block_type)
{
  int block = INTERPOLATION_THREAD_BLOCK_SIZE;
  int grid = interpolation.num_blocks[block_type];

  if (grid > 0) {
    switch (block_type) {
      case 0: interpolation_kernel<1,0><<<grid, block>>>(d_level_f, id_f, prescale_f, d_level_c, id_c, interpolation); break;
      case 1: interpolation_kernel<1,1><<<grid, block>>>(d_level_f, id_f, prescale_f, d_level_c, id_c, interpolation); break;
    }
  }
}

