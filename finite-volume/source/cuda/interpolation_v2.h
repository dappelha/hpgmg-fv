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

#define INTERPOLATION_V2_BLOCK_SIZE         128		// number of threads per block

template<int block_type>
__global__ void interpolation_v2_kernel(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation){
  // load current block
  blockCopy_type block = interpolation.blocks[block_type][blockIdx.x];

  // interpolate 3D array from read_i,j,k of read[] to write_i,j,k in write[]
  int write_dim_i   = block.dim.i<<1; // calculate the dimensions of the resultant fine block
  int write_dim_j   = block.dim.j<<1;
  int write_dim_k   = block.dim.k<<1;
  //if(threadIdx.x==0) printf("%d:\t(%d, %d, %d)\n",blockIdx.x,write_dim_i,write_dim_j,write_dim_k);

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
     read_jStride = level_c.my_boxes[block.read.box ].jStride;
     read_kStride = level_c.my_boxes[block.read.box ].kStride;
     read = level_c.my_boxes[ block.read.box].vectors[id_c] + level_c.my_boxes[ block.read.box].ghosts*(1+ read_jStride+ read_kStride);
  }
  if(block.write.box>=0){
    write_jStride = level_f.my_boxes[block.write.box].jStride;
    write_kStride = level_f.my_boxes[block.write.box].kStride;
    write = level_f.my_boxes[block.write.box].vectors[id_f] + level_f.my_boxes[block.write.box].ghosts*(1+write_jStride+write_kStride);
  }


//  #ifdef USE_NAIVE_INTERP
  // naive 27pt per fine grid cell
  double c1 = 1.0/8.0;
//  int i = threadIdx.x % write_dim_i;
//  int jBlockStride = blockDim.x / write_dim_i;
//  for(int j=threadIdx.x/write_dim_i; j<write_dim_j; j+=jBlockStride)
//  for(int k=0; k<write_dim_k; k++) {
  for(int gid=threadIdx.x; gid<write_dim_i*write_dim_j*write_dim_k; gid+=blockDim.x){
    int k=(gid/write_dim_i)/write_dim_j;
    int j=(gid/write_dim_i)%write_dim_j;
    int i=gid%write_dim_i;
    double c1i=c1;if(i&0x1){c1i=-c1;}
    double c1j=c1;if(j&0x1){c1j=-c1;}
    double c1k=c1;if(k&0x1){c1k=-c1;}
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    //
    // |  1/8  |  1.0  | -1/8  | coarse grid
    // |---+---|---+---|---+---|
    // |   |   |???|   |   |   | fine grid
    //

    write[write_ijk] = prescale_f*write[write_ijk] +
                       + c1k*( + c1j*( c1i*read[read_ijk-1-read_jStride-read_kStride] + read[read_ijk-read_jStride-read_kStride] - c1i*read[read_ijk+1-read_jStride-read_kStride] )
                               +     ( c1i*read[read_ijk-1             -read_kStride] + read[read_ijk             -read_kStride] - c1i*read[read_ijk+1             -read_kStride] )
                               - c1j*( c1i*read[read_ijk-1+read_jStride-read_kStride] + read[read_ijk+read_jStride-read_kStride] - c1i*read[read_ijk+1+read_jStride-read_kStride] ) )
                       +     ( + c1j*( c1i*read[read_ijk-1-read_jStride             ] + read[read_ijk-read_jStride             ] - c1i*read[read_ijk+1-read_jStride             ] )
                               +     ( c1i*read[read_ijk-1                          ] + read[read_ijk                          ] - c1i*read[read_ijk+1                          ] )
                               - c1j*( c1i*read[read_ijk-1+read_jStride             ] + read[read_ijk+read_jStride             ] - c1i*read[read_ijk+1+read_jStride             ] ) )
                       - c1k*( + c1j*( c1i*read[read_ijk-1-read_jStride+read_kStride] + read[read_ijk-read_jStride+read_kStride] - c1i*read[read_ijk+1-read_jStride+read_kStride] )
                               +     ( c1i*read[read_ijk-1             +read_kStride] + read[read_ijk             +read_kStride] - c1i*read[read_ijk+1             +read_kStride] )
                               - c1j*( c1i*read[read_ijk-1+read_jStride+read_kStride] + read[read_ijk+read_jStride+read_kStride] - c1i*read[read_ijk+1+read_jStride+read_kStride] ) );
  }
/*
  #else
  int i,j,k;
  double c1 = 1.0/8.0;
  for(k=0;k<write_dim_k;k+=2){
  for(j=0;j<write_dim_j;j+=2){
  for(i=0;i<write_dim_i;i+=2){
    int write_ijk = ((i   )+write_i) + (((j   )+write_j)*write_jStride) + (((k   )+write_k)*write_kStride);
    int  read_ijk = ((i>>1)+ read_i) + (((j>>1)+ read_j)* read_jStride) + (((k>>1)+ read_k)* read_kStride);
    //
    // |  1/8  |  1.0  | -1/8  | coarse grid
    // |---+---|---+---|---+---|
    // |   |   |???|   |   |   | fine grid
    //

    // grab all coarse grid points...
    const double c000=read[read_ijk-1-read_jStride-read_kStride], c100=read[read_ijk  -read_jStride-read_kStride], c200=read[read_ijk+1-read_jStride-read_kStride];
    const double c010=read[read_ijk-1             -read_kStride], c110=read[read_ijk               -read_kStride], c210=read[read_ijk+1             -read_kStride];
    const double c020=read[read_ijk-1+read_jStride-read_kStride], c120=read[read_ijk  +read_jStride-read_kStride], c220=read[read_ijk+1+read_jStride-read_kStride];
    const double c001=read[read_ijk-1-read_jStride             ], c101=read[read_ijk  -read_jStride             ], c201=read[read_ijk+1-read_jStride             ];
    const double c011=read[read_ijk-1                          ], c111=read[read_ijk                            ], c211=read[read_ijk+1                          ];
    const double c021=read[read_ijk-1+read_jStride             ], c121=read[read_ijk  +read_jStride             ], c221=read[read_ijk+1+read_jStride             ];
    const double c002=read[read_ijk-1-read_jStride+read_kStride], c102=read[read_ijk  -read_jStride+read_kStride], c202=read[read_ijk+1-read_jStride+read_kStride];
    const double c012=read[read_ijk-1             +read_kStride], c112=read[read_ijk               +read_kStride], c212=read[read_ijk+1             +read_kStride];
    const double c022=read[read_ijk-1+read_jStride+read_kStride], c122=read[read_ijk  +read_jStride+read_kStride], c222=read[read_ijk+1+read_jStride+read_kStride];

    // interpolate in i to create fine i / coarse jk points...
    //
    // +-------+-------+-------+      :.......+---+---+.......:
    // |       |       |       |      :       |   |   |       :
    // |   c   |   c   |   c   |      :       | f | f |       :
    // |       |       |       |      :       |   |   |       :
    // +-------+-------+-------+      :.......+---+---+.......:
    // |       |       |       |      :       |   |   |       :
    // |   c   |   c   |   c   |  ->  :       | f | f |       :
    // |       |       |       |      :       |   |   |       :
    // +-------+-------+-------+      :.......+---+---+.......:
    // |       |       |       |      :       |   |   |       :
    // |   c   |   c   |   c   |      :       | f | f |       :
    // |       |       |       |      :       |   |   |       :
    // +-------+-------+-------+      :.......+---+---+.......:
    //
    const double f0c00 = ( c100 + c1*(c000-c200) ); // same as original 3pt stencil... f0c00 = ( c1*c000 + c100 - c1*c200 );
    const double f1c00 = ( c100 - c1*(c000-c200) );
    const double f0c10 = ( c110 + c1*(c010-c210) );
    const double f1c10 = ( c110 - c1*(c010-c210) );
    const double f0c20 = ( c120 + c1*(c020-c220) );
    const double f1c20 = ( c120 - c1*(c020-c220) );

    const double f0c01 = ( c101 + c1*(c001-c201) );
    const double f1c01 = ( c101 - c1*(c001-c201) );
    const double f0c11 = ( c111 + c1*(c011-c211) );
    const double f1c11 = ( c111 - c1*(c011-c211) );
    const double f0c21 = ( c121 + c1*(c021-c221) );
    const double f1c21 = ( c121 - c1*(c021-c221) );

    const double f0c02 = ( c102 + c1*(c002-c202) );
    const double f1c02 = ( c102 - c1*(c002-c202) );
    const double f0c12 = ( c112 + c1*(c012-c212) );
    const double f1c12 = ( c112 - c1*(c012-c212) );
    const double f0c22 = ( c122 + c1*(c022-c222) );
    const double f1c22 = ( c122 - c1*(c022-c222) );

    // interpolate in j to create fine ij / coarse k points...
    //
    // :.......+---+---+.......:      :.......:.......:.......:
    // :       |   |   |       :      :       :       :       :
    // :       |   |   |       :      :       :       :       :
    // :       |   |   |       :      :       :       :       :
    // :.......+---+---+.......:      :.......+---+---+.......:
    // :       |   |   |       :      :       |   |   |       :
    // :       |   |   |       :  ->  :       +---+---+       :
    // :       |   |   |       :      :       |   |   |       :
    // :.......+---+---+.......:      :.......+---+---+.......:
    // :       |   |   |       :      :       :       :       :
    // :       |   |   |       :      :       :       :       :
    // :       |   |   |       :      :       :       :       :
    // :.......+---+---+.......:      :.......:.......:.......:
    //
    const double f00c0 = ( f0c10 + c1*(f0c00-f0c20) );
    const double f10c0 = ( f1c10 + c1*(f1c00-f1c20) );
    const double f01c0 = ( f0c10 - c1*(f0c00-f0c20) );
    const double f11c0 = ( f1c10 - c1*(f1c00-f1c20) );

    const double f00c1 = ( f0c11 + c1*(f0c01-f0c21) );
    const double f10c1 = ( f1c11 + c1*(f1c01-f1c21) );
    const double f01c1 = ( f0c11 - c1*(f0c01-f0c21) );
    const double f11c1 = ( f1c11 - c1*(f1c01-f1c21) );

    const double f00c2 = ( f0c12 + c1*(f0c02-f0c22) );
    const double f10c2 = ( f1c12 + c1*(f1c02-f1c22) );
    const double f01c2 = ( f0c12 - c1*(f0c02-f0c22) );
    const double f11c2 = ( f1c12 - c1*(f1c02-f1c22) );

    // interpolate in k to create fine ijk points...
    const double f000 = ( f00c1 + c1*(f00c0-f00c2) );
    const double f100 = ( f10c1 + c1*(f10c0-f10c2) );
    const double f010 = ( f01c1 + c1*(f01c0-f01c2) );
    const double f110 = ( f11c1 + c1*(f11c0-f11c2) );
    const double f001 = ( f00c1 - c1*(f00c0-f00c2) );
    const double f101 = ( f10c1 - c1*(f10c0-f10c2) );
    const double f011 = ( f01c1 - c1*(f01c0-f01c2) );
    const double f111 = ( f11c1 - c1*(f11c0-f11c2) );

    // commit to memory...
    write[write_ijk                              ] = prescale_f*write[write_ijk                              ] + f000;
    write[write_ijk+1                            ] = prescale_f*write[write_ijk+1                            ] + f100;
    write[write_ijk  +write_jStride              ] = prescale_f*write[write_ijk  +write_jStride              ] + f010;
    write[write_ijk+1+write_jStride              ] = prescale_f*write[write_ijk+1+write_jStride              ] + f110;
    write[write_ijk                +write_kStride] = prescale_f*write[write_ijk                +write_kStride] + f001;
    write[write_ijk+1              +write_kStride] = prescale_f*write[write_ijk+1              +write_kStride] + f101;
    write[write_ijk  +write_jStride+write_kStride] = prescale_f*write[write_ijk  +write_jStride+write_kStride] + f011;
    write[write_ijk+1+write_jStride+write_kStride] = prescale_f*write[write_ijk+1+write_jStride+write_kStride] + f111;
  }}}
  #endif
*/
}

extern "C"
void cuda_interpolation_v2(level_type level_f, int id_f, double prescale_f, level_type level_c, int id_c, communicator_type interpolation, int block_type)
{
  int block = INTERPOLATION_V2_BLOCK_SIZE;
  int grid = interpolation.num_blocks[block_type];

  if(grid<=0) return;

  switch(block_type){
    case 0: interpolation_v2_kernel<0><<<grid,block>>>(level_f,id_f,prescale_f,level_c,id_c,interpolation); break;
    case 1: interpolation_v2_kernel<1><<<grid,block>>>(level_f,id_f,prescale_f,level_c,id_c,interpolation); break;
  }
}
