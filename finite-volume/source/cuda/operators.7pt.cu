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
//------------------------------------------------------------------------------------------------------------------------------
// Nikolay Sakharnykh
// nsakharnykh@nvidia.com
// Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.
//------------------------------------------------------------------------------------------------------------------------------
// Samuel Williams
// SWWilliams@lbl.gov
// Lawrence Berkeley National Lab
//------------------------------------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
//------------------------------------------------------------------------------------------------------------------------------
#ifdef _OPENMP
#include <omp.h>
#endif
//------------------------------------------------------------------------------------------------------------------------------
#include "../timers.h"
#include "../defines.h"
#include "../level.h"
#include "../operators.h"
//------------------------------------------------------------------------------------------------------------------------------
#define STENCIL_VARIABLE_COEFFICIENT
//------------------------------------------------------------------------------------------------------------------------------
// below are stencil operators versions using different memory load options
//------------------------------------------------------------------------------------------------------------------------------
#ifndef CUDA_STENCIL_OPT_TEX
//------------------------------------------------------------------------------------------------------------------------------
// calculate Dinv?
#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / (a*alpha[ijk] - b*h2inv*(                             \
             + beta_i[ijk        ]*( valid[ijk-1      ] - 2.0 ) \
             + beta_j[ijk        ]*( valid[ijk-jStride] - 2.0 ) \
             + beta_k[ijk        ]*( valid[ijk-kStride] - 2.0 ) \
             + beta_i[ijk+1      ]*( valid[ijk+1      ] - 2.0 ) \
             + beta_j[ijk+jStride]*( valid[ijk+jStride] - 2.0 ) \
             + beta_k[ijk+kStride]*( valid[ijk+kStride] - 2.0 ) \
          ))                                                    \
  )
  #else // variable coefficient Poisson ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / ( -b*h2inv*(                                          \
             + beta_i[ijk        ]*( valid[ijk-1      ] - 2.0 ) \
             + beta_j[ijk        ]*( valid[ijk-jStride] - 2.0 ) \
             + beta_k[ijk        ]*( valid[ijk-kStride] - 2.0 ) \
             + beta_i[ijk+1      ]*( valid[ijk+1      ] - 2.0 ) \
             + beta_j[ijk+jStride]*( valid[ijk+jStride] - 2.0 ) \
             + beta_k[ijk+kStride]*( valid[ijk+kStride] - 2.0 ) \
          ))                                                    \
  )
  #endif
#else // constant coefficient case... 
  #define calculate_Dinv()          \
  (                                 \
    1.0 / (a - b*h2inv*(            \
             + valid[ijk-1      ]   \
             + valid[ijk-jStride]   \
             + valid[ijk-kStride]   \
             + valid[ijk+1      ]   \
             + valid[ijk+jStride]   \
             + valid[ijk+kStride]   \
             - 12.0                 \
          ))                        \
  )
#endif

#if defined(STENCIL_FUSE_DINV) && defined(STENCIL_FUSE_BC)
#define Dinv_ijk() calculate_Dinv() // recalculate it
#else
#define Dinv_ijk() Dinv[ijk]        // simply retrieve it rather than recalculating it
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      a*alpha[ijk]*x[ijk]                                                                     \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk        ]*( valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk        ]*( valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) - 2.0*x[ijk] ) \
        + beta_i[ijk+1      ]*( valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk+jStride]*( valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk+kStride]*( valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) - 2.0*x[ijk] ) \
      )                                                                                       \
    )
    #else // variable coefficient Poisson ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk        ]*( valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk        ]*( valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) - 2.0*x[ijk] ) \
        + beta_i[ijk+1      ]*( valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) - 2.0*x[ijk] ) \
        + beta_j[ijk+jStride]*( valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) - 2.0*x[ijk] ) \
        + beta_k[ijk+kStride]*( valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) - 2.0*x[ijk] ) \
      )                                                                                       \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)                                \
    (                                                    \
      a*x[ijk] - b*h2inv*(                               \
        + valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) \
        + valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) \
        + valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) \
        + valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) \
        + valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) \
        + valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) \
                       -12.0*( x[ijk]                  ) \
      )                                                  \
    )
  #endif // variable/constant coefficient

#endif

//------------------------------------------------------------------------------------------------------------------------------
#ifndef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz...
    #define apply_op_ijk(x)                               \
    (                                                     \
      a*alpha[ijk]*x[ijk]                                 \
     -b*h2inv*(                                           \
        + beta_i[ijk+1      ]*( x[ijk+1      ] - x[ijk] ) \
        + beta_i[ijk        ]*( x[ijk-1      ] - x[ijk] ) \
        + beta_j[ijk+jStride]*( x[ijk+jStride] - x[ijk] ) \
        + beta_j[ijk        ]*( x[ijk-jStride] - x[ijk] ) \
        + beta_k[ijk+kStride]*( x[ijk+kStride] - x[ijk] ) \
        + beta_k[ijk        ]*( x[ijk-kStride] - x[ijk] ) \
      )                                                   \
    )
    #else // variable coefficient Poisson...
    #define apply_op_ijk(x)                               \
    (                                                     \
      -b*h2inv*(                                          \
        + beta_i[ijk+1      ]*( x[ijk+1      ] - x[ijk] ) \
        + beta_i[ijk        ]*( x[ijk-1      ] - x[ijk] ) \
        + beta_j[ijk+jStride]*( x[ijk+jStride] - x[ijk] ) \
        + beta_j[ijk        ]*( x[ijk-jStride] - x[ijk] ) \
        + beta_k[ijk+kStride]*( x[ijk+kStride] - x[ijk] ) \
        + beta_k[ijk        ]*( x[ijk-kStride] - x[ijk] ) \
      )                                                   \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)            \
    (                                \
      a*x[ijk] - b*h2inv*(           \
        + x[ijk+1      ]             \
        + x[ijk-1      ]             \
        + x[ijk+jStride]             \
        + x[ijk-jStride]             \
        + x[ijk+kStride]             \
        + x[ijk-kStride]             \
        - x[ijk        ]*6.0         \
      )                              \
    )
  #endif // variable/constant coefficient

#endif // BCs
//------------------------------------------------------------------------------------------------------------------------------
#else  // stencil-opt-tex
//------------------------------------------------------------------------------------------------------------------------------
// calculate Dinv?
#ifdef STENCIL_VARIABLE_COEFFICIENT
  #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / (a*alpha[ijk] - b*h2inv*(                             \
             + beta_i[ijk        ]*( __ldg(valid + ijk-1      ) - 2.0 ) \
             + beta_j[ijk        ]*( __ldg(valid + ijk-jStride) - 2.0 ) \
             + beta_k_cur         *( valid_kprev                - 2.0 ) \
             + beta_i[ijk+1      ]*( __ldg(valid + ijk+1      ) - 2.0 ) \
             + beta_j[ijk+jStride]*( __ldg(valid + ijk+jStride) - 2.0 ) \
             + beta_k_next        *( valid_knext                - 2.0 ) \
          ))                                                    \
  )
  #else // variable coefficient Poisson ...
  // optimization: 
  //   use registers to cache valid along k dim for valid and beta_k
  //   use ldg to cache along i-j plane for valid
  #define calculate_Dinv()                                      \
  (                                                             \
    1.0 / ( -b*h2inv*(                                          \
             + beta_i[ijk        ]*( __ldg(valid + ijk-1      ) - 2.0 ) \
             + beta_j[ijk        ]*( __ldg(valid + ijk-jStride) - 2.0 ) \
             + beta_k_cur         *( valid_kprev                - 2.0 ) \
             + beta_i[ijk+1      ]*( __ldg(valid + ijk+1      ) - 2.0 ) \
             + beta_j[ijk+jStride]*( __ldg(valid + ijk+jStride) - 2.0 ) \
             + beta_k_next        *( valid_knext                - 2.0 ) \
          ))                                                    \
  )
  #endif
#else // constant coefficient case... 
  #define calculate_Dinv()          \
  (                                 \
    1.0 / (a - b*h2inv*(            \
             + valid[ijk-1      ]   \
             + valid[ijk-jStride]   \
             + valid[ijk-kStride]   \
             + valid[ijk+1      ]   \
             + valid[ijk+jStride]   \
             + valid[ijk+kStride]   \
             - 12.0                 \
          ))                        \
  )
#endif

#if defined(STENCIL_FUSE_DINV) && defined(STENCIL_FUSE_BC)
#define Dinv_ijk() calculate_Dinv() // recalculate it
#else
#define Dinv_ijk() Dinv[ijk]        // simply retriev it rather than recalculating it
#endif
//------------------------------------------------------------------------------------------------------------------------------
#ifdef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz ...
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      a*alpha[ijk]*x_cur                                                                      \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( __ldg(valid + ijk-1      )*( x_cur  + __ldg(x + ijk-1      ) ) - 2.0*x_cur ) \
        + beta_j[ijk        ]*( __ldg(valid + ijk-jStride)*( x_cur  + __ldg(x + ijk-jStride) ) - 2.0*x_cur ) \
        + beta_k_cur         *( valid_kprev               *( x_cur  + x_kprev                ) - 2.0*x_cur ) \
        + beta_i[ijk+1      ]*( __ldg(valid + ijk+1      )*( x_cur  + __ldg(x + ijk+1      ) ) - 2.0*x_cur ) \
        + beta_j[ijk+jStride]*( __ldg(valid + ijk+jStride)*( x_cur  + __ldg(x + ijk+jStride) ) - 2.0*x_cur ) \
        + beta_k_next        *( valid_knext               *( x_cur  + x_knext                ) - 2.0*x_cur ) \
      )                                                                                       \
    )
    #else // variable coefficient Poisson ...
    // optimizations: 
    //   use registers to cache points along k dim for x, valid and beta_k
    //   use ldg to cache along i-j plane for x and valid
    #define apply_op_ijk(x)                                                                   \
    (                                                                                         \
      -b*h2inv*(                                                                              \
        + beta_i[ijk        ]*( __ldg(valid + ijk-1      )*( x_cur  + __ldg(x + ijk-1      ) ) - 2.0*x_cur ) \
        + beta_j[ijk        ]*( __ldg(valid + ijk-jStride)*( x_cur  + __ldg(x + ijk-jStride) ) - 2.0*x_cur ) \
        + beta_k_cur         *( valid_kprev               *( x_cur  + x_kprev                ) - 2.0*x_cur ) \
        + beta_i[ijk+1      ]*( __ldg(valid + ijk+1      )*( x_cur  + __ldg(x + ijk+1      ) ) - 2.0*x_cur ) \
        + beta_j[ijk+jStride]*( __ldg(valid + ijk+jStride)*( x_cur  + __ldg(x + ijk+jStride) ) - 2.0*x_cur ) \
        + beta_k_next        *( valid_knext               *( x_cur  + x_knext                ) - 2.0*x_cur ) \
      )                                                                                       \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)                                \
    (                                                    \
      a*x[ijk] - b*h2inv*(                               \
        + valid[ijk-1      ]*( x[ijk] + x[ijk-1      ] ) \
        + valid[ijk-jStride]*( x[ijk] + x[ijk-jStride] ) \
        + valid[ijk-kStride]*( x[ijk] + x[ijk-kStride] ) \
        + valid[ijk+1      ]*( x[ijk] + x[ijk+1      ] ) \
        + valid[ijk+jStride]*( x[ijk] + x[ijk+jStride] ) \
        + valid[ijk+kStride]*( x[ijk] + x[ijk+kStride] ) \
                       -12.0*( x[ijk]                  ) \
      )                                                  \
    )
  #endif // variable/constant coefficient

#endif

//------------------------------------------------------------------------------------------------------------------------------
#ifndef STENCIL_FUSE_BC

  #ifdef STENCIL_VARIABLE_COEFFICIENT
    #ifdef USE_HELMHOLTZ // variable coefficient Helmholtz...
    #define apply_op_ijk(x)                               \
    (                                                     \
      a*alpha[ijk]*x_cur                                  \
     -b*h2inv*(                                           \
        + beta_i[ijk+1      ]*( __ldg(x + ijk+1      ) - x_cur ) \
        + beta_i[ijk        ]*( __ldg(x + ijk-1      ) - x_cur ) \
        + beta_j[ijk+jStride]*( __ldg(x + ijk+jStride) - x_cur ) \
        + beta_j[ijk        ]*( __ldg(x + ijk-jStride) - x_cur ) \
        + beta_k_next        *(                x_knext - x_cur ) \
        + beta_k_cur         *(                x_kprev - x_cur ) \
      )                                                   \
    )
    #else // variable coefficient Poisson...
    // optimizations: 
    //   use registers to cache points along k dim for x and beta_k
    //   use ldg to cache along i-j plane for x
    #define apply_op_ijk(x)                               \
    (                                                     \
      -b*h2inv*(                                          \
        + beta_i[ijk+1      ]*( __ldg(x + ijk+1      ) - x_cur ) \
        + beta_i[ijk        ]*( __ldg(x + ijk-1      ) - x_cur ) \
        + beta_j[ijk+jStride]*( __ldg(x + ijk+jStride) - x_cur ) \
        + beta_j[ijk        ]*( __ldg(x + ijk-jStride) - x_cur ) \
        + beta_k_next        *(                x_knext - x_cur ) \
        + beta_k_cur         *(                x_kprev - x_cur ) \
      )                                                   \
    )
    #endif
  #else  // constant coefficient case...  
    #define apply_op_ijk(x)            \
    (                                \
      a*x[ijk] - b*h2inv*(           \
        + x[ijk+1      ]             \
        + x[ijk-1      ]             \
        + x[ijk+jStride]             \
        + x[ijk-jStride]             \
        + x[ijk+kStride]             \
        + x[ijk-kStride]             \
        - x[ijk        ]*6.0         \
      )                              \
    )
  #endif // variable/constant coefficient

#endif // BCs
#endif // stencil-opt-tex

//------------------------------------------------------------------------------------------------------------------------------
#ifdef  USE_GSRB
#define NUM_SMOOTHS      2 // RBRB
#include "gsrb.h"
#elif   USE_CHEBY
#define NUM_SMOOTHS      1
#define CHEBYSHEV_DEGREE 4 // i.e. one degree-4 polynomial smoother
#include "chebyshev.h"
#else
#error You must compile CUDA code with either -DUSE_GSRB or -DUSE_CHEBY, other smoothers are not currently supported
#endif
//------------------------------------------------------------------------------------------------------------------------------
#include "residual.h"
#include "blockCopy.h"
#include "misc.h"
#include "boundary_fd.h"
#include "restriction.h"
#include "interpolation.h"
//------------------------------------------------------------------------------------------------------------------------------
