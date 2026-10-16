#include <tl_templates/cuda/instruction/mma.h>
#include <math_constants.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float cu_local_cast[4];
  float out[16];
  float maximum[2];
  float denominator[2];
  float out_1[32];
  float maximum_1[4];
  float denominator_1[4];
  float scores[8];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  bfloat16_t probs_local_cast_1[2];
  float scores_1[16];
  float previous_1[4];
  float maximum_clear_1[4];
  float factor_1[4];
  float subtotal_1[4];
  if (((int)threadIdx.x) < 24) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 24)] = -CUDART_INF_F;
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i = 0; i < 6; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu_local_cast + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    *(float4*)(((float*)buf_dyn_shmem) + (((i * 512) + (((int)threadIdx.x) * 4)) + 48)) = *(float4*)(cu_local_cast + 0);
  }
  __syncthreads();
  for (int r = 0; r < 7; ++r) {
    int shard_begin = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
    int shard_end = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
    if (shard_begin < shard_end) {
      if (Count[((((int)blockIdx.x) * 7) + r)] <= 16) {
        int begin = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
        int end = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
        #pragma unroll
        for (int i_1 = 0; i_1 < 2; ++i_1) {
          int src = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_1 * 8)) + (((int)threadIdx.x) >> 4))];
          bool valid = (((i_1 * 8) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 7) + r)]);
          bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval;
          if (((valid && (0 <= src)) && (src < 24))) {
            condval = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (((int64_t)src) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_1 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22624)) = condval;
        }
        __syncthreads();
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; ++i_2) {
          int src_1 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + ((i_2 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_1 = ((((i_2 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          float2 condval_1;
          if (valid_1) {
            condval_1 = *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48));
          } else {
            condval_1 = make_float2(broadcast_var_2, broadcast_var_2);
          }
          *(float2*)(out + (i_2 * 2)) = condval_1;
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 2; ++i_3) {
          int src_2 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_3 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_2 = (((i_3 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          float condval_2;
          if (valid_2) {
            condval_2 = ((float*)buf_dyn_shmem)[(((int64_t)src_2) + (int64_t)24)];
          } else {
            condval_2 = -CUDART_INF_F;
          }
          maximum[i_3] = condval_2;
          float condval_3;
          if (valid_2) {
            condval_3 = ((float*)buf_dyn_shmem)[((int64_t)src_2)];
          } else {
            condval_3 = 0x0p+0f/*0.000000e+00*/;
          }
          denominator[i_3] = condval_3;
        }
        for (int tile = 0; tile < (((end + 63) - begin) >> 6); ++tile) {
          __syncthreads();
          #pragma unroll
          for (int i_4 = 0; i_4 < 8; ++i_4) {
            int condval_4;
            if ((0 <= ((((tile * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin))) {
              condval_4 = Indices[((((int)blockIdx.x) * 87083) + min(((((tile * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin), 87082))];
            } else {
              condval_4 = 0;
            }
            int token = condval_4;
            bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_5;
            if ((((((((tile * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 87083))) {
              condval_5 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)11146624) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_5 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)) = condval_5;
            bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_6;
            if ((((((((tile * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 87083))) {
              condval_6 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)11146624) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_4 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 14432)) = condval_6;
          }
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores + (i_5 * 4)) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
          }
          {
            bfloat16_t A_local[8];
            bfloat16_t B_local[8];
            __syncthreads();
            for (int ki = 0; ki < 8; ++ki) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 22624)])), (&(A_local[0])));
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)])), (&(B_local[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 4), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 4));
            }
          }
          for (int i_s = 0; i_s < 8; ++i_s) {
            bool valid_3 = (((((i_s & 3) >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
            float condval_7;
            if ((valid_3 && (((((((tile * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((i_s >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_s & 1)) < end))) {
              condval_7 = (scores[i_s] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_7 = -CUDART_INF_F;
            }
            scores[i_s] = condval_7;
          }
          *(float2*)(previous + 0) = *(float2*)(maximum + 0);
          __syncthreads();
          #pragma unroll
          for (int i_6 = 0; i_6 < 2; ++i_6) {
            maximum_clear[i_6] = -CUDART_INF_F;
            #pragma unroll
            for (int rv = 0; rv < 4; ++rv) {
              maximum_clear[i_6] = max(maximum_clear[i_6], scores[((((rv & 1) * 4) + (i_6 * 2)) + (rv >> 1))]);
            }
            maximum_clear[i_6] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_6], (&(((float*)buf_dyn_shmem)[14768])));
            maximum_clear[i_6] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear[i_6]);
            maximum[i_6] = max(maximum[i_6], maximum_clear[i_6]);
          }
          #pragma unroll
          for (int i_7 = 0; i_7 < 2; ++i_7) {
            float condval_8;
            if ((previous[i_7] == -CUDART_INF_F)) {
              condval_8 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_8 = exp2f(((previous[i_7] - maximum[i_7]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor[i_7] = condval_8;
          }
          #pragma unroll
          for (int i_8 = 0; i_8 < 8; ++i_8) {
            float condval_9;
            if ((scores[i_8] == -CUDART_INF_F)) {
              condval_9 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_9 = exp2f(((scores[i_8] - maximum[((i_8 & 3) >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores[i_8] = condval_9;
          }
          __syncthreads();
          #pragma unroll
          for (int i_9 = 0; i_9 < 2; ++i_9) {
            subtotal[i_9] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_1 = 0; rv_1 < 4; ++rv_1) {
              subtotal[i_9] = (subtotal[i_9] + scores[((((rv_1 & 1) * 4) + (i_9 * 2)) + (rv_1 >> 1))]);
            }
            subtotal[i_9] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal[i_9], (&(((float*)buf_dyn_shmem)[14640])));
            subtotal[i_9] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal[i_9]);
          }
          #pragma unroll
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            denominator[i_10] = ((denominator[i_10] * factor[i_10]) + subtotal[i_10]);
          }
          #pragma unroll
          for (int i_11 = 0; i_11 < 16; ++i_11) {
            out[i_11] = (out[i_11] * factor[((i_11 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_12 = 0; i_12 < 4; ++i_12) {
            uint1 __1;
            float2 v_ = *(float2*)(scores + (i_12 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
            *(uint1*)(probs_local_cast_1 + 0) = __1;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((i_12 & 1) * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + (((((((((int)threadIdx.x) >> 5) * 16) + ((i_12 >> 1) * 8)) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 7) >> 2) + (i_12 >> 1)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 26720)) = *(uint1*)(probs_local_cast_1 + 0);
          }
          {
            bfloat16_t A_local_1[8];
            bfloat16_t B_local_1[16];
            __syncthreads();
            for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 26720)])), (&(A_local_1[0])));
              for (int i_13 = 0; i_13 < 2; ++i_13) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_13) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 14432)])), (&(B_local_1[(i_13 * 8)])));
              }
              for (int j = 0; j < 2; ++j) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + (j * 8)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + (j * 8)));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + ((j * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + ((j * 8) + 4)));
              }
            }
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i_14 = 0; i_14 < 8; ++i_14) {
          int src_3 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_4 = ((((i_14 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          if (valid_4) {
            *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_3) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + ((((int64_t)i_14) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48)) = *(float2*)(out + (i_14 * 2));
          }
        }
        __syncthreads();
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_15 = 0; i_15 < 2; ++i_15) {
            int src_4 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            bool valid_5 = (((i_15 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
            if (valid_5) {
              ((float*)buf_dyn_shmem)[(((int64_t)src_4) + (int64_t)24)] = maximum[i_15];
              ((float*)buf_dyn_shmem)[((int64_t)src_4)] = denominator[i_15];
            }
          }
        }
      } else {
        int begin_1 = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * ((int)blockIdx.y)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
        int end_1 = (((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) / 3) + ((((Offsets[(((((int)blockIdx.x) * 8) + r) + 1)] - Offsets[((((int)blockIdx.x) * 8) + r)]) * (((int)blockIdx.y) + 1)) % 3) >> 31)) + Offsets[((((int)blockIdx.x) * 8) + r)]);
        #pragma unroll
        for (int i_16 = 0; i_16 < 4; ++i_16) {
          int src_5 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_16 * 8)) + (((int)threadIdx.x) >> 4))];
          bool valid_6 = (((i_16 * 8) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 7) + r)]);
          bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_10;
          if (((valid_6 && (0 <= src_5)) && (src_5 < 24))) {
            condval_10 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (((int64_t)src_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_10 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_16 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22624)) = condval_10;
        }
        __syncthreads();
        #pragma unroll
        for (int i_17 = 0; i_17 < 16; ++i_17) {
          int src_6 = RowMap[(((((((int)blockIdx.x) * 224) + (r * 32)) + ((i_17 >> 3) * 16)) + ((i_17 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_7 = (((((i_17 >> 3) * 16) + ((i_17 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
          float2 condval_11;
          if (valid_7) {
            condval_11 = *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_6) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + (((((int64_t)i_17) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48));
          } else {
            condval_11 = make_float2(broadcast_var_7, broadcast_var_7);
          }
          *(float2*)(out_1 + (i_17 * 2)) = condval_11;
        }
        #pragma unroll
        for (int i_18 = 0; i_18 < 4; ++i_18) {
          int src_7 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_18 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_8 = (((i_18 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          float condval_12;
          if (valid_8) {
            condval_12 = ((float*)buf_dyn_shmem)[(((int64_t)src_7) + (int64_t)24)];
          } else {
            condval_12 = -CUDART_INF_F;
          }
          maximum_1[i_18] = condval_12;
          float condval_13;
          if (valid_8) {
            condval_13 = ((float*)buf_dyn_shmem)[((int64_t)src_7)];
          } else {
            condval_13 = 0x0p+0f/*0.000000e+00*/;
          }
          denominator_1[i_18] = condval_13;
        }
        for (int tile_1 = 0; tile_1 < (((end_1 + 63) - begin_1) >> 6); ++tile_1) {
          __syncthreads();
          #pragma unroll
          for (int i_19 = 0; i_19 < 8; ++i_19) {
            int condval_14;
            if ((0 <= ((((tile_1 * 64) + (i_19 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1))) {
              condval_14 = Indices[((((int)blockIdx.x) * 87083) + min(((((tile_1 * 64) + (i_19 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1), 87082))];
            } else {
              condval_14 = 0;
            }
            int token_1 = condval_14;
            bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_15;
            if ((((((((tile_1 * 64) + (i_19 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 87083))) {
              condval_15 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)11146624) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_15 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_19 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)) = condval_15;
            bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_16;
            if ((((((((tile_1 * 64) + (i_19 * 8)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 87083))) {
              condval_16 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)11146624) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_16 = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_19 * 512)) + ((((int)threadIdx.x) >> 4) * 64)) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 14432)) = condval_16;
          }
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_1 + (i_20 * 4)) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
          }
          {
            bfloat16_t A_local_2[16];
            bfloat16_t B_local_2[8];
            __syncthreads();
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              for (int i_21 = 0; i_21 < 2; ++i_21) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_2 >> 2) * 2048) + (i_21 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 22624)])), (&(A_local_2[(i_21 * 8)])));
              }
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_2 >> 2) * 4096) + ((((int)threadIdx.x) >> 4) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)])), (&(B_local_2[0])));
              for (int i_22 = 0; i_22 < 2; ++i_22) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + (i_22 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + ((i_22 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_22 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 4));
              }
            }
          }
          #pragma unroll
          for (int i_23 = 0; i_23 < 2; ++i_23) {
            for (int vec_s = 0; vec_s < 8; ++vec_s) {
              bool valid_9 = ((((i_23 * 16) + (((vec_s & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
              float condval_17;
              if ((valid_9 && (((((((tile_1 * 64) + ((((int)threadIdx.x) >> 5) * 16)) + ((vec_s >> 2) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_1) + (vec_s & 1)) < end_1))) {
                condval_17 = (scores_1[((i_23 * 8) + vec_s)] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
              } else {
                condval_17 = -CUDART_INF_F;
              }
              scores_1[((i_23 * 8) + vec_s)] = condval_17;
            }
          }
          *(float4*)(previous_1 + 0) = *(float4*)(maximum_1 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_24 = 0; i_24 < 4; ++i_24) {
            maximum_clear_1[i_24] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_2 = 0; rv_2 < 4; ++rv_2) {
              maximum_clear_1[i_24] = max(maximum_clear_1[i_24], scores_1[(((((i_24 >> 1) * 8) + ((rv_2 & 1) * 4)) + ((i_24 & 1) * 2)) + (rv_2 >> 1))]);
            }
            maximum_clear_1[i_24] = tl::AllReduce<tl::MaxOp, 128, 32, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_24], (&(((float*)buf_dyn_shmem)[14512])));
            maximum_clear_1[i_24] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<128>>::run(maximum_clear_1[i_24]);
            maximum_1[i_24] = max(maximum_1[i_24], maximum_clear_1[i_24]);
          }
          #pragma unroll
          for (int i_25 = 0; i_25 < 4; ++i_25) {
            float condval_18;
            if ((previous_1[i_25] == -CUDART_INF_F)) {
              condval_18 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_18 = exp2f(((previous_1[i_25] - maximum_1[i_25]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_1[i_25] = condval_18;
          }
          #pragma unroll
          for (int i_26 = 0; i_26 < 16; ++i_26) {
            float condval_19;
            if ((scores_1[i_26] == -CUDART_INF_F)) {
              condval_19 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_19 = exp2f(((scores_1[i_26] - maximum_1[(((i_26 >> 3) * 2) + ((i_26 & 3) >> 1))]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_1[i_26] = condval_19;
          }
          __syncthreads();
          #pragma unroll
          for (int i_27 = 0; i_27 < 4; ++i_27) {
            subtotal_1[i_27] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_3 = 0; rv_3 < 4; ++rv_3) {
              subtotal_1[i_27] = (subtotal_1[i_27] + scores_1[(((((i_27 >> 1) * 8) + ((rv_3 & 1) * 4)) + ((i_27 & 1) * 2)) + (rv_3 >> 1))]);
            }
            subtotal_1[i_27] = tl::AllReduce<tl::SumOp, 128, 32, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_27], (&(((float*)buf_dyn_shmem)[14384])));
            subtotal_1[i_27] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<128>>::run(subtotal_1[i_27]);
          }
          #pragma unroll
          for (int i_28 = 0; i_28 < 4; ++i_28) {
            denominator_1[i_28] = ((denominator_1[i_28] * factor_1[i_28]) + subtotal_1[i_28]);
          }
          #pragma unroll
          for (int i_29 = 0; i_29 < 32; ++i_29) {
            out_1[i_29] = (out_1[i_29] * factor_1[(((i_29 >> 4) * 2) + ((i_29 & 3) >> 1))]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_30 = 0; i_30 < 2; ++i_30) {
            tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_30 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + ((((((int)threadIdx.x) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 26720)])), __pack_half2(((bfloat16_t)scores_1[(i_30 * 8)]), ((bfloat16_t)scores_1[((i_30 * 8) + 1)])), __pack_half2(((bfloat16_t)scores_1[((i_30 * 8) + 2)]), ((bfloat16_t)scores_1[((i_30 * 8) + 3)])), __pack_half2(((bfloat16_t)scores_1[((i_30 * 8) + 4)]), ((bfloat16_t)scores_1[((i_30 * 8) + 5)])), __pack_half2(((bfloat16_t)scores_1[((i_30 * 8) + 6)]), ((bfloat16_t)scores_1[((i_30 * 8) + 7)])));
          }
          {
            bfloat16_t A_local_3[16];
            bfloat16_t B_local_3[16];
            __syncthreads();
            for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
              for (int i_31 = 0; i_31 < 2; ++i_31) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_31 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 26720)])), (&(A_local_3[(i_31 * 8)])));
              }
              for (int i_32 = 0; i_32 < 2; ++i_32) {
                tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 6) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + i_32) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 14432)])), (&(B_local_3[(i_32 * 8)])));
              }
              for (int i_33 = 0; i_33 < 2; ++i_33) {
                for (int j_1 = 0; j_1 < 2; ++j_1) {
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + ((i_33 * 16) + (j_1 * 8))), reinterpret_cast<const unsigned*>(A_local_3 + (i_33 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + (j_1 * 8)));
                  tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + (((i_33 * 16) + (j_1 * 8)) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_33 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + ((j_1 * 8) + 4)));
                }
              }
            }
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i_34 = 0; i_34 < 16; ++i_34) {
          int src_8 = RowMap[(((((((int)blockIdx.x) * 224) + (r * 32)) + ((i_34 >> 3) * 16)) + ((i_34 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_10 = (((((i_34 >> 3) * 16) + ((i_34 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
          if (valid_10) {
            *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_8) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)32)) + (((((int64_t)i_34) & (int64_t)7) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48)) = *(float2*)(out_1 + (i_34 * 2));
          }
        }
        __syncthreads();
        if ((((((int)threadIdx.x) & 3) * 4) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_35 = 0; i_35 < 4; ++i_35) {
            int src_9 = RowMap[((((((int)blockIdx.x) * 224) + (r * 32)) + (i_35 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            bool valid_11 = (((i_35 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 7) + r)]);
            if (valid_11) {
              ((float*)buf_dyn_shmem)[(((int64_t)src_9) + (int64_t)24)] = maximum_1[i_35];
              ((float*)buf_dyn_shmem)[((int64_t)src_9)] = denominator_1[i_35];
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_36 = 0; i_36 < 6; ++i_36) {
    *(float4*)(U + ((((((int)blockIdx.x) * 9216) + (((int)blockIdx.y) * 3072)) + (i_36 * 512)) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_36 * 512) + (((int)threadIdx.x) * 4)) + 48));
  }
  if (tl::tl_shuffle_elect<128>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Mout[((((int)blockIdx.x) * 72) + (((int)blockIdx.y) * 24))])), (&(((float*)buf_dyn_shmem)[24])), 96);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
    tl::tma_store((&(Lout[((((int)blockIdx.x) * 72) + (((int)blockIdx.y) * 24))])), (&(((float*)buf_dyn_shmem)[0])), 96);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
}

