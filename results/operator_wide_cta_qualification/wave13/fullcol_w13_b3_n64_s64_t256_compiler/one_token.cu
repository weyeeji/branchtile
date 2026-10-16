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
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(const int* __restrict__ Count, const int* __restrict__ Indices, const bfloat16_t* __restrict__ K, float* __restrict__ Lout, float* __restrict__ Mout, const int* __restrict__ Offsets, const bfloat16_t* __restrict__ Q, const int* __restrict__ RowMap, float* __restrict__ U, const bfloat16_t* __restrict__ V) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float cu_local_cast[4];
  float out[8];
  float maximum[2];
  float denominator[2];
  float out_1[16];
  float maximum_1[4];
  float denominator_1[4];
  float scores[4];
  float previous[2];
  float maximum_clear[2];
  float factor[2];
  float subtotal[2];
  bfloat16_t probs_local_cast_1[2];
  float scores_1[8];
  float previous_1[4];
  float maximum_clear_1[4];
  float factor_1[4];
  float subtotal_1[4];
  if (((int)threadIdx.x) < 24) {
    ((float*)buf_dyn_shmem)[(((int)threadIdx.x) + 24)] = -CUDART_INF_F;
    ((float*)buf_dyn_shmem)[((int)threadIdx.x)] = 0x0p+0f/*0.000000e+00*/;
  }
  #pragma unroll
  for (int i = 0; i < 3; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(cu_local_cast + 0) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
    *(float4*)(((float*)buf_dyn_shmem) + (((i * 1024) + (((int)threadIdx.x) * 4)) + 48)) = *(float4*)(cu_local_cast + 0);
  }
  __syncthreads();
  for (int r = 0; r < 2; ++r) {
    int shard_begin = ((((Offsets[(((((int)blockIdx.x) * 3) + r) + 1)] - Offsets[((((int)blockIdx.x) * 3) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 3) + r)]);
    int shard_end = ((((Offsets[(((((int)blockIdx.x) * 3) + r) + 1)] - Offsets[((((int)blockIdx.x) * 3) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 3) + r)]);
    if (shard_begin < shard_end) {
      if (Count[((((int)blockIdx.x) * 2) + r)] <= 16) {
        int begin = ((((Offsets[(((((int)blockIdx.x) * 3) + r) + 1)] - Offsets[((((int)blockIdx.x) * 3) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 3) + r)]);
        int end = ((((Offsets[(((((int)blockIdx.x) * 3) + r) + 1)] - Offsets[((((int)blockIdx.x) * 3) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 3) + r)]);
        int src = RowMap[(((((int)blockIdx.x) * 64) + (r * 32)) + (((int)threadIdx.x) >> 4))];
        bool valid = ((((int)threadIdx.x) >> 4) < Count[((((int)blockIdx.x) * 2) + r)]);
        bfloat16_t broadcast_var_1 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
        uint4 condval;
        if (((valid && (0 <= src)) && (src < 24))) {
          condval = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (((int64_t)src) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
        } else {
          condval = make_uint4(__pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1), __pack_nv_bfloat162(broadcast_var_1, broadcast_var_1));
        }
        *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22624)) = condval;
        __syncthreads();
        #pragma unroll
        for (int i_1 = 0; i_1 < 4; ++i_1) {
          int src_1 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + ((i_1 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_1 = ((((i_1 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
          float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
          float2 condval_1;
          if (valid_1) {
            condval_1 = *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_1) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_1) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48));
          } else {
            condval_1 = make_float2(broadcast_var_2, broadcast_var_2);
          }
          *(float2*)(out + (i_1 * 2)) = condval_1;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 2; ++i_2) {
          int src_2 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + (i_2 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_2 = (((i_2 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
          float condval_2;
          if (valid_2) {
            condval_2 = ((float*)buf_dyn_shmem)[(((int64_t)src_2) + (int64_t)24)];
          } else {
            condval_2 = -CUDART_INF_F;
          }
          maximum[i_2] = condval_2;
          float condval_3;
          if (valid_2) {
            condval_3 = ((float*)buf_dyn_shmem)[((int64_t)src_2)];
          } else {
            condval_3 = 0x0p+0f/*0.000000e+00*/;
          }
          denominator[i_2] = condval_3;
        }
        for (int tile = 0; tile < (((end + 63) - begin) >> 6); ++tile) {
          __syncthreads();
          #pragma unroll
          for (int i_3 = 0; i_3 < 4; ++i_3) {
            int condval_4;
            if ((0 <= ((((tile * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 4)) + begin))) {
              condval_4 = Indices[((((int)blockIdx.x) * 259) + min(((((tile * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 4)) + begin), 258))];
            } else {
              condval_4 = 0;
            }
            int token = condval_4;
            bfloat16_t broadcast_var_3 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_5;
            if ((((((((tile * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
              condval_5 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_5 = make_uint4(__pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3), __pack_nv_bfloat162(broadcast_var_3, broadcast_var_3));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)) = condval_5;
            bfloat16_t broadcast_var_4 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_6;
            if ((((((((tile * 64) + (i_3 * 16)) + (((int)threadIdx.x) >> 4)) + begin) < end) && (0 <= token)) && (token < 259))) {
              condval_6 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_6 = make_uint4(__pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4), __pack_nv_bfloat162(broadcast_var_4, broadcast_var_4));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 14432)) = condval_6;
          }
          float broadcast_var_5 = 0x0p+0f/*0.000000e+00*/;
          *(float4*)(scores + 0) = make_float4(broadcast_var_5, broadcast_var_5, broadcast_var_5, broadcast_var_5);
          {
            bfloat16_t A_local[8];
            bfloat16_t B_local[4];
            __syncthreads();
            for (int ki = 0; ki < 8; ++ki) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((ki >> 2) * 2048) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 22624)])), (&(A_local[0])));
              tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)])), (&(B_local[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores + 0), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + 0));
            }
          }
          for (int i_s = 0; i_s < 4; ++i_s) {
            bool valid_3 = ((((i_s >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
            float condval_7;
            if ((valid_3 && ((((((tile * 64) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin) + (i_s & 1)) < end))) {
              condval_7 = (scores[i_s] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_7 = -CUDART_INF_F;
            }
            scores[i_s] = condval_7;
          }
          *(float2*)(previous + 0) = *(float2*)(maximum + 0);
          __syncthreads();
          #pragma unroll
          for (int i_4 = 0; i_4 < 2; ++i_4) {
            maximum_clear[i_4] = -CUDART_INF_F;
            #pragma unroll
            for (int rv = 0; rv < 2; ++rv) {
              maximum_clear[i_4] = max(maximum_clear[i_4], scores[((i_4 * 2) + rv)]);
            }
            maximum_clear[i_4] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(maximum_clear[i_4], (&(((float*)buf_dyn_shmem)[14896])));
            maximum_clear[i_4] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(maximum_clear[i_4]);
            maximum[i_4] = max(maximum[i_4], maximum_clear[i_4]);
          }
          #pragma unroll
          for (int i_5 = 0; i_5 < 2; ++i_5) {
            float condval_8;
            if ((previous[i_5] == -CUDART_INF_F)) {
              condval_8 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_8 = exp2f(((previous[i_5] - maximum[i_5]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor[i_5] = condval_8;
          }
          #pragma unroll
          for (int i_6 = 0; i_6 < 4; ++i_6) {
            float condval_9;
            if ((scores[i_6] == -CUDART_INF_F)) {
              condval_9 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_9 = exp2f(((scores[i_6] - maximum[(i_6 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores[i_6] = condval_9;
          }
          __syncthreads();
          #pragma unroll
          for (int i_7 = 0; i_7 < 2; ++i_7) {
            subtotal[i_7] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_1 = 0; rv_1 < 2; ++rv_1) {
              subtotal[i_7] = (subtotal[i_7] + scores[((i_7 * 2) + rv_1)]);
            }
            subtotal[i_7] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(subtotal[i_7], (&(((float*)buf_dyn_shmem)[15152])));
            subtotal[i_7] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(subtotal[i_7]);
          }
          #pragma unroll
          for (int i_8 = 0; i_8 < 2; ++i_8) {
            denominator[i_8] = ((denominator[i_8] * factor[i_8]) + subtotal[i_8]);
          }
          #pragma unroll
          for (int i_9 = 0; i_9 < 8; ++i_9) {
            out[i_9] = (out[i_9] * factor[((i_9 & 3) >> 1)]);
          }
          __syncthreads();
          #pragma unroll
          for (int i_10 = 0; i_10 < 2; ++i_10) {
            uint1 __1;
            float2 v_ = *(float2*)(scores + (i_10 * 2));
            (reinterpret_cast<__nv_bfloat162*>(&__1))[0] = __float22bfloat162_rn(((float2*)(&v_))[0]);
            *(uint1*)(probs_local_cast_1 + 0) = __1;
            *(uint1*)(((bfloat16_t*)buf_dyn_shmem) + (((((((i_10 * 512) + (((((int)threadIdx.x) & 31) >> 2) * 64)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 15) >> 3)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + 26720)) = *(uint1*)(probs_local_cast_1 + 0);
          }
          {
            bfloat16_t A_local_1[8];
            bfloat16_t B_local_1[8];
            __syncthreads();
            for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
              tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 15) >> 3) * 512) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 26720)])), (&(A_local_1[0])));
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 7) * 4096) + (ki_1 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 14432)])), (&(B_local_1[0])));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + 0), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 0));
              tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out + 4), reinterpret_cast<const unsigned*>(A_local_1 + 0), reinterpret_cast<const unsigned*>(B_local_1 + 4));
            }
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i_11 = 0; i_11 < 4; ++i_11) {
          int src_3 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + ((i_11 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_4 = ((((i_11 & 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
          if (valid_4) {
            *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_3) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + ((((int64_t)i_11) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48)) = *(float2*)(out + (i_11 * 2));
          }
        }
        __syncthreads();
        if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_12 = 0; i_12 < 2; ++i_12) {
            int src_4 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + (i_12 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            bool valid_5 = (((i_12 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
            if (valid_5) {
              ((float*)buf_dyn_shmem)[(((int64_t)src_4) + (int64_t)24)] = maximum[i_12];
              ((float*)buf_dyn_shmem)[((int64_t)src_4)] = denominator[i_12];
            }
          }
        }
      } else {
        int begin_1 = ((((Offsets[(((((int)blockIdx.x) * 3) + r) + 1)] - Offsets[((((int)blockIdx.x) * 3) + r)]) * ((int)blockIdx.y)) >> 6) + Offsets[((((int)blockIdx.x) * 3) + r)]);
        int end_1 = ((((Offsets[(((((int)blockIdx.x) * 3) + r) + 1)] - Offsets[((((int)blockIdx.x) * 3) + r)]) * (((int)blockIdx.y) + 1)) >> 6) + Offsets[((((int)blockIdx.x) * 3) + r)]);
        #pragma unroll
        for (int i_13 = 0; i_13 < 2; ++i_13) {
          int src_5 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + (i_13 * 16)) + (((int)threadIdx.x) >> 4))];
          bool valid_6 = (((i_13 * 16) + (((int)threadIdx.x) >> 4)) < Count[((((int)blockIdx.x) * 2) + r)]);
          bfloat16_t broadcast_var_6 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_10;
          if (((valid_6 && (0 <= src_5)) && (src_5 < 24))) {
            condval_10 = *(uint4*)(Q + (((((int64_t)((int)blockIdx.x)) * (int64_t)3072) + (((int64_t)src_5) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_10 = make_uint4(__pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6), __pack_nv_bfloat162(broadcast_var_6, broadcast_var_6));
          }
          *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 2048) + (i_13 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 22624)) = condval_10;
        }
        __syncthreads();
        #pragma unroll
        for (int i_14 = 0; i_14 < 8; ++i_14) {
          int src_6 = RowMap[(((((((int)blockIdx.x) * 64) + (r * 32)) + ((i_14 >> 2) * 16)) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_7 = (((((i_14 >> 2) * 16) + ((i_14 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
          float broadcast_var_7 = 0x0p+0f/*0.000000e+00*/;
          float2 condval_11;
          if (valid_7) {
            condval_11 = *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_6) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_14) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48));
          } else {
            condval_11 = make_float2(broadcast_var_7, broadcast_var_7);
          }
          *(float2*)(out_1 + (i_14 * 2)) = condval_11;
        }
        #pragma unroll
        for (int i_15 = 0; i_15 < 4; ++i_15) {
          int src_7 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + (i_15 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_8 = (((i_15 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
          float condval_12;
          if (valid_8) {
            condval_12 = ((float*)buf_dyn_shmem)[(((int64_t)src_7) + (int64_t)24)];
          } else {
            condval_12 = -CUDART_INF_F;
          }
          maximum_1[i_15] = condval_12;
          float condval_13;
          if (valid_8) {
            condval_13 = ((float*)buf_dyn_shmem)[((int64_t)src_7)];
          } else {
            condval_13 = 0x0p+0f/*0.000000e+00*/;
          }
          denominator_1[i_15] = condval_13;
        }
        for (int tile_1 = 0; tile_1 < (((end_1 + 63) - begin_1) >> 6); ++tile_1) {
          __syncthreads();
          #pragma unroll
          for (int i_16 = 0; i_16 < 4; ++i_16) {
            int condval_14;
            if ((0 <= ((((tile_1 * 64) + (i_16 * 16)) + (((int)threadIdx.x) >> 4)) + begin_1))) {
              condval_14 = Indices[((((int)blockIdx.x) * 259) + min(((((tile_1 * 64) + (i_16 * 16)) + (((int)threadIdx.x) >> 4)) + begin_1), 258))];
            } else {
              condval_14 = 0;
            }
            int token_1 = condval_14;
            bfloat16_t broadcast_var_8 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_15;
            if ((((((((tile_1 * 64) + (i_16 * 16)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 259))) {
              condval_15 = *(uint4*)(K + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_15 = make_uint4(__pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8), __pack_nv_bfloat162(broadcast_var_8, broadcast_var_8));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_16 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)) = condval_15;
            bfloat16_t broadcast_var_9 = bfloat16_t(0x0p+0f/*0.000000e+00*/);
            uint4 condval_16;
            if ((((((((tile_1 * 64) + (i_16 * 16)) + (((int)threadIdx.x) >> 4)) + begin_1) < end_1) && (0 <= token_1)) && (token_1 < 259))) {
              condval_16 = *(uint4*)(V + (((((int64_t)((int)blockIdx.x)) * (int64_t)33152) + (((int64_t)token_1) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
            } else {
              condval_16 = make_uint4(__pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9), __pack_nv_bfloat162(broadcast_var_9, broadcast_var_9));
            }
            *(uint4*)(((bfloat16_t*)buf_dyn_shmem) + (((((((((((int)threadIdx.x) & 15) >> 3) * 4096) + (i_16 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 14432)) = condval_16;
          }
          #pragma unroll
          for (int i_17 = 0; i_17 < 2; ++i_17) {
            float broadcast_var_10 = 0x0p+0f/*0.000000e+00*/;
            *(float4*)(scores_1 + (i_17 * 4)) = make_float4(broadcast_var_10, broadcast_var_10, broadcast_var_10, broadcast_var_10);
          }
          {
            bfloat16_t A_local_2[16];
            bfloat16_t B_local_2[4];
            __syncthreads();
            for (int ki_2 = 0; ki_2 < 8; ++ki_2) {
              for (int i_18 = 0; i_18 < 2; ++i_18) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((((ki_2 >> 2) * 2048) + (i_18 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 22624)])), (&(A_local_2[(i_18 * 8)])));
              }
              tl::ptx_ldmatrix_x2((&(((bfloat16_t*)buf_dyn_shmem)[((((((((ki_2 >> 2) * 4096) + ((((((int)threadIdx.x) >> 5) + ((((int)threadIdx.x) & 31) >> 4)) & 7) * 512)) + ((((int)threadIdx.x) & 7) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_2 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 6240)])), (&(B_local_2[0])));
              for (int i_19 = 0; i_19 < 2; ++i_19) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(scores_1 + (i_19 * 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_19 * 8)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
              }
            }
          }
          for (int i_s_1 = 0; i_s_1 < 8; ++i_s_1) {
            bool valid_9 = ((((i_s_1 >> 1) * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
            float condval_17;
            if ((valid_9 && ((((((tile_1 * 64) + ((((int)threadIdx.x) >> 5) * 8)) + ((((int)threadIdx.x) & 3) * 2)) + begin_1) + (i_s_1 & 1)) < end_1))) {
              condval_17 = (scores_1[i_s_1] * 0x1.6a09e667f3bccp-4f/*8.838835e-02*/);
            } else {
              condval_17 = -CUDART_INF_F;
            }
            scores_1[i_s_1] = condval_17;
          }
          *(float4*)(previous_1 + 0) = *(float4*)(maximum_1 + 0);
          __syncthreads();
          #pragma unroll
          for (int i_20 = 0; i_20 < 4; ++i_20) {
            maximum_clear_1[i_20] = -CUDART_INF_F;
            #pragma unroll
            for (int rv_2 = 0; rv_2 < 2; ++rv_2) {
              maximum_clear_1[i_20] = max(maximum_clear_1[i_20], scores_1[((i_20 * 2) + rv_2)]);
            }
            maximum_clear_1[i_20] = tl::AllReduce<tl::MaxOp, 256, 32, 0, tl::NamedBarrier<256>>::run(maximum_clear_1[i_20], (&(((float*)buf_dyn_shmem)[14640])));
            maximum_clear_1[i_20] = tl::AllReduce<tl::MaxOp, 4, 1, 0, tl::NamedBarrier<256>>::run(maximum_clear_1[i_20]);
            maximum_1[i_20] = max(maximum_1[i_20], maximum_clear_1[i_20]);
          }
          #pragma unroll
          for (int i_21 = 0; i_21 < 4; ++i_21) {
            float condval_18;
            if ((previous_1[i_21] == -CUDART_INF_F)) {
              condval_18 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_18 = exp2f(((previous_1[i_21] - maximum_1[i_21]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            factor_1[i_21] = condval_18;
          }
          #pragma unroll
          for (int i_22 = 0; i_22 < 8; ++i_22) {
            float condval_19;
            if ((scores_1[i_22] == -CUDART_INF_F)) {
              condval_19 = 0x0p+0f/*0.000000e+00*/;
            } else {
              condval_19 = exp2f(((scores_1[i_22] - maximum_1[(i_22 >> 1)]) * 0x1.71547652b82fep+0f/*1.442695e+00*/));
            }
            scores_1[i_22] = condval_19;
          }
          __syncthreads();
          #pragma unroll
          for (int i_23 = 0; i_23 < 4; ++i_23) {
            subtotal_1[i_23] = 0x0p+0f/*0.000000e+00*/;
            #pragma unroll
            for (int rv_3 = 0; rv_3 < 2; ++rv_3) {
              subtotal_1[i_23] = (subtotal_1[i_23] + scores_1[((i_23 * 2) + rv_3)]);
            }
            subtotal_1[i_23] = tl::AllReduce<tl::SumOp, 256, 32, 0, tl::NamedBarrier<256>>::run(subtotal_1[i_23], (&(((float*)buf_dyn_shmem)[14384])));
            subtotal_1[i_23] = tl::AllReduce<tl::SumOp, 4, 1, 0, tl::NamedBarrier<256>>::run(subtotal_1[i_23]);
          }
          #pragma unroll
          for (int i_24 = 0; i_24 < 4; ++i_24) {
            denominator_1[i_24] = ((denominator_1[i_24] * factor_1[i_24]) + subtotal_1[i_24]);
          }
          #pragma unroll
          for (int i_25 = 0; i_25 < 16; ++i_25) {
            out_1[i_25] = (out_1[i_25] * factor_1[(((i_25 >> 3) * 2) + ((i_25 & 3) >> 1))]);
          }
          tl::ptx_stmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[(((((((int)threadIdx.x) & 31) >> 3) * 512) + ((((((((int)threadIdx.x) & 31) * 64) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 26720)])), __pack_half2(((bfloat16_t)scores_1[0]), ((bfloat16_t)scores_1[1])), __pack_half2(((bfloat16_t)scores_1[2]), ((bfloat16_t)scores_1[3])), __pack_half2(((bfloat16_t)scores_1[4]), ((bfloat16_t)scores_1[5])), __pack_half2(((bfloat16_t)scores_1[6]), ((bfloat16_t)scores_1[7])));
          {
            bfloat16_t A_local_3[16];
            bfloat16_t B_local_3[8];
            __syncthreads();
            for (int ki_3 = 0; ki_3 < 4; ++ki_3) {
              for (int i_26 = 0; i_26 < 2; ++i_26) {
                tl::ptx_ldmatrix_x4((&(((bfloat16_t*)buf_dyn_shmem)[((((i_26 * 1024) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_3 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_3 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 26720)])), (&(A_local_3[(i_26 * 8)])));
              }
              tl::ptx_ldmatrix_x4_trans((&(((bfloat16_t*)buf_dyn_shmem)[((((((((int)threadIdx.x) >> 7) * 4096) + (ki_3 * 1024)) + (((((int)threadIdx.x) & 15) >> 3) * 512)) + ((((((((int)threadIdx.x) & 15) * 64) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) & 511)) + 14432)])), (&(B_local_3[0])));
              for (int i_27 = 0; i_27 < 2; ++i_27) {
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + (i_27 * 8)), reinterpret_cast<const unsigned*>(A_local_3 + (i_27 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 0));
                tl::mma_sync<tl::DataType::kBFloat16, tl::DataType::kBFloat16, tl::DataType::kFloat32, 16, 8, 16, false, true>(reinterpret_cast<float*>(out_1 + ((i_27 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_3 + (i_27 * 8)), reinterpret_cast<const unsigned*>(B_local_3 + 4));
              }
            }
          }
        }
        __syncthreads();
        #pragma unroll
        for (int i_28 = 0; i_28 < 8; ++i_28) {
          int src_8 = RowMap[(((((((int)blockIdx.x) * 64) + (r * 32)) + ((i_28 >> 2) * 16)) + ((i_28 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
          bool valid_10 = (((((i_28 >> 2) * 16) + ((i_28 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
          if (valid_10) {
            *(float2*)(((float*)buf_dyn_shmem) + (((((((int64_t)src_8) * (int64_t)128) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)5) * (int64_t)16)) + (((((int64_t)i_28) & (int64_t)3) >> (int64_t)1) * (int64_t)8)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)3) * (int64_t)2)) + (int64_t)48)) = *(float2*)(out_1 + (i_28 * 2));
          }
        }
        __syncthreads();
        if ((((((int)threadIdx.x) & 3) * 8) + (((int)threadIdx.x) >> 5)) == 0) {
          #pragma unroll
          for (int i_29 = 0; i_29 < 4; ++i_29) {
            int src_9 = RowMap[((((((int)blockIdx.x) * 64) + (r * 32)) + (i_29 * 8)) + ((((int)threadIdx.x) & 31) >> 2))];
            bool valid_11 = (((i_29 * 8) + ((((int)threadIdx.x) & 31) >> 2)) < Count[((((int)blockIdx.x) * 2) + r)]);
            if (valid_11) {
              ((float*)buf_dyn_shmem)[(((int64_t)src_9) + (int64_t)24)] = maximum_1[i_29];
              ((float*)buf_dyn_shmem)[((int64_t)src_9)] = denominator_1[i_29];
            }
          }
        }
      }
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_30 = 0; i_30 < 3; ++i_30) {
    *(float4*)(U + ((((((int)blockIdx.x) * 196608) + (((int)blockIdx.y) * 3072)) + (i_30 * 1024)) + (((int)threadIdx.x) * 4))) = *(float4*)(((float*)buf_dyn_shmem) + (((i_30 * 1024) + (((int)threadIdx.x) * 4)) + 48));
  }
  if (tl::tl_shuffle_elect<256>()) {
    tl::fence_proxy_async();
    tl::tma_store((&(Mout[((((int)blockIdx.x) * 1536) + (((int)blockIdx.y) * 24))])), (&(((float*)buf_dyn_shmem)[24])), 96);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
    tl::tma_store((&(Lout[((((int)blockIdx.x) * 1536) + (((int)blockIdx.y) * 24))])), (&(((float*)buf_dyn_shmem)[0])), 96);
    tl::tma_store_arrive();
    tl::tma_store_wait<0>();
  }
}

