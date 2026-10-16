// tilelang target: c -keys=cpu 
#define TVM_EXPORTS
#include "tvm/runtime/base.h"
#include "tvm/runtime/c_backend_api.h"
#include "tvm/ffi/c_api.h"
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#ifdef __OBJC__
#include "tvm/runtime/device_api.h"
#include "tvm/ffi/function.h"
#include <Metal/Metal.h>
#include <Foundation/Foundation.h>
#include <torch/mps.h>
#endif
void* __tvm_ffi__library_ctx = NULL;
static void* __tvm_error_ndim_mismatch_packed = NULL;
static void* __tvm_error_dtype_mismatch_packed = NULL;
static void* __tvm_error_expect_eq_packed = NULL;
static void* __tvm_error_byte_offset_mismatch_packed = NULL;
static void* __tvm_error_device_type_mismatch_packed = NULL;
static void* __tvm_error_null_ptr_packed = NULL;
static void* __tvm_set_device_packed = NULL;
static void* main_kernel_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result);
#ifdef __cplusplus
extern "C"
#endif
int32_t __tvm_ffi_main(void* self_handle, void* args, int32_t num_args, void* result) {
  __attribute__((aligned(64))) TVMFFIAny stack[18];
  void* stack_ffi_any = stack;
  if (!((num_args == 11))) {
    char __tvm_assert_msg_buf[512];
    snprintf(__tvm_assert_msg_buf, 512, "%s; expected: %lld, got: %lld", "main: num_args should be 11", (long long)(num_args), (long long)(11));
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", __tvm_assert_msg_buf);
    return -1;
  }
  if (!(!(args == NULL))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main: args pointer is NULL");
    return -1;
  }
  int32_t Q_handle_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((Q_handle_type_index == 0) || (Q_handle_type_index == 4)) || (Q_handle_type_index == 7)) || (64 <= Q_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Q expected pointer or tensor handle");
    return -1;
  }
  int32_t K_handle_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((K_handle_type_index == 0) || (K_handle_type_index == 4)) || (K_handle_type_index == 7)) || (64 <= K_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input K expected pointer or tensor handle");
    return -1;
  }
  int32_t V_handle_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((V_handle_type_index == 0) || (V_handle_type_index == 4)) || (V_handle_type_index == 7)) || (64 <= V_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input V expected pointer or tensor handle");
    return -1;
  }
  int32_t Indices_handle_type_index = (((TVMFFIAny*)args)[3].type_index);
  if (!(((((Indices_handle_type_index == 0) || (Indices_handle_type_index == 4)) || (Indices_handle_type_index == 7)) || (64 <= Indices_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Indices expected pointer or tensor handle");
    return -1;
  }
  int32_t Offsets_handle_type_index = (((TVMFFIAny*)args)[4].type_index);
  if (!(((((Offsets_handle_type_index == 0) || (Offsets_handle_type_index == 4)) || (Offsets_handle_type_index == 7)) || (64 <= Offsets_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Offsets expected pointer or tensor handle");
    return -1;
  }
  int32_t RowMap_handle_type_index = (((TVMFFIAny*)args)[5].type_index);
  if (!(((((RowMap_handle_type_index == 0) || (RowMap_handle_type_index == 4)) || (RowMap_handle_type_index == 7)) || (64 <= RowMap_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input RowMap expected pointer or tensor handle");
    return -1;
  }
  int32_t Count_handle_type_index = (((TVMFFIAny*)args)[6].type_index);
  if (!(((((Count_handle_type_index == 0) || (Count_handle_type_index == 4)) || (Count_handle_type_index == 7)) || (64 <= Count_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Count expected pointer or tensor handle");
    return -1;
  }
  int32_t Owner_handle_type_index = (((TVMFFIAny*)args)[7].type_index);
  if (!(((((Owner_handle_type_index == 0) || (Owner_handle_type_index == 4)) || (Owner_handle_type_index == 7)) || (64 <= Owner_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Owner expected pointer or tensor handle");
    return -1;
  }
  int32_t U_handle_type_index = (((TVMFFIAny*)args)[8].type_index);
  if (!(((((U_handle_type_index == 0) || (U_handle_type_index == 4)) || (U_handle_type_index == 7)) || (64 <= U_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input U expected pointer or tensor handle");
    return -1;
  }
  int32_t Mout_handle_type_index = (((TVMFFIAny*)args)[9].type_index);
  if (!(((((Mout_handle_type_index == 0) || (Mout_handle_type_index == 4)) || (Mout_handle_type_index == 7)) || (64 <= Mout_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Mout expected pointer or tensor handle");
    return -1;
  }
  int32_t Lout_handle_type_index = (((TVMFFIAny*)args)[10].type_index);
  if (!(((((Lout_handle_type_index == 0) || (Lout_handle_type_index == 4)) || (Lout_handle_type_index == 7)) || (64 <= Lout_handle_type_index)))) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "kernel main input Lout expected pointer or tensor handle");
    return -1;
  }
  void* Q_handle = ((Q_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  void* K_handle = ((K_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  void* V_handle = ((V_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  void* Indices_handle = ((Indices_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[3].v_ptr) + 24)) : (((TVMFFIAny*)args)[3].v_ptr));
  void* Offsets_handle = ((Offsets_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[4].v_ptr) + 24)) : (((TVMFFIAny*)args)[4].v_ptr));
  void* RowMap_handle = ((RowMap_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[5].v_ptr) + 24)) : (((TVMFFIAny*)args)[5].v_ptr));
  void* Count_handle = ((Count_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[6].v_ptr) + 24)) : (((TVMFFIAny*)args)[6].v_ptr));
  void* Owner_handle = ((Owner_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[7].v_ptr) + 24)) : (((TVMFFIAny*)args)[7].v_ptr));
  void* U_handle = ((U_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[8].v_ptr) + 24)) : (((TVMFFIAny*)args)[8].v_ptr));
  void* Mout_handle = ((Mout_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[9].v_ptr) + 24)) : (((TVMFFIAny*)args)[9].v_ptr));
  void* Lout_handle = ((Lout_handle_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[10].v_ptr) + 24)) : (((TVMFFIAny*)args)[10].v_ptr));
  bool main_Q_is_null = (Q_handle == NULL);
  if (!(!main_Q_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Q is expected to have non-NULL pointer");
    return -1;
  }
  bool main_K_is_null = (K_handle == NULL);
  if (!(!main_K_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.K is expected to have non-NULL pointer");
    return -1;
  }
  bool main_V_is_null = (V_handle == NULL);
  if (!(!main_V_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.V is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Indices_is_null = (Indices_handle == NULL);
  if (!(!main_Indices_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Indices is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Offsets_is_null = (Offsets_handle == NULL);
  if (!(!main_Offsets_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Offsets is expected to have non-NULL pointer");
    return -1;
  }
  bool main_RowMap_is_null = (RowMap_handle == NULL);
  if (!(!main_RowMap_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.RowMap is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Count_is_null = (Count_handle == NULL);
  if (!(!main_Count_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Count is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Owner_is_null = (Owner_handle == NULL);
  bool main_U_is_null = (U_handle == NULL);
  if (!(!main_U_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.U is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Mout_is_null = (Mout_handle == NULL);
  if (!(!main_Mout_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Mout is expected to have non-NULL pointer");
    return -1;
  }
  bool main_Lout_is_null = (Lout_handle == NULL);
  if (!(!main_Lout_is_null)) {
    TVMFFIErrorSetRaisedFromCStr("RuntimeError", "main.Lout is expected to have non-NULL pointer");
    return -1;
  }
  void* main_Q_shape = (((DLTensor*)Q_handle)[0].shape);
  void* main_K_shape = (((DLTensor*)K_handle)[0].shape);
  void* main_V_shape = (((DLTensor*)V_handle)[0].shape);
  void* main_Indices_shape = (((DLTensor*)Indices_handle)[0].shape);
  void* main_Offsets_shape = (((DLTensor*)Offsets_handle)[0].shape);
  void* main_RowMap_shape = (((DLTensor*)RowMap_handle)[0].shape);
  void* main_Count_shape = (((DLTensor*)Count_handle)[0].shape);
  void* condval;
  if (!main_Owner_is_null) {
    condval = (((DLTensor*)Owner_handle)[0].shape);
  } else {
    uint64_t v_ = (uint64_t)0;
    condval = (*(void* *)(&(v_)));
  }
  void* main_Owner_shape = condval;
  void* main_U_shape = (((DLTensor*)U_handle)[0].shape);
  void* main_Mout_shape = (((DLTensor*)Mout_handle)[0].shape);
  void* main_Lout_shape = (((DLTensor*)Lout_handle)[0].shape);
  if ((((DLTensor*)Q_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_1;
    result_1.type_index = kTVMFFINone;
    result_1.zero_padding = 0;
    result_1.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_1) != 0) {
      return -1;
    }
  }
  void* main_Q_strides = (((DLTensor*)Q_handle)[0].strides);
  int32_t dev_id = (((DLTensor*)Q_handle)[0].device.device_id);
  void* Q = (((DLTensor*)Q_handle)[0].data);
  if ((((DLTensor*)K_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_2;
    result_2.type_index = kTVMFFINone;
    result_2.zero_padding = 0;
    result_2.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_2) != 0) {
      return -1;
    }
  }
  void* main_K_strides = (((DLTensor*)K_handle)[0].strides);
  void* K = (((DLTensor*)K_handle)[0].data);
  if ((((DLTensor*)V_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_3;
    result_3.type_index = kTVMFFINone;
    result_3.zero_padding = 0;
    result_3.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_3) != 0) {
      return -1;
    }
  }
  void* main_V_strides = (((DLTensor*)V_handle)[0].strides);
  void* V = (((DLTensor*)V_handle)[0].data);
  if ((((DLTensor*)Indices_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_4;
    result_4.type_index = kTVMFFINone;
    result_4.zero_padding = 0;
    result_4.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_4) != 0) {
      return -1;
    }
  }
  void* main_Indices_strides = (((DLTensor*)Indices_handle)[0].strides);
  void* Indices = (((DLTensor*)Indices_handle)[0].data);
  if ((((DLTensor*)Offsets_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_5;
    result_5.type_index = kTVMFFINone;
    result_5.zero_padding = 0;
    result_5.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_5) != 0) {
      return -1;
    }
  }
  void* main_Offsets_strides = (((DLTensor*)Offsets_handle)[0].strides);
  void* Offsets = (((DLTensor*)Offsets_handle)[0].data);
  if ((((DLTensor*)RowMap_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_6;
    result_6.type_index = kTVMFFINone;
    result_6.zero_padding = 0;
    result_6.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_6) != 0) {
      return -1;
    }
  }
  void* main_RowMap_strides = (((DLTensor*)RowMap_handle)[0].strides);
  void* RowMap = (((DLTensor*)RowMap_handle)[0].data);
  if ((((DLTensor*)Count_handle)[0].ndim) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_7;
    result_7.type_index = kTVMFFINone;
    result_7.zero_padding = 0;
    result_7.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_7) != 0) {
      return -1;
    }
  }
  void* main_Count_strides = (((DLTensor*)Count_handle)[0].strides);
  void* Count = (((DLTensor*)Count_handle)[0].data);
  if (!main_Owner_is_null) {
    if ((((DLTensor*)Owner_handle)[0].ndim) != 3) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].ndim));
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
      if (__tvm_error_ndim_mismatch_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_8;
      result_8.type_index = kTVMFFINone;
      result_8.zero_padding = 0;
      result_8.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_8) != 0) {
        return -1;
      }
    }
  } else {
  }
  void* condval_1;
  if (!main_Owner_is_null) {
    condval_1 = (((DLTensor*)Owner_handle)[0].strides);
  } else {
    uint64_t v__1 = (uint64_t)0;
    condval_1 = (*(void* *)(&(v__1)));
  }
  void* main_Owner_strides = condval_1;
  void* condval_2;
  if (!main_Owner_is_null) {
    condval_2 = (((DLTensor*)Owner_handle)[0].data);
  } else {
    uint64_t v__2 = (uint64_t)0;
    condval_2 = (*(void* *)(&(v__2)));
  }
  void* Owner = condval_2;
  if ((((DLTensor*)U_handle)[0].ndim) != 4) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)4;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_9;
    result_9.type_index = kTVMFFINone;
    result_9.zero_padding = 0;
    result_9.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_9) != 0) {
      return -1;
    }
  }
  void* main_U_strides = (((DLTensor*)U_handle)[0].strides);
  void* U = (((DLTensor*)U_handle)[0].data);
  if ((((DLTensor*)Mout_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_10;
    result_10.type_index = kTVMFFINone;
    result_10.zero_padding = 0;
    result_10.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_10) != 0) {
      return -1;
    }
  }
  void* main_Mout_strides = (((DLTensor*)Mout_handle)[0].strides);
  void* Mout = (((DLTensor*)Mout_handle)[0].data);
  if ((((DLTensor*)Lout_handle)[0].ndim) != 3) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)3;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].ndim));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_ndim_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_ndim_mismatch", &__tvm_error_ndim_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_11;
    result_11.type_index = kTVMFFINone;
    result_11.zero_padding = 0;
    result_11.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_ndim_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_11) != 0) {
      return -1;
    }
  }
  void* main_Lout_strides = (((DLTensor*)Lout_handle)[0].strides);
  void* Lout = (((DLTensor*)Lout_handle)[0].data);
  if ((((((DLTensor*)Q_handle)[0].dtype.code) != (uint8_t)4) || ((((DLTensor*)Q_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)Q_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)4;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_12;
    result_12.type_index = kTVMFFINone;
    result_12.zero_padding = 0;
    result_12.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_12) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Q_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Q_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_13;
    result_13.type_index = kTVMFFINone;
    result_13.zero_padding = 0;
    result_13.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_13) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Q_shape)[1]) != 24) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Q_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_14;
    result_14.type_index = kTVMFFINone;
    result_14.zero_padding = 0;
    result_14.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_14) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Q_shape)[2]) != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Q_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_15;
    result_15.type_index = kTVMFFINone;
    result_15.zero_padding = 0;
    result_15.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_15) != 0) {
      return -1;
    }
  }
  int32_t condval_3;
  if ((main_Q_strides == NULL)) {
    condval_3 = 1;
  } else {
    condval_3 = ((int32_t)((int64_t*)main_Q_strides)[2]);
  }
  if (condval_3 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_4;
    if ((main_Q_strides == NULL)) {
      condval_4 = 1;
    } else {
      condval_4 = ((int32_t)((int64_t*)main_Q_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_4);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_16;
    result_16.type_index = kTVMFFINone;
    result_16.zero_padding = 0;
    result_16.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_16) != 0) {
      return -1;
    }
  }
  int32_t condval_5;
  if ((main_Q_strides == NULL)) {
    condval_5 = 1;
  } else {
    condval_5 = ((int32_t)((int64_t*)main_Q_strides)[1]);
  }
  if (condval_5 != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_6;
    if ((main_Q_strides == NULL)) {
      condval_6 = 1;
    } else {
      condval_6 = ((int32_t)((int64_t*)main_Q_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_6);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_17;
    result_17.type_index = kTVMFFINone;
    result_17.zero_padding = 0;
    result_17.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_17) != 0) {
      return -1;
    }
  }
  int32_t condval_7;
  if ((main_Q_strides == NULL)) {
    condval_7 = 1;
  } else {
    condval_7 = ((int32_t)((int64_t*)main_Q_strides)[0]);
  }
  if (condval_7 != 3072) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)3072;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_8;
    if ((main_Q_strides == NULL)) {
      condval_8 = 1;
    } else {
      condval_8 = ((int32_t)((int64_t*)main_Q_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_8);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_18;
    result_18.type_index = kTVMFFINone;
    result_18.zero_padding = 0;
    result_18.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_18) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)Q_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_19;
    result_19.type_index = kTVMFFINone;
    result_19.zero_padding = 0;
    result_19.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_19) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Q_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_20;
    result_20.type_index = kTVMFFINone;
    result_20.zero_padding = 0;
    result_20.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_20) != 0) {
      return -1;
    }
  }
  if (Q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Q";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_21;
    result_21.type_index = kTVMFFINone;
    result_21.zero_padding = 0;
    result_21.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_21) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)K_handle)[0].dtype.code) != (uint8_t)4) || ((((DLTensor*)K_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)K_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)4;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_22;
    result_22.type_index = kTVMFFINone;
    result_22.zero_padding = 0;
    result_22.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_22) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_K_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_K_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_23;
    result_23.type_index = kTVMFFINone;
    result_23.zero_padding = 0;
    result_23.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_23) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_K_shape)[1]) != 90066) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)90066;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_K_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_24;
    result_24.type_index = kTVMFFINone;
    result_24.zero_padding = 0;
    result_24.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_24) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_K_shape)[2]) != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_K_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_25;
    result_25.type_index = kTVMFFINone;
    result_25.zero_padding = 0;
    result_25.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_25) != 0) {
      return -1;
    }
  }
  int32_t condval_9;
  if ((main_K_strides == NULL)) {
    condval_9 = 1;
  } else {
    condval_9 = ((int32_t)((int64_t*)main_K_strides)[2]);
  }
  if (condval_9 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_10;
    if ((main_K_strides == NULL)) {
      condval_10 = 1;
    } else {
      condval_10 = ((int32_t)((int64_t*)main_K_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_10);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_26;
    result_26.type_index = kTVMFFINone;
    result_26.zero_padding = 0;
    result_26.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_26) != 0) {
      return -1;
    }
  }
  int32_t condval_11;
  if ((main_K_strides == NULL)) {
    condval_11 = 1;
  } else {
    condval_11 = ((int32_t)((int64_t*)main_K_strides)[1]);
  }
  if (condval_11 != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_12;
    if ((main_K_strides == NULL)) {
      condval_12 = 1;
    } else {
      condval_12 = ((int32_t)((int64_t*)main_K_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_12);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_27;
    result_27.type_index = kTVMFFINone;
    result_27.zero_padding = 0;
    result_27.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_27) != 0) {
      return -1;
    }
  }
  int32_t condval_13;
  if ((main_K_strides == NULL)) {
    condval_13 = 1;
  } else {
    condval_13 = ((int32_t)((int64_t*)main_K_strides)[0]);
  }
  if (condval_13 != 11528448) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)11528448;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_14;
    if ((main_K_strides == NULL)) {
      condval_14 = 1;
    } else {
      condval_14 = ((int32_t)((int64_t*)main_K_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_14);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_28;
    result_28.type_index = kTVMFFINone;
    result_28.zero_padding = 0;
    result_28.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_28) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)K_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_29;
    result_29.type_index = kTVMFFINone;
    result_29.zero_padding = 0;
    result_29.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_29) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)K_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_30;
    result_30.type_index = kTVMFFINone;
    result_30.zero_padding = 0;
    result_30.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_30) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)K_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)K_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_31;
    result_31.type_index = kTVMFFINone;
    result_31.zero_padding = 0;
    result_31.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_31) != 0) {
      return -1;
    }
  }
  if (K == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"K";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_32;
    result_32.type_index = kTVMFFINone;
    result_32.zero_padding = 0;
    result_32.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_32) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)V_handle)[0].dtype.code) != (uint8_t)4) || ((((DLTensor*)V_handle)[0].dtype.bits) != (uint8_t)16)) || ((((DLTensor*)V_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)4;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)16;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_33;
    result_33.type_index = kTVMFFINone;
    result_33.zero_padding = 0;
    result_33.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_33) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_V_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_V_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_34;
    result_34.type_index = kTVMFFINone;
    result_34.zero_padding = 0;
    result_34.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_34) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_V_shape)[1]) != 90066) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)90066;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_V_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_35;
    result_35.type_index = kTVMFFINone;
    result_35.zero_padding = 0;
    result_35.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_35) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_V_shape)[2]) != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_V_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_36;
    result_36.type_index = kTVMFFINone;
    result_36.zero_padding = 0;
    result_36.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_36) != 0) {
      return -1;
    }
  }
  int32_t condval_15;
  if ((main_V_strides == NULL)) {
    condval_15 = 1;
  } else {
    condval_15 = ((int32_t)((int64_t*)main_V_strides)[2]);
  }
  if (condval_15 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_16;
    if ((main_V_strides == NULL)) {
      condval_16 = 1;
    } else {
      condval_16 = ((int32_t)((int64_t*)main_V_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_16);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_37;
    result_37.type_index = kTVMFFINone;
    result_37.zero_padding = 0;
    result_37.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_37) != 0) {
      return -1;
    }
  }
  int32_t condval_17;
  if ((main_V_strides == NULL)) {
    condval_17 = 1;
  } else {
    condval_17 = ((int32_t)((int64_t*)main_V_strides)[1]);
  }
  if (condval_17 != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_18;
    if ((main_V_strides == NULL)) {
      condval_18 = 1;
    } else {
      condval_18 = ((int32_t)((int64_t*)main_V_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_18);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_38;
    result_38.type_index = kTVMFFINone;
    result_38.zero_padding = 0;
    result_38.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_38) != 0) {
      return -1;
    }
  }
  int32_t condval_19;
  if ((main_V_strides == NULL)) {
    condval_19 = 1;
  } else {
    condval_19 = ((int32_t)((int64_t*)main_V_strides)[0]);
  }
  if (condval_19 != 11528448) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)11528448;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_20;
    if ((main_V_strides == NULL)) {
      condval_20 = 1;
    } else {
      condval_20 = ((int32_t)((int64_t*)main_V_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_20);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_39;
    result_39.type_index = kTVMFFINone;
    result_39.zero_padding = 0;
    result_39.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_39) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)V_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_40;
    result_40.type_index = kTVMFFINone;
    result_40.zero_padding = 0;
    result_40.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_40) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)V_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_41;
    result_41.type_index = kTVMFFINone;
    result_41.zero_padding = 0;
    result_41.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_41) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)V_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)V_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_42;
    result_42.type_index = kTVMFFINone;
    result_42.zero_padding = 0;
    result_42.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_42) != 0) {
      return -1;
    }
  }
  if (V == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"V";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_43;
    result_43.type_index = kTVMFFINone;
    result_43.zero_padding = 0;
    result_43.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_43) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)Indices_handle)[0].dtype.code) != (uint8_t)0) || ((((DLTensor*)Indices_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)Indices_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_44;
    result_44.type_index = kTVMFFINone;
    result_44.zero_padding = 0;
    result_44.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_44) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Indices_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Indices_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_45;
    result_45.type_index = kTVMFFINone;
    result_45.zero_padding = 0;
    result_45.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_45) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Indices_shape)[1]) != 90066) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)90066;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Indices_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_46;
    result_46.type_index = kTVMFFINone;
    result_46.zero_padding = 0;
    result_46.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_46) != 0) {
      return -1;
    }
  }
  int32_t condval_21;
  if ((main_Indices_strides == NULL)) {
    condval_21 = 1;
  } else {
    condval_21 = ((int32_t)((int64_t*)main_Indices_strides)[1]);
  }
  if (condval_21 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_22;
    if ((main_Indices_strides == NULL)) {
      condval_22 = 1;
    } else {
      condval_22 = ((int32_t)((int64_t*)main_Indices_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_22);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_47;
    result_47.type_index = kTVMFFINone;
    result_47.zero_padding = 0;
    result_47.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_47) != 0) {
      return -1;
    }
  }
  int32_t condval_23;
  if ((main_Indices_strides == NULL)) {
    condval_23 = 1;
  } else {
    condval_23 = ((int32_t)((int64_t*)main_Indices_strides)[0]);
  }
  if (condval_23 != 90066) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)90066;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_24;
    if ((main_Indices_strides == NULL)) {
      condval_24 = 1;
    } else {
      condval_24 = ((int32_t)((int64_t*)main_Indices_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_24);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_48;
    result_48.type_index = kTVMFFINone;
    result_48.zero_padding = 0;
    result_48.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_48) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)Indices_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_49;
    result_49.type_index = kTVMFFINone;
    result_49.zero_padding = 0;
    result_49.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_49) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Indices_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_50;
    result_50.type_index = kTVMFFINone;
    result_50.zero_padding = 0;
    result_50.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_50) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Indices_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Indices_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_51;
    result_51.type_index = kTVMFFINone;
    result_51.zero_padding = 0;
    result_51.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_51) != 0) {
      return -1;
    }
  }
  if (Indices == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Indices";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_52;
    result_52.type_index = kTVMFFINone;
    result_52.zero_padding = 0;
    result_52.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_52) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)Offsets_handle)[0].dtype.code) != (uint8_t)0) || ((((DLTensor*)Offsets_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)Offsets_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_53;
    result_53.type_index = kTVMFFINone;
    result_53.zero_padding = 0;
    result_53.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_53) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Offsets_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Offsets_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_54;
    result_54.type_index = kTVMFFINone;
    result_54.zero_padding = 0;
    result_54.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_54) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Offsets_shape)[1]) != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Offsets_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_55;
    result_55.type_index = kTVMFFINone;
    result_55.zero_padding = 0;
    result_55.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_55) != 0) {
      return -1;
    }
  }
  int32_t condval_25;
  if ((main_Offsets_strides == NULL)) {
    condval_25 = 1;
  } else {
    condval_25 = ((int32_t)((int64_t*)main_Offsets_strides)[1]);
  }
  if (condval_25 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_26;
    if ((main_Offsets_strides == NULL)) {
      condval_26 = 1;
    } else {
      condval_26 = ((int32_t)((int64_t*)main_Offsets_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_26);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_56;
    result_56.type_index = kTVMFFINone;
    result_56.zero_padding = 0;
    result_56.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_56) != 0) {
      return -1;
    }
  }
  int32_t condval_27;
  if ((main_Offsets_strides == NULL)) {
    condval_27 = 1;
  } else {
    condval_27 = ((int32_t)((int64_t*)main_Offsets_strides)[0]);
  }
  if (condval_27 != 8) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)8;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_28;
    if ((main_Offsets_strides == NULL)) {
      condval_28 = 1;
    } else {
      condval_28 = ((int32_t)((int64_t*)main_Offsets_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_28);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_57;
    result_57.type_index = kTVMFFINone;
    result_57.zero_padding = 0;
    result_57.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_57) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)Offsets_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_58;
    result_58.type_index = kTVMFFINone;
    result_58.zero_padding = 0;
    result_58.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_58) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Offsets_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_59;
    result_59.type_index = kTVMFFINone;
    result_59.zero_padding = 0;
    result_59.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_59) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Offsets_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Offsets_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_60;
    result_60.type_index = kTVMFFINone;
    result_60.zero_padding = 0;
    result_60.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_60) != 0) {
      return -1;
    }
  }
  if (Offsets == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Offsets";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_61;
    result_61.type_index = kTVMFFINone;
    result_61.zero_padding = 0;
    result_61.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_61) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)RowMap_handle)[0].dtype.code) != (uint8_t)0) || ((((DLTensor*)RowMap_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)RowMap_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_62;
    result_62.type_index = kTVMFFINone;
    result_62.zero_padding = 0;
    result_62.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_62) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_RowMap_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_RowMap_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_63;
    result_63.type_index = kTVMFFINone;
    result_63.zero_padding = 0;
    result_63.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_63) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_RowMap_shape)[1]) != 7) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)7;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_RowMap_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_64;
    result_64.type_index = kTVMFFINone;
    result_64.zero_padding = 0;
    result_64.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_64) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_RowMap_shape)[2]) != 32) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_RowMap_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_65;
    result_65.type_index = kTVMFFINone;
    result_65.zero_padding = 0;
    result_65.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_65) != 0) {
      return -1;
    }
  }
  int32_t condval_29;
  if ((main_RowMap_strides == NULL)) {
    condval_29 = 1;
  } else {
    condval_29 = ((int32_t)((int64_t*)main_RowMap_strides)[2]);
  }
  if (condval_29 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_30;
    if ((main_RowMap_strides == NULL)) {
      condval_30 = 1;
    } else {
      condval_30 = ((int32_t)((int64_t*)main_RowMap_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_30);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_66;
    result_66.type_index = kTVMFFINone;
    result_66.zero_padding = 0;
    result_66.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_66) != 0) {
      return -1;
    }
  }
  int32_t condval_31;
  if ((main_RowMap_strides == NULL)) {
    condval_31 = 1;
  } else {
    condval_31 = ((int32_t)((int64_t*)main_RowMap_strides)[1]);
  }
  if (condval_31 != 32) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_32;
    if ((main_RowMap_strides == NULL)) {
      condval_32 = 1;
    } else {
      condval_32 = ((int32_t)((int64_t*)main_RowMap_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_32);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_67;
    result_67.type_index = kTVMFFINone;
    result_67.zero_padding = 0;
    result_67.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_67) != 0) {
      return -1;
    }
  }
  int32_t condval_33;
  if ((main_RowMap_strides == NULL)) {
    condval_33 = 1;
  } else {
    condval_33 = ((int32_t)((int64_t*)main_RowMap_strides)[0]);
  }
  if (condval_33 != 224) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)224;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_34;
    if ((main_RowMap_strides == NULL)) {
      condval_34 = 1;
    } else {
      condval_34 = ((int32_t)((int64_t*)main_RowMap_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_34);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_68;
    result_68.type_index = kTVMFFINone;
    result_68.zero_padding = 0;
    result_68.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_68) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)RowMap_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_69;
    result_69.type_index = kTVMFFINone;
    result_69.zero_padding = 0;
    result_69.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_69) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)RowMap_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_70;
    result_70.type_index = kTVMFFINone;
    result_70.zero_padding = 0;
    result_70.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_70) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)RowMap_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)RowMap_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_71;
    result_71.type_index = kTVMFFINone;
    result_71.zero_padding = 0;
    result_71.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_71) != 0) {
      return -1;
    }
  }
  if (RowMap == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"RowMap";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_72;
    result_72.type_index = kTVMFFINone;
    result_72.zero_padding = 0;
    result_72.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_72) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)Count_handle)[0].dtype.code) != (uint8_t)0) || ((((DLTensor*)Count_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)Count_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_73;
    result_73.type_index = kTVMFFINone;
    result_73.zero_padding = 0;
    result_73.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_73) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Count_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Count_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_74;
    result_74.type_index = kTVMFFINone;
    result_74.zero_padding = 0;
    result_74.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_74) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Count_shape)[1]) != 7) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)7;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Count_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_75;
    result_75.type_index = kTVMFFINone;
    result_75.zero_padding = 0;
    result_75.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_75) != 0) {
      return -1;
    }
  }
  int32_t condval_35;
  if ((main_Count_strides == NULL)) {
    condval_35 = 1;
  } else {
    condval_35 = ((int32_t)((int64_t*)main_Count_strides)[1]);
  }
  if (condval_35 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_36;
    if ((main_Count_strides == NULL)) {
      condval_36 = 1;
    } else {
      condval_36 = ((int32_t)((int64_t*)main_Count_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_36);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_76;
    result_76.type_index = kTVMFFINone;
    result_76.zero_padding = 0;
    result_76.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_76) != 0) {
      return -1;
    }
  }
  int32_t condval_37;
  if ((main_Count_strides == NULL)) {
    condval_37 = 1;
  } else {
    condval_37 = ((int32_t)((int64_t*)main_Count_strides)[0]);
  }
  if (condval_37 != 7) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)7;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_38;
    if ((main_Count_strides == NULL)) {
      condval_38 = 1;
    } else {
      condval_38 = ((int32_t)((int64_t*)main_Count_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_38);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_77;
    result_77.type_index = kTVMFFINone;
    result_77.zero_padding = 0;
    result_77.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_77) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)Count_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_78;
    result_78.type_index = kTVMFFINone;
    result_78.zero_padding = 0;
    result_78.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_78) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Count_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_79;
    result_79.type_index = kTVMFFINone;
    result_79.zero_padding = 0;
    result_79.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_79) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Count_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Count_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_80;
    result_80.type_index = kTVMFFINone;
    result_80.zero_padding = 0;
    result_80.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_80) != 0) {
      return -1;
    }
  }
  if (Count == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Count";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_81;
    result_81.type_index = kTVMFFINone;
    result_81.zero_padding = 0;
    result_81.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_81) != 0) {
      return -1;
    }
  }
  uint8_t condval_39;
  if (!main_Owner_is_null) {
    condval_39 = (((DLTensor*)Owner_handle)[0].dtype.code);
  } else {
    condval_39 = (uint8_t)0;
  }
  uint8_t condval_40;
  if (!main_Owner_is_null) {
    condval_40 = (((DLTensor*)Owner_handle)[0].dtype.bits);
  } else {
    condval_40 = (uint8_t)32;
  }
  uint16_t condval_41;
  if (!main_Owner_is_null) {
    condval_41 = (((DLTensor*)Owner_handle)[0].dtype.lanes);
  } else {
    condval_41 = (uint16_t)1;
  }
  if (!main_Owner_is_null && (((condval_39 != (uint8_t)0) || (condval_40 != (uint8_t)32)) || (condval_41 != (uint16_t)1))) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_82;
    result_82.type_index = kTVMFFINone;
    result_82.zero_padding = 0;
    result_82.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_82) != 0) {
      return -1;
    }
  }
  if (!main_Owner_is_null) {
    if (((int32_t)((int64_t*)main_Owner_shape)[0]) != 2) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Owner_shape)[0]));
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_83;
      result_83.type_index = kTVMFFINone;
      result_83.zero_padding = 0;
      result_83.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_83) != 0) {
        return -1;
      }
    }
    if (((int32_t)((int64_t*)main_Owner_shape)[1]) != 7) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)7;
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Owner_shape)[1]));
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_84;
      result_84.type_index = kTVMFFINone;
      result_84.zero_padding = 0;
      result_84.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_84) != 0) {
        return -1;
      }
    }
    if (((int32_t)((int64_t*)main_Owner_shape)[2]) != 24) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Owner_shape)[2]));
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_85;
      result_85.type_index = kTVMFFINone;
      result_85.zero_padding = 0;
      result_85.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_85) != 0) {
        return -1;
      }
    }
    int32_t condval_42;
    if ((main_Owner_strides == NULL)) {
      condval_42 = 1;
    } else {
      condval_42 = ((int32_t)((int64_t*)main_Owner_strides)[2]);
    }
    if (condval_42 != 1) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      int32_t condval_43;
      if ((main_Owner_strides == NULL)) {
        condval_43 = 1;
      } else {
        condval_43 = ((int32_t)((int64_t*)main_Owner_strides)[2]);
      }
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_43);
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_86;
      result_86.type_index = kTVMFFINone;
      result_86.zero_padding = 0;
      result_86.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_86) != 0) {
        return -1;
      }
    }
    int32_t condval_44;
    if ((main_Owner_strides == NULL)) {
      condval_44 = 1;
    } else {
      condval_44 = ((int32_t)((int64_t*)main_Owner_strides)[1]);
    }
    if (condval_44 != 24) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      int32_t condval_45;
      if ((main_Owner_strides == NULL)) {
        condval_45 = 1;
      } else {
        condval_45 = ((int32_t)((int64_t*)main_Owner_strides)[1]);
      }
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_45);
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_87;
      result_87.type_index = kTVMFFINone;
      result_87.zero_padding = 0;
      result_87.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_87) != 0) {
        return -1;
      }
    }
    int32_t condval_46;
    if ((main_Owner_strides == NULL)) {
      condval_46 = 1;
    } else {
      condval_46 = ((int32_t)((int64_t*)main_Owner_strides)[0]);
    }
    if (condval_46 != 168) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)168;
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      int32_t condval_47;
      if ((main_Owner_strides == NULL)) {
        condval_47 = 1;
      } else {
        condval_47 = ((int32_t)((int64_t*)main_Owner_strides)[0]);
      }
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_47);
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_88;
      result_88.type_index = kTVMFFINone;
      result_88.zero_padding = 0;
      result_88.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_88) != 0) {
        return -1;
      }
    }
  }
  if (!main_Owner_is_null) {
    if ((uint64_t)0 != (((DLTensor*)Owner_handle)[0].byte_offset)) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].byte_offset));
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
      if (__tvm_error_byte_offset_mismatch_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_89;
      result_89.type_index = kTVMFFINone;
      result_89.zero_padding = 0;
      result_89.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_89) != 0) {
        return -1;
      }
    }
  } else {
  }
  if (!main_Owner_is_null) {
    if ((((DLTensor*)Owner_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].device.device_id));
      (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
      if (__tvm_error_expect_eq_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_90;
      result_90.type_index = kTVMFFINone;
      result_90.zero_padding = 0;
      result_90.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_90) != 0) {
        return -1;
      }
    }
  }
  if (!main_Owner_is_null) {
    if ((((DLTensor*)Owner_handle)[0].device.device_type) != 2) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Owner_handle)[0].device.device_type));
      (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
      if (__tvm_error_device_type_mismatch_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_91;
      result_91.type_index = kTVMFFINone;
      result_91.zero_padding = 0;
      result_91.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_91) != 0) {
        return -1;
      }
    }
  } else {
  }
  if (!main_Owner_is_null) {
    if (Owner == NULL) {
      (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
      (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Owner";
      (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
      (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
      (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
      (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
      (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
      if (__tvm_error_null_ptr_packed == NULL) {
        if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
          return -1;
        }
      }
      TVMFFIAny result_92;
      result_92.type_index = kTVMFFINone;
      result_92.zero_padding = 0;
      result_92.v_int64 = 0;
      if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_92) != 0) {
        return -1;
      }
    }
  } else {
  }
  if ((((((DLTensor*)U_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)U_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)U_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_93;
    result_93.type_index = kTVMFFINone;
    result_93.zero_padding = 0;
    result_93.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_93) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_U_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_U_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_94;
    result_94.type_index = kTVMFFINone;
    result_94.zero_padding = 0;
    result_94.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_94) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_U_shape)[1]) != 64) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)64;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_U_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_95;
    result_95.type_index = kTVMFFINone;
    result_95.zero_padding = 0;
    result_95.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_95) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_U_shape)[2]) != 24) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_U_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_96;
    result_96.type_index = kTVMFFINone;
    result_96.zero_padding = 0;
    result_96.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_96) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_U_shape)[3]) != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[3]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_U_shape)[3]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_97;
    result_97.type_index = kTVMFFINone;
    result_97.zero_padding = 0;
    result_97.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_97) != 0) {
      return -1;
    }
  }
  int32_t condval_48;
  if ((main_U_strides == NULL)) {
    condval_48 = 1;
  } else {
    condval_48 = ((int32_t)((int64_t*)main_U_strides)[3]);
  }
  if (condval_48 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[3]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_49;
    if ((main_U_strides == NULL)) {
      condval_49 = 1;
    } else {
      condval_49 = ((int32_t)((int64_t*)main_U_strides)[3]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_49);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_98;
    result_98.type_index = kTVMFFINone;
    result_98.zero_padding = 0;
    result_98.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_98) != 0) {
      return -1;
    }
  }
  int32_t condval_50;
  if ((main_U_strides == NULL)) {
    condval_50 = 1;
  } else {
    condval_50 = ((int32_t)((int64_t*)main_U_strides)[2]);
  }
  if (condval_50 != 128) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)128;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_51;
    if ((main_U_strides == NULL)) {
      condval_51 = 1;
    } else {
      condval_51 = ((int32_t)((int64_t*)main_U_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_51);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_99;
    result_99.type_index = kTVMFFINone;
    result_99.zero_padding = 0;
    result_99.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_99) != 0) {
      return -1;
    }
  }
  int32_t condval_52;
  if ((main_U_strides == NULL)) {
    condval_52 = 1;
  } else {
    condval_52 = ((int32_t)((int64_t*)main_U_strides)[1]);
  }
  if (condval_52 != 3072) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)3072;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_53;
    if ((main_U_strides == NULL)) {
      condval_53 = 1;
    } else {
      condval_53 = ((int32_t)((int64_t*)main_U_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_53);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_100;
    result_100.type_index = kTVMFFINone;
    result_100.zero_padding = 0;
    result_100.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_100) != 0) {
      return -1;
    }
  }
  int32_t condval_54;
  if ((main_U_strides == NULL)) {
    condval_54 = 1;
  } else {
    condval_54 = ((int32_t)((int64_t*)main_U_strides)[0]);
  }
  if (condval_54 != 196608) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)196608;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_55;
    if ((main_U_strides == NULL)) {
      condval_55 = 1;
    } else {
      condval_55 = ((int32_t)((int64_t*)main_U_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_55);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_101;
    result_101.type_index = kTVMFFINone;
    result_101.zero_padding = 0;
    result_101.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_101) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)U_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_102;
    result_102.type_index = kTVMFFINone;
    result_102.zero_padding = 0;
    result_102.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_102) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)U_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_103;
    result_103.type_index = kTVMFFINone;
    result_103.zero_padding = 0;
    result_103.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_103) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)U_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)U_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_104;
    result_104.type_index = kTVMFFINone;
    result_104.zero_padding = 0;
    result_104.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_104) != 0) {
      return -1;
    }
  }
  if (U == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"U";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_105;
    result_105.type_index = kTVMFFINone;
    result_105.zero_padding = 0;
    result_105.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_105) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)Mout_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)Mout_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)Mout_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_106;
    result_106.type_index = kTVMFFINone;
    result_106.zero_padding = 0;
    result_106.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_106) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Mout_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Mout_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_107;
    result_107.type_index = kTVMFFINone;
    result_107.zero_padding = 0;
    result_107.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_107) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Mout_shape)[1]) != 64) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)64;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Mout_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_108;
    result_108.type_index = kTVMFFINone;
    result_108.zero_padding = 0;
    result_108.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_108) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Mout_shape)[2]) != 24) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Mout_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_109;
    result_109.type_index = kTVMFFINone;
    result_109.zero_padding = 0;
    result_109.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_109) != 0) {
      return -1;
    }
  }
  int32_t condval_56;
  if ((main_Mout_strides == NULL)) {
    condval_56 = 1;
  } else {
    condval_56 = ((int32_t)((int64_t*)main_Mout_strides)[2]);
  }
  if (condval_56 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_57;
    if ((main_Mout_strides == NULL)) {
      condval_57 = 1;
    } else {
      condval_57 = ((int32_t)((int64_t*)main_Mout_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_57);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_110;
    result_110.type_index = kTVMFFINone;
    result_110.zero_padding = 0;
    result_110.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_110) != 0) {
      return -1;
    }
  }
  int32_t condval_58;
  if ((main_Mout_strides == NULL)) {
    condval_58 = 1;
  } else {
    condval_58 = ((int32_t)((int64_t*)main_Mout_strides)[1]);
  }
  if (condval_58 != 24) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_59;
    if ((main_Mout_strides == NULL)) {
      condval_59 = 1;
    } else {
      condval_59 = ((int32_t)((int64_t*)main_Mout_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_59);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_111;
    result_111.type_index = kTVMFFINone;
    result_111.zero_padding = 0;
    result_111.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_111) != 0) {
      return -1;
    }
  }
  int32_t condval_60;
  if ((main_Mout_strides == NULL)) {
    condval_60 = 1;
  } else {
    condval_60 = ((int32_t)((int64_t*)main_Mout_strides)[0]);
  }
  if (condval_60 != 1536) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1536;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_61;
    if ((main_Mout_strides == NULL)) {
      condval_61 = 1;
    } else {
      condval_61 = ((int32_t)((int64_t*)main_Mout_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_61);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_112;
    result_112.type_index = kTVMFFINone;
    result_112.zero_padding = 0;
    result_112.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_112) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)Mout_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_113;
    result_113.type_index = kTVMFFINone;
    result_113.zero_padding = 0;
    result_113.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_113) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Mout_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_114;
    result_114.type_index = kTVMFFINone;
    result_114.zero_padding = 0;
    result_114.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_114) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Mout_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Mout_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_115;
    result_115.type_index = kTVMFFINone;
    result_115.zero_padding = 0;
    result_115.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_115) != 0) {
      return -1;
    }
  }
  if (Mout == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Mout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_116;
    result_116.type_index = kTVMFFINone;
    result_116.zero_padding = 0;
    result_116.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_116) != 0) {
      return -1;
    }
  }
  if ((((((DLTensor*)Lout_handle)[0].dtype.code) != (uint8_t)2) || ((((DLTensor*)Lout_handle)[0].dtype.bits) != (uint8_t)32)) || ((((DLTensor*)Lout_handle)[0].dtype.lanes) != (uint16_t)1)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].dtype.code));
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].dtype.bits));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].dtype.lanes));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = (int64_t)32;
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = (int64_t)0;
    if (__tvm_error_dtype_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_dtype_mismatch", &__tvm_error_dtype_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_117;
    result_117.type_index = kTVMFFINone;
    result_117.zero_padding = 0;
    result_117.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_dtype_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 8, &result_117) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Lout_shape)[0]) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Lout_shape)[0]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_118;
    result_118.type_index = kTVMFFINone;
    result_118.zero_padding = 0;
    result_118.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_118) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Lout_shape)[1]) != 64) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)64;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Lout_shape)[1]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_119;
    result_119.type_index = kTVMFFINone;
    result_119.zero_padding = 0;
    result_119.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_119) != 0) {
      return -1;
    }
  }
  if (((int32_t)((int64_t*)main_Lout_shape)[2]) != 24) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"shape[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)((int32_t)((int64_t*)main_Lout_shape)[2]));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_120;
    result_120.type_index = kTVMFFINone;
    result_120.zero_padding = 0;
    result_120.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_120) != 0) {
      return -1;
    }
  }
  int32_t condval_62;
  if ((main_Lout_strides == NULL)) {
    condval_62 = 1;
  } else {
    condval_62 = ((int32_t)((int64_t*)main_Lout_strides)[2]);
  }
  if (condval_62 != 1) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[2]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_63;
    if ((main_Lout_strides == NULL)) {
      condval_63 = 1;
    } else {
      condval_63 = ((int32_t)((int64_t*)main_Lout_strides)[2]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_63);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_121;
    result_121.type_index = kTVMFFINone;
    result_121.zero_padding = 0;
    result_121.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_121) != 0) {
      return -1;
    }
  }
  int32_t condval_64;
  if ((main_Lout_strides == NULL)) {
    condval_64 = 1;
  } else {
    condval_64 = ((int32_t)((int64_t*)main_Lout_strides)[1]);
  }
  if (condval_64 != 24) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[1]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)24;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_65;
    if ((main_Lout_strides == NULL)) {
      condval_65 = 1;
    } else {
      condval_65 = ((int32_t)((int64_t*)main_Lout_strides)[1]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_65);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_122;
    result_122.type_index = kTVMFFINone;
    result_122.zero_padding = 0;
    result_122.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_122) != 0) {
      return -1;
    }
  }
  int32_t condval_66;
  if ((main_Lout_strides == NULL)) {
    condval_66 = 1;
  } else {
    condval_66 = ((int32_t)((int64_t*)main_Lout_strides)[0]);
  }
  if (condval_66 != 1536) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"strides[0]";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)1536;
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    int32_t condval_67;
    if ((main_Lout_strides == NULL)) {
      condval_67 = 1;
    } else {
      condval_67 = ((int32_t)((int64_t*)main_Lout_strides)[0]);
    }
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)condval_67);
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_123;
    result_123.type_index = kTVMFFINone;
    result_123.zero_padding = 0;
    result_123.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_123) != 0) {
      return -1;
    }
  }
  if ((uint64_t)0 != (((DLTensor*)Lout_handle)[0].byte_offset)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].byte_offset));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_byte_offset_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_byte_offset_mismatch", &__tvm_error_byte_offset_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_124;
    result_124.type_index = kTVMFFINone;
    result_124.zero_padding = 0;
    result_124.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_byte_offset_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_124) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Lout_handle)[0].device.device_id) != (((DLTensor*)Q_handle)[0].device.device_id)) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"device_id";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Q_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].device.device_id));
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = (int64_t)0;
    if (__tvm_error_expect_eq_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_expect_eq", &__tvm_error_expect_eq_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_125;
    result_125.type_index = kTVMFFINone;
    result_125.zero_padding = 0;
    result_125.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_expect_eq_packed, (TVMFFIAny*) stack_ffi_any, 5, &result_125) != 0) {
      return -1;
    }
  }
  if ((((DLTensor*)Lout_handle)[0].device.device_type) != 2) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)2;
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)(((DLTensor*)Lout_handle)[0].device.device_type));
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = (int64_t)0;
    if (__tvm_error_device_type_mismatch_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_device_type_mismatch", &__tvm_error_device_type_mismatch_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_126;
    result_126.type_index = kTVMFFINone;
    result_126.zero_padding = 0;
    result_126.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_device_type_mismatch_packed, (TVMFFIAny*) stack_ffi_any, 4, &result_126) != 0) {
      return -1;
    }
  }
  if (Lout == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = (void*)"main";
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = (void*)"Lout";
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 8;
    (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
    (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = (void*)"data pointer";
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
    (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = (int64_t)0;
    if (__tvm_error_null_ptr_packed == NULL) {
      if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_error_null_ptr", &__tvm_error_null_ptr_packed) != 0) {
        return -1;
      }
    }
    TVMFFIAny result_127;
    result_127.type_index = kTVMFFINone;
    result_127.zero_padding = 0;
    result_127.v_int64 = 0;
    if (TVMFFIFunctionCall(__tvm_error_null_ptr_packed, (TVMFFIAny*) stack_ffi_any, 3, &result_127) != 0) {
      return -1;
    }
  }
  (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = ((int64_t)dev_id);
  (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = (int64_t)0;
  if (__tvm_set_device_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "__tvm_set_device", &__tvm_set_device_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_128;
  result_128.type_index = kTVMFFINone;
  result_128.zero_padding = 0;
  result_128.v_int64 = 0;
  if (TVMFFIFunctionCall(__tvm_set_device_packed, (TVMFFIAny*) stack_ffi_any, 2, &result_128) != 0) {
    return -1;
  }
  if (Count == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[0].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[0].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[0].v_ptr) = Count;
  if (Indices == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[1].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[1].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[1].v_ptr) = Indices;
  if (K == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[2].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[2].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[2].v_ptr) = K;
  if (Lout == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_ptr) = Lout;
  if (Mout == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_ptr) = Mout;
  if (Offsets == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_ptr) = Offsets;
  if (Q == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[6].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[6].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[6].v_ptr) = Q;
  if (RowMap == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[7].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[7].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[7].v_ptr) = RowMap;
  if (U == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[8].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[8].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[8].v_ptr) = U;
  if (V == NULL) {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 0;
  } else {
    (((TVMFFIAny*)stack_ffi_any)[9].type_index) = 4;
  }
  (((TVMFFIAny*)stack_ffi_any)[9].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_int64) = 0;
  (((TVMFFIAny*)stack_ffi_any)[9].v_ptr) = V;
  (((TVMFFIAny*)stack_ffi_any)[10].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[10].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[10].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[11].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[11].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[11].v_int64) = ((int64_t)64);
  (((TVMFFIAny*)stack_ffi_any)[12].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[12].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[12].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[13].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[13].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[13].v_int64) = ((int64_t)256);
  (((TVMFFIAny*)stack_ffi_any)[14].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[14].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[14].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[15].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[15].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[15].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[16].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[16].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[16].v_int64) = ((int64_t)61632);
  (((TVMFFIAny*)stack_ffi_any)[17].type_index) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[17].v_int64) = (int64_t)0;
  if (main_kernel_packed == NULL) {
    if (TVMBackendGetFuncFromEnv(__tvm_ffi__library_ctx, "main_kernel", &main_kernel_packed) != 0) {
      return -1;
    }
  }
  TVMFFIAny result_129;
  result_129.type_index = kTVMFFINone;
  result_129.zero_padding = 0;
  result_129.v_int64 = 0;
  if (TVMFFIFunctionCall(main_kernel_packed, (TVMFFIAny*) stack_ffi_any, 17, &result_129) != 0) {
    return -1;
  }
  return 0;
}

