/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/rvx/rvx_kernel.h
 * \brief Use abstracted rvx library kernels.
 */

#ifndef TVM_RUNTIME_CONTRIB_RVX_RVX_KERNEL_H_
#define TVM_RUNTIME_CONTRIB_RVX_RVX_KERNEL_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>

#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

TVM_DLL void rvx_conv2d(uint8_t* data, uint8_t* weights, uint8_t* out, int p_N_, int p_C_,
                                    int p_H_, int p_W_, int p_O_, int p_G_, int p_Ph0_, int p_Pw0_,
                                    int p_Ph1_, int p_Pw1_, int p_Kh_, int p_Kw_, int p_Sh_,
                                    int p_Sw_);

TVM_DLL void rvx_fused_conv2d_relu(uint8_t* data, uint8_t* weights, uint8_t* out, int p_N_,
                                               int p_C_, int p_H_, int p_W_, int p_O_, int p_G_,
                                               int p_Ph0_, int p_Pw0_, int p_Ph1_, int p_Pw1_,
                                               int p_Kh_, int p_Kw_, int p_Sh_, int p_Sw_);

TVM_DLL void rvx_fused_conv2d_bias_relu(uint8_t* data, uint8_t* weights, uint8_t* bias,
                                                    uint8_t* out, int p_N_, int p_C_, int p_H_,
                                                    int p_W_, int p_O_, int p_G_, int p_Ph0_,
                                                    int p_Pw0_, int p_Ph1_, int p_Pw1_, int p_Kh_,
                                                    int p_Kw_, int p_Sh_, int p_Sw_);

TVM_DLL void rvx_dense(uint8_t* data, uint8_t* weight, uint8_t* out, int p_B_, int p_I_,
                                   int p_O_);

TVM_DLL void rvx_relu(uint8_t* data, uint8_t* out, std::vector<int64_t> shape);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_CONTRIB_DNNL_DNNL_KERNEL_H_
