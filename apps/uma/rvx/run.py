# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
from tvm.micro.testing.aot_test_utils import AOT_DEFAULT_RUNNER
import tvm
from tvm import relay
from backend import RvxBackend
from tvm.relay import transform
from collections import OrderedDict
import numpy as np


from tvm.testing.aot import (
    AOTTestModel as AOTModel,
    AOTTestRunner as AOTRunner,
    generate_ref_data,
    compile_and_run,
)


def create_conv2d(groups=1, runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    data0 = relay.var("data", shape=ishape, dtype=dtype)
    weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    main_f = relay.Function([data0, weight0], out)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1_data = np.random.uniform(0, 1, wshape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner

def create_qnn_conv2d(groups=1, runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "int8"
    # ishape = (1, 32, 14, 14)
    # wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    # data0 = relay.var("data", shape=ishape, dtype=dtype)
    # weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    # qnn_out = relay.qnn.conv2d(data0, weight0, 0, 0, 0.125, 0.125, channels=10, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    # out = relay.qnn.requantize(qnn_out, 0.125, 0, 0.125, 0, axis=1, out_dtype=dtype)
    data_shape = [1, 8, 32, 32]
    weight_shape = [16, 8, 3, 3]
    data = relay.var("data", shape=data_shape, dtype="int8")
    weight = relay.var("weight", shape=weight_shape, dtype="int8")
    # op0 = relay.qnn.quantize(data, relay.const(0.078), relay.const(0), out_dtype="uint8")
    # op1 = relay.qnn.quantize(weight, relay.const(0.07), relay.const(0), out_dtype="int8")
    op2 = relay.qnn.conv2d(
        data,
        weight,
        input_zero_point=relay.const(0),
        kernel_zero_point=relay.const(0),
        input_scale=relay.const(0.078),
        kernel_scale=relay.const(0.07),
        padding=[0, 0, 0, 0],
        channels=16,
        kernel_size=[3, 3],
    )
    op5 = relay.qnn.requantize(
        op2,
        input_scale=relay.const(0.05),
        input_zero_point=relay.const(0),
        output_scale=relay.const(0.21),
        output_zero_point=relay.const(61),
        out_dtype="int8",
    )

    main_f = relay.Function([data, weight], op5)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    i_data = np.random.uniform(0, 1, data_shape).astype(dtype)
    w1_data = np.random.uniform(0, 1, weight_shape).astype(dtype)
    inputs = OrderedDict([("data", i_data), ("weight", w1_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner


def create_requantize(groups=1, runner=AOT_DEFAULT_RUNNER, weight_shape=32):
    dtype = "int8"
    # ishape = (1, 32, 14, 14)
    # wshape = (32, weight_shape, 3, 3)
    pass_config = {"tir.usmp.enable": True}
    runner = AOTRunner(
        makefile=runner.makefile,
        prologue=runner.prologue,
        epilogue=runner.epilogue,
        includes=runner.includes,
        parameters=runner.parameters,
        pass_config=pass_config,
    )
    # data0 = relay.var("data", shape=ishape, dtype=dtype)
    # weight0 = relay.var("weight", shape=wshape, dtype=dtype)
    # qnn_out = relay.qnn.conv2d(data0, weight0, 0, 0, 0.125, 0.125, channels=10, kernel_size=(3, 3), padding=(1, 1), groups=groups)
    # out = relay.qnn.requantize(qnn_out, 0.125, 0, 0.125, 0, axis=1, out_dtype=dtype)
    data_shape = [1, 8, 32, 32]
    weight_shape = [16, 8, 3, 3]
    data = relay.var("data", shape=data_shape, dtype="int8")
    op5 = relay.qnn.requantize(
        data,
        input_scale=relay.const(0.05),
        input_zero_point=relay.const(0),
        output_scale=relay.const(0.21),
        output_zero_point=relay.const(61),
        out_dtype="int8",
    )

    main_f = relay.Function([data], op5)
    mod = tvm.IRModule()
    mod["main"] = main_f
    mod = transform.InferType()(mod)
    i_data = np.random.uniform(0, 1, data_shape).astype(dtype)
    w1_data = np.random.uniform(0, 1, weight_shape).astype(dtype)
    inputs = OrderedDict([("data", i_data)])
    output_list = generate_ref_data(mod, inputs)
    return mod, inputs, output_list, runner


def main():
    mod, inputs, output_list, runner = create_conv2d()  # pass
    # mod, inputs, output_list, runner = create_qnn_conv2d() # composite function is not working
    # mod, inputs, output_list, runner = create_requantize()  # partition okay

    uma_backend = RvxBackend()
    uma_backend.register()
    mod = uma_backend.partition(mod)
    print(mod)
    target = tvm.target.Target("rvx", host=tvm.target.Target("c"))

    export_directory = tvm.contrib.utils.tempdir(keep_for_debug=True).path
    print(f"Generated files are in {export_directory}")
    compile_and_run(
        AOTModel(module=mod, inputs=inputs, outputs=output_list),
        runner,
        interface_api="c",
        use_unpacked_api=True,
        target=target,
        test_dir=str(export_directory),
    )


if __name__ == "__main__":
    main()
