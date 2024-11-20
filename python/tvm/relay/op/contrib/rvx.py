"""RISC-V with Customized Instruction"""
import tvm.ir
from tvm.target import Target
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ...dataflow_pattern import is_constant, is_op, wildcard
from .register import register_pattern_table

tvm._ffi._init_api("relay.ext.rvx.transform", __name__)


def enabled():
    return "rvx" in Target.list_kinds()


def partition_for_rvx(mod, params=None, mod_name="default", **opts):
    """Partition the graph greedily offloading supported
    operators on RVX

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    mod_name: str, optional
        The module name

    Returns
    -------
    ret : Module
        annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("rvx"),
            transform.PartitionGraph(mod_name=mod_name),
            # GenerateCMSISNNConstants(),
            # CMSISNNFusePads(),
            # ScalarToTensorConstants(),
            # ExtractConstantsFromPartitionedFunction(),
            transform.InferType(),
        ]
    )
    return seq(mod)


@register_pattern_table("RVX")
def pattern_table():
    """Get the RVX library pattern table."""
    def qnn_conv2d_pattern(with_pad):
        """Create pattern for qnn.conv2D with optional pad and/or optional fused relu."""
        conv2d_input = wildcard()
        if with_pad:
            conv2d_input = is_op("nn.pad")(wildcard(), is_constant())
        qnn_conv2d = is_op("qnn.conv2d")(
            conv2d_input,
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
            is_constant(),
        )
        bias_add = is_op("nn.bias_add")(qnn_conv2d, is_constant())
        req = is_op("qnn.requantize")(
            qnn_conv2d | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
        )
        clip_or_req = req.optional(is_op("clip"))
        return clip_or_req

    def check_qnn_conv2d(pattern):
        """Check if the Conv2D is supported by rvx."""
        if str(pattern.op.name) == "clip":
            relu = pattern
            requantize = relu.args[0]
        else:
            requantize = pattern
        requantize_input = requantize.args[0]
        bias_add = None
        if str(requantize_input.op.name) == "nn.bias_add":
            bias_add = requantize_input
            conv2d = bias_add.args[0]
        else:
            conv2d = requantize_input
        conv2d_input = conv2d.args[0]
        conv2d_weight = conv2d.args[1]

        # check if depthwise Conv2D
        kernel_layout = conv2d.attrs.kernel_layout
        pos_o = kernel_layout.index("O")
        groups = conv2d.attrs.groups
        is_depthwise = False
        if groups == int(conv2d_input.checked_type.shape[3]) and groups == int(
            conv2d_weight.checked_type.shape[pos_o]
        ):
            is_depthwise = True

        # check if dtypes are supported for the following entities
        # (input_dtype, weight_dtype, bias_dtype, out_dtype, pattern_dtype)
        are_dtypes_valid = False
        conv2d_input_dtype = conv2d_input.checked_type.dtype
        if bias_add:
            bias_dtype = bias_add.args[1].checked_type.dtype
        else:
            # this is only to enable to following check that validates all sorts of dtypes
            bias_dtype = "int32" if conv2d_input_dtype == "int8" else "int64"
        valid_dtypes = None
        if conv2d_input_dtype == "int8":
            valid_dtypes = ("int8", "int8", "int32", "int32", "int8")
        elif conv2d_input_dtype == "int16":
            valid_dtypes = ("int16", "int8", "int64", "int64", "int16")

        if (
            conv2d_input_dtype,
            conv2d_weight.checked_type.dtype,
            bias_dtype,
            conv2d.attrs.out_dtype,
            pattern.checked_type.dtype,
        ) == valid_dtypes:
            are_dtypes_valid = True

        # input_zero_point should be 0 when int16
        valid_input_zp = True
        if conv2d_input_dtype == "int16" and conv2d.args[2].data.numpy().item(0) != 0:
            valid_input_zp = False

        # kernel zero_point should be 0
        kernel_zp = conv2d.args[3].data.numpy()
        kernel_zp = [kernel_zp] if kernel_zp.ndim == 0 else kernel_zp

        # combination of all checks to decide if pattern is eligible for partitioning
        ret = (
            are_dtypes_valid
            and valid_input_zp
            and all([zp == 0 for zp in kernel_zp])
            and (not is_depthwise or bias_add is not None)
        )
        return ret
    
    def qnn_fully_connected_pattern():
        """Create pattern for qnn.dense with optional Relu."""
        qnn_fc = is_op("qnn.dense")(
            wildcard(), is_constant(), is_constant(), is_constant(), is_constant(), is_constant()
        )
        bias_add = is_op("nn.bias_add")(qnn_fc, is_constant())
        req = is_op("qnn.requantize")(
            qnn_fc | bias_add, is_constant(), is_constant(), is_constant(), is_constant()
        )
        clip_or_req = req.optional(is_op("clip"))
        return clip_or_req

    def check_qnn_fully_connected(pattern):
        """Check if the fully connected is supported by CMSIS-NN."""
        if str(pattern.op.name) == "clip":
            relu = pattern
            requantize = relu.args[0]
        else:
            requantize = pattern
        requantize_input = requantize.args[0]
        bias_add = None
        if str(requantize_input.op.name) == "nn.bias_add":
            bias_add = requantize_input
            fc = bias_add.args[0]
        else:
            fc = requantize_input
        fc_input = fc.args[0]
        fc_weight = fc.args[1]

        are_dtypes_valid = False
        fc_input_dtype = fc_input.checked_type.dtype
        if bias_add:
            bias_dtype = bias_add.args[1].checked_type.dtype
        else:
            bias_dtype = "int32" if fc_input_dtype == "int8" else "int64"

        valid_dtypes = None
        if fc_input_dtype == "int8":
            valid_dtypes = ("int8", "int8", "int32", "int32", "int8")
        elif fc_input_dtype == "int16":
            valid_dtypes = ("int16", "int8", "int64", "int64", "int16")

        if (
            fc_input_dtype,
            fc_weight.checked_type.dtype,
            bias_dtype,
            fc.attrs.out_dtype,
            pattern.checked_type.dtype,
        ) == valid_dtypes:
            are_dtypes_valid = True

        # kernel zero_point should be 0
        kernel_zp = fc.args[3].data.numpy().item(0)

        return are_dtypes_valid and kernel_zp == 0
    
    return [
        ("rvx.qnn_conv2d", qnn_conv2d_pattern(with_pad=False), check_qnn_conv2d),
        # ("rvx.qnn_conv2d", qnn_conv2d_pattern(with_pad=True), check_qnn_conv2d_pad),
        ("rvx.qnn_fully_connected", qnn_fully_connected_pattern(), check_qnn_fully_connected),
        # ("rvx.qnn_avg_pool2d", qnn_avg_pool2d_pattern(), check_qnn_avg_pool2d),
        # ("rvx.qnn_max_pool2d", qnn_max_pool2d_pattern(), check_qnn_max_pool2d),
        # ("rvx.qnn_mul", binary_op_pattern("mul"), check_qnn_binary_op),
        # ("rvx.qnn_add", binary_op_pattern("add"), check_qnn_binary_op),
        # ("rvx.qnn_softmax", qnn_softmax_pattern(), check_qnn_softmax),
    ]