# ===----------------------------------------------------------------------=== #
# Convolution Backward Kernels for PyTorch MAX Backend
#
# Implements grad_weight computation for 2D convolution backward pass.
# ===----------------------------------------------------------------------=== #

from compiler import register
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor
from utils.index import IndexList


@compiler.register("conv2d_backward_weight")
struct Conv2dBackwardWeightKernel:
    """
    Computes grad_weight for 2D convolution backward pass.

    Algorithm:
        grad_weight[f, c, r, s] = Σ(grad_output[n, f, ho, wo] * input[n, c, h, w])
        where h = ho * stride_h - pad_h + r * dil_h
              w = wo * stride_w - pad_w + s * dil_w

    Layouts:
        grad_output: NCHW [N, F, HO, WO]
        input: NCHW [N, C, H, W]
        grad_weight: OIHW [F, C, R, S]

    This is essentially a correlation between input and grad_output.
    """

    @staticmethod
    fn execute[
        dtype: DType,
        rank_grad_out: Int,  # Should be 4 for NCHW
        rank_input: Int,     # Should be 4 for NCHW
        rank_grad_weight: Int,  # Should be 4 for OIHW
        //,
        target: StaticString,
        stride_h: Int,
        stride_w: Int,
        pad_h: Int,
        pad_w: Int,
        dil_h: Int,
        dil_w: Int,
    ](
        grad_weight: OutputTensor[dtype=dtype, rank=rank_grad_weight],
        grad_output: InputTensor[dtype=dtype, rank=rank_grad_out],
        input: InputTensor[dtype=dtype, rank=rank_input],
        ctx: DeviceContextPtr,
    ) raises:
        """
        Execute the grad_weight computation.

        Dimensions:
            grad_output: [N, F, HO, WO]
            input: [N, C, H, W]
            grad_weight: [F, C, R, S]
        """

        # Extract dimensions from tensor spec
        # grad_output dimensions
        var N = Int(grad_output.spec.shape[0])
        var F = Int(grad_output.spec.shape[1])
        var HO = Int(grad_output.spec.shape[2])
        var WO = Int(grad_output.spec.shape[3])

        # input dimensions
        var C = Int(input.spec.shape[1])
        var H = Int(input.spec.shape[2])
        var W = Int(input.spec.shape[3])

        # grad_weight dimensions
        var R = Int(grad_weight.spec.shape[2])
        var S = Int(grad_weight.spec.shape[3])

        # Compute each element of grad_weight
        # Note: This is a naive implementation. For production, we'd want:
        # - Parallelization over F dimension
        # - SIMD vectorization over C dimension
        # - Cache-aware tiling
        # - GPU-specific optimizations

        for f in range(F):
            for c in range(C):
                for r in range(R):
                    for s in range(S):
                        var accumulator = Scalar[dtype](0)

                        # Accumulate over batch and spatial output dimensions
                        for n in range(N):
                            for ho in range(HO):
                                for wo in range(WO):
                                    # Compute corresponding input position
                                    var h = ho * stride_h - pad_h + r * dil_h
                                    var w = wo * stride_w - pad_w + s * dil_w

                                    # Bounds check
                                    if h >= 0 and h < H and w >= 0 and w < W:
                                        # Load values
                                        var grad_out_idx = IndexList[4](n, f, ho, wo)
                                        var input_idx = IndexList[4](n, c, h, w)

                                        var grad_out_val = grad_output.load(grad_out_idx)
                                        var input_val = input.load(input_idx)

                                        accumulator += grad_out_val * input_val

                        # Store result
                        var weight_idx = IndexList[4](f, c, r, s)
                        grad_weight.store(weight_idx, accumulator)
