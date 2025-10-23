import math
from compiler import register
from gpu.id import global_idx
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor, foreach
from utils import IndexList, StaticTuple

@register("gelu_backward")
struct GeluBackwardKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
        approximate: StaticString,
    ](
        output: OutputTensor[dtype = dtype, rank = rank],
        grad_output: InputTensor[dtype = dtype, rank = rank],
        input: InputTensor[dtype = dtype, rank = rank],
        ctx: DeviceContextPtr,
    ) raises:
        print("output.shape", output)

        @parameter
        @always_inline
        fn func[
            width: Int,
        ](idx: IndexList[rank]) -> SIMD[dtype, width]:
            var i = idx[0]
            var x = input.load[width](idx)
            var grad_out = grad_output.load[width](idx)
            var result = SIMD[dtype, width]()

            if approximate == "none":
                for i in range(width):
                    # Exact GELU backward using error function
                    # Formula: grad = dy * (CDF + x * PDF)
                    # where CDF = 0.5 * (1 + erf(x * M_SQRT1_2))
                    #       PDF = (M_2_SQRTPI * M_SQRT1_2 * 0.5) * exp(-0.5 * x²)

                    # Constants from PyTorch implementation
                    alias M_SQRT1_2 = 0.7071067811865476  # sqrt(1/2) = 1/sqrt(2)
                    alias PDF_CONSTANT = 0.39894228040143276  # M_2_SQRTPI * M_SQRT1_2 * 0.5

                    # Compute CDF term: 0.5 * (1 + erf(x * M_SQRT1_2))
                    cdf = 0.5 * (1.0 + math.erf(x * M_SQRT1_2))

                    # Compute PDF term: PDF_CONSTANT * exp(-0.5 * x²)
                    x_squared = x * x
                    pdf = PDF_CONSTANT * math.exp(-0.5 * x_squared)

                    # Gradient: grad_out * (CDF + x * PDF)
                    result = grad_out[i] * (cdf + x * pdf)
            return result

        foreach[
            func,
            target=target,
        ](output, ctx)

