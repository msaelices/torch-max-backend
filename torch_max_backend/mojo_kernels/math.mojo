from compiler import register
from gpu.host import DeviceContext
from gpu.id import block_idx
from layout import LayoutTensor
from math import ceil
from runtime.asyncrt import DeviceContextPtr
from tensor_internal import InputTensor, OutputTensor, foreach

from utils.index import IndexList


@compiler.register("ceil")
struct CeilKernel:
    @staticmethod
    fn execute[
        dtype: DType,
        rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype = dtype, rank = rank],
        x: InputTensor[dtype = dtype, rank = rank],
        ctx: DeviceContextPtr,
    ) raises:

        @parameter
        @always_inline
        fn elementwise_ceil[
            width: Int
        ](idx: IndexList[x.rank]) -> SIMD[x.dtype, width]:
            return ceil(x.load[width](idx))

        foreach[elementwise_ceil, target=target](output, ctx)