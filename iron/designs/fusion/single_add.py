# single_add.py — baseline: ONE elementwise add, 1 column, 4096 elements.
# Same topology/DMA footprint as fused_add_add.py minus stage 2, for a fair
# 1-dispatch-vs-2-dispatch comparison.
from ml_dtypes import bfloat16
import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_single_add(dev, num_elements):
    per_tile_elements = 1024
    n = per_tile_elements
    if num_elements % n != 0:
        raise ValueError(f"num_elements must be a multiple of {n}")
    N_div_n = num_elements // n
    dtype = bfloat16
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    of_A = ObjectFifo(tile_ty, name="A")
    of_B = ObjectFifo(tile_ty, name="B")
    of_C = ObjectFifo(tile_ty, name="C")
    add_kernel = Kernel("eltwise_add_bf16_vector", "add.o", [tile_ty, tile_ty, tile_ty])

    def core_add(in1, in2, out, k):
        for _ in range_(N_div_n):
            a = in1.acquire(1); b = in2.acquire(1); o = out.acquire(1)
            k(a, b, o); in1.release(1); in2.release(1); out.release(1)

    w = Worker(core_add, [of_A.cons(), of_B.cons(), of_C.prod(), add_kernel])
    # tap transfers ALL num_elements (worker processes them in N_div_n tiles of per_tile)
    tap = TensorAccessPattern((1, num_elements), 0, [1, 1, 1, num_elements], [0, 0, 0, 1])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(w)
        tg = rt.task_group()
        rt.fill(of_A.prod(), A, tap, task_group=tg)
        rt.fill(of_B.prod(), B, tap, task_group=tg)
        rt.drain(of_C.cons(), C, tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)
    return Program(dev, rt).resolve_program()


print(my_single_add(NPU1(), 4096))
