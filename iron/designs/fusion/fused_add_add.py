# fused_add_add.py — OP-FUSION EXPERIMENT for the Phoenix NPU1
#
# D = (A + B) + C  as two chained stages; intermediate T = A+B is ON-DEVICE only.
# Minimal topology (2 workers, 1 logical path) to test the fusion mechanism cleanly.
from ml_dtypes import bfloat16
import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_


def my_fused_add_add(dev, num_elements):
    # Single column, tile-per-stage. Minimal DMA footprint.
    per_tile_elements = 1024
    n = per_tile_elements
    if num_elements % n != 0:
        raise ValueError(f"num_elements ({num_elements}) must be a multiple of {n}")
    N_div_n = num_elements // n
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    # Host inputs
    of_A = ObjectFifo(tile_ty, name="A")     # host -> stage1
    of_B = ObjectFifo(tile_ty, name="B")     # host -> stage1
    of_C = ObjectFifo(tile_ty, name="C")     # host -> stage2
    # ON-DEVICE intermediate (never filled/drained from host)
    of_T = ObjectFifo(tile_ty, name="T", depth=2)   # stage1 -> stage2
    # Host output
    of_D = ObjectFifo(tile_ty, name="D")     # stage2 -> host

    add_kernel = Kernel("eltwise_add_bf16_vector", "add.o", [tile_ty, tile_ty, tile_ty])

    def core_add(in1, in2, out, k):
        for _ in range_(N_div_n):
            a = in1.acquire(1); b = in2.acquire(1); o = out.acquire(1)
            k(a, b, o)
            in1.release(1); in2.release(1); out.release(1)

    # Stage 1: A + B -> T   (worker on one compute tile)
    w1 = Worker(core_add, [of_A.cons(), of_B.cons(), of_T.prod(), add_kernel])
    # Stage 2: T + C -> D   (worker on another compute tile; T arrives on-device)
    w2 = Worker(core_add, [of_T.cons(), of_C.cons(), of_D.prod(), add_kernel])

    # tap transfers ALL num_elements (each stage's worker processes them in N_div_n tiles)
    tap = TensorAccessPattern((1, num_elements), 0, [1, 1, 1, num_elements], [0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty, tensor_ty) as (A, B, C, D):
        rt.start(w1, w2)
        tg = rt.task_group()
        rt.fill(of_A.prod(), A, tap, task_group=tg)
        rt.fill(of_B.prod(), B, tap, task_group=tg)
        rt.fill(of_C.prod(), C, tap, task_group=tg)
        rt.drain(of_D.cons(), D, tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()


print(my_fused_add_add(NPU1(), 4096))
