# Multi-config GEMM/GEMV on Phoenix NPU — Strategy A milestones (A)+(C)

Date: 2026-06-13. Machine: Ryzen 7 7840HS, Phoenix NPU (NPU1/AIE2), fw 1.5.5.391.
Toolchain: mlir-aie v1.3.2 (IRON) + llvm-aie/Peano, Arch Linux, Python 3.14.
All results verified EXACT vs numpy (i32 accumulator). Real hardware (strace-confirmed
AMDXDNA_EXEC_CMD).

## (A) Throughput scaling: 1-col vs 4-col

| design | shape | cores | time/op | GOPS |
|---|---|---|---|---|
| single_core | 256³ | 1 | 1.10 ms | 30.5 |
| **whole_array** | **512³** | **16 (4×4)** | **1.01 ms** | **264.7** |

4-col design is **8.7× single-core throughput** — near-linear across 16 cores.

## (C) The inference op: batch=1 matrix-vector (GEMV)

| design | shape | cores | time/op | GOPS | vs CPU |
|---|---|---|---|---|---|
| matrix_vector | 288², batch=1 | 1 | 0.376 ms | 0.4 | **8× SLOWER** |

**Key finding:** batch=1 is host↔NPU round-trip bound. The 0.376ms is ~all
PCIe/BO-sync overhead (166K MACs is trivial compute). The NPU LOSES for
single-query serving unless the whole model fuses into one dispatch. See
[ARCHITECTURE.md](../ARCHITECTURE.md).

## Batched serving (the viable path)

The `npu_server.py` scaffold routes batch=1→CPU, batch≥threshold→NPU:

```
batch=512 (4-col GEMM): NPU 1.84 ms (146 GOPS)  vs  CPU 64.0 ms
                         => NPU 34.7x faster for batched embedding
```

## Takeaway
- **NPU wins decisively for batched workloads** (index building): 34.7× over CPU.
- **NPU loses for batch=1** (live query): 8× slower than CPU, round-trip bound.
- The serving layer must batch + route. Full-model single-query NPU serving
  needs op fusion (Strategy B's domain).
