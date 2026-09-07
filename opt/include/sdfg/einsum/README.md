# Einsum

*Einsum* (short for **Einstein summation**) is a compact notation for describing
operations on tensors (multi-dimensional arrays). Instead of writing explicit
loops, you label each dimension of every operand with an index and let the
notation describe how those indices relate.

A matrix multiplication `C = A · B` is written as:

```
C[i,j] = A[i,k] * B[k,j]   summed over k
```

Read this as: *for every output element `C[i,j]`, multiply the matching
elements of `A` and `B` and sum the products over the shared index `k`.*

The core rules are:

- **Indices that appear in the output** (`i`, `j`) select *where* results are
  written. Each combination produces one output element.
- **Indices that appear only on the inputs** (`k`) are **summation indices**:
  the computation reduces (sums) over every value they take.
- **Repeated input indices** describe how operands are contracted together.

With this handful of rules a single expression can express matrix
multiplication, dot products, outer products, transposition, trace, batched
contractions, and more.

## Einsum Detection: Lifting loops into tensor operations

Numerical code usually expresses tensor math as explicit, hand-written loop
nests:

```c
for (i = 0; i < N; i++)
  for (j = 0; j < M; j++)
    for (k = 0; k < K; k++)
      C[i][j] += A[i][k] * B[k][j];
```

At the loop level, a compiler only sees scalar loads, a multiply, an add, and
stores. The fact that this *is* a matrix multiplication — and that a highly
tuned `gemm` kernel exists for it — is lost.

The purpose of this module is to **raise the level of abstraction**: detect the
tensor-algebra pattern buried in the loops and replace it with a single
`EinsumNode`. Once the computation is expressed as an einsum, the compiler can:

- **Map it to optimized libraries** (BLAS `dot`, `gemm`, ...) that are far
  faster than naive loops.
- **Reason about it as one operation** instead of many scalar statements
  (e.g. estimate FLOPs, fuse/extend contractions).
- **Re-lower it to loops** when no better target is available, so nothing is
  ever lost by lifting.
