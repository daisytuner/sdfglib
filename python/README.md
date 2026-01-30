# docc-compiler

A JIT compiler for Numpy-based Python programs targeting various hardware backends.

## Installation

```bash
pip install docc-compiler
```

## Features

- **JIT Compilation**: Automatically compile Python/NumPy code to optimized native code
- **Multiple Backends**: Support for CPU (OpenMP, SIMD), CUDA, and other accelerators
- **Stateful Dataflow Graphs (SDFGs)**: Based on a powerful intermediate representation for optimization
- **Performance Portability**: Write once, run optimized on different hardware

## Quick Start

```python
import numpy as np
import docc

@docc.program
def matrix_multiply(A, B):
    return A @ B

# Automatically compiled and optimized
A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
C = matrix_multiply(A, B)
```

## Requirements

- Python >= 3.11
- NumPy >= 1.19.0

## Attribution

The Python module is implemented based on the DaCe [reference implementation](https://github.com/spcl/dace).
The license of the reference implementation is included in the licenses/ folder.

If you use the Python bindings and frontend, please cite:

```bibtex
@inproceedings{dace,
  author    = {Ben-Nun, Tal and de~Fine~Licht, Johannes and Ziogas, Alexandros Nikolaos and Schneider, Timo and Hoefler, Torsten},
  title     = {Stateful Dataflow Multigraphs: A Data-Centric Model for Performance Portability on Heterogeneous Architectures},
  year      = {2019},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series = {SC '19}
}
```

## License

BSD-3 Clause
