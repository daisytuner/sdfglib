[![Coverage Status](https://coveralls.io/repos/github/daisytuner/sdfglib/badge.svg?branch=main&t=Kti3Xd)](https://coveralls.io/github/daisytuner/sdfglib?branch=main)

sdfglib is a C++ library for generating stateful dataflow multigraphs (SDFG).

## Usage

### Dependencies

```
sudo apt-get install -y libgmp-dev libzstd-dev
sudo apt-get install -y nlohmann-json3-dev
sudo apt-get install -y libboost-graph-dev libboost-graph1.74.0
sudo apt-get install -y libisl-dev
```

### Build

```
mkdir build
cd build
cmake -DBUILD_TESTS:BOOL=OFF -DBUILD_BENCHMARKS:BOOL=OFF -DBUILD_BENCHMARKS_GOOGLE:BOOL=OFF -DHWY_ENABLE_TESTS=OFF ..
cmake --build .
```

## Attribution

sdfglib is implemented based on the specification of the [original paper](https://www.arxiv.org/abs/1902.10345) and its [reference implementation](https://github.com/spcl/dace).
The license of the reference implementation is included in the licenses/ folder.

If you use the sdfglib, cite the paper:
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

sdfglib is published under the new BSD license, see [LICENSE](LICENSE).
