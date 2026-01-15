[![Coverage Status](https://coveralls.io/repos/github/daisytuner/sdfg/badge.svg?branch=main&t=Kti3Xd)](https://coveralls.io/github/daisytuner/sdfglib?branch=main)

The sdfg module is the core library for generating stateful dataflow multigraphs (SDFG).

## Dependencies

```
sudo apt-get install -y libgmp-dev libzstd-dev
sudo apt-get install -y nlohmann-json3-dev
sudo apt-get install -y libboost-graph-dev libboost-graph1.74.0
sudo apt-get install -y libisl-dev
```

## Documentation

sdfg uses [Doxygen](https://www.doxygen.nl/) for API documentation. The type system is fully documented with Doxygen comments.

### Generate Documentation

Install Doxygen:
```bash
sudo apt-get install -y doxygen graphviz
```

Generate the documentation:
```bash
doxygen Doxyfile
```

View the documentation by opening `docs/html/index.html` in a web browser.

For more details, see [DOCUMENTATION.md](DOCUMENTATION.md).

## Attribution

sdfg is implemented based on the specification of the [original paper](https://www.arxiv.org/abs/1902.10345) and its [reference implementation](https://github.com/spcl/dace).
The license of the reference implementation is included in the licenses/ folder.

If you use the sdfg, cite the paper:
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

sdfg is published under the new BSD license, see [LICENSE](LICENSE).
