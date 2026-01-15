The python module contains the Python frontend of the Daisytuner Optimizing Compiler Collection.
The frontend just-in-time compiles Python functions to SDFGs, offloading Python code to various hardware backends.

## Attribution

The Python frontend is implemented based on the DaCe [reference implementation](https://github.com/spcl/dace).
The license of the reference implementation is included in the licenses/ folder.

If you use the Python frontend, cite the paper:
```bibtex
@inproceedings{dace,
  author    = {Ben-Nun, Tal and de~Fine~Licht, Johannes and Ziogas, Alexandros Nikolaos and Schneider, Timo and Hoefler, Torsten},
  title     = {Stateful Dataflow Multigraphs: A Data-Centric Model for Performance Portability on Heterogeneous Architectures},
  year      = {2019},
  booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
  series = {SC '19}
}
```
