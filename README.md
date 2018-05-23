# Tranus Python LCAL module
 
This project aims to implement the Tranus (http://www.tranus.com/tranus-english/download-install) software in Python. To the current date, only the Lcan Use Calibration (LCAL) module is available. This implementation integrates with the other Tranus modules, and can be used as an standalone program.

The main features of this program, is that the implementation is different from the original Tranus software, reformulating the whole calibration as an optimisation problem. The internal parameters are computed minimising a cost function enabling semi-automatic parameter calibration.
Details on the mathematics of the Tranus implementation and Python Tranus con be found here:
* Detailed scientific [article](https://www.sciencedirect.com/science/article/pii/S0198971517302181?via%3Dihub) about the Tranus Python implementation.
* [Mathematical description](http://docs.google.com/viewer?a=v&pid=sites&srcid=ZGVmYXVsdGRvbWFpbnx0cmFudXNtb2RlbHxneDo3YWQzYTk0OTkxN2RlN2Rj) of the Tranus software implementation.



## Getting Started

### Prerequisites
This program is a Python module, the following python packages are required:
* numpy
* pandas 
* scipy
* cython

It is recommended to install anaconda python : https://anaconda.org/anaconda/python and you are good.
If you don't use anaconda, you can also install using the supplied [setup.py](setup.py) doing:

```python setup.py --install```











## Authors

* **Thomas Capelle** - *Initial work* - [Cape](https://gitlab.inria.fr/tcapelle)

## License

This project is licensed under some  License

## Acknowledgments

* Thank you Fausto Lo Feudo / Brian Morton / Peter Sturm / Arthur Vidard / Tomas de la Barra
