"""RBE-based privacy-preserving machine learning (PPML).

Implements Section III of the paper "RBE-based privacy-preserving machine
learning model": logistic regression trained and evaluated on RBE-encrypted
tabular data, and linear-SVM inference on RBE-encrypted CIFAR-10 features.

The cryptographic core is the original RBE implementation in the `rbe`
package (src/rbe), used without modification.
"""

__version__ = "0.1.0"
