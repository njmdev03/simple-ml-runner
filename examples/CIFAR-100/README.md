# CIFAR-100 Image Classification

The CIFAR-100 dataset provides a more complex example than MNIST, with many more classes and more complex input data. The examples here use a larger MLP and more complex CNN than the other examples to account for this.

## MLP

The CIFAR-100 dataset is verey complicated, so simple MLPs tend not to perform terribly well. Instead it is much better to use other methods such as CNNs or Vision Transformers. The MLP example is mostly provided to show how even with a relatively complex network, only a low accuracy of <30% is achievable using this method.

## CNN

CNNs perform much better on the CIFAR-100 dataset. With this method a much better accuracy over 90% is relatively easily achievable even with a simple optimizer. It is likely that training over more epochs with a learning rate decay would allow for much higher accuracy.

## Vision Transformer


