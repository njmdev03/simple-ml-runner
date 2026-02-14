# UCI Adult Income

The UCI Adult Income dataset is based on U.S. Census data with the aim being to classify individuals into income brackets of >50K/year or <50K/year.

Note that this dataset is defined in the `adult_dataset.py` file, with data preprocessing included. The dataset includes some missing datasets which are automatically corrected by the dataset's definition. This is a great reference for creating your own datasets for use with this tool.

## MLP

With simple tabular data like this, and MLP very quickly reaches a high accuracy on this dataset. In only a couple of epochs, accuracies of >98% are possible. Each epoch also runs relatively quickly, with execution only taking a few seconds per epoch on a Nvidia 2070 Super.

## CNN

CNNs are normally used for image data, but with the UCi Adult Income dataset being tabular, it shouldn't be possible to apply a CNN. Of course, it is possible to convert each column of the input tables into a pixel in a 2d matrix, which allows for the application of a CNN. With the high performance of MLPs on this dataset, a CNN really isn't necessary, but is included to show how tabular data can be run through powerful image networks.
