# Binary Encoder
#### Created by Dmytro Zakharov under the supervision of Aleksandr Kuznetsov

**Binary encoder** is a part of the research project that extracts a feature vector from the face and converts it to the binary string.

### Examples

#### Binary string for a pair of images of the same person

![Binary Encoder Example](https://github.com/ZamDimon/Binary-Encoder/blob/main/Images/presentation/same_people.png "Binary Encoder Example")

#### Binary string for a pair of images of two different people

![Binary Encoder Example](https://github.com/ZamDimon/Binary-Encoder/blob/main/Images/presentation/different-people.png "Binary Encoder Example")

### Structure of a project

In project we mainly considered two models: *Keras Facenet* and *Face Recognition*. In `csv_loader.ipynb` we convert images from the datasets to the feature vectors and save them into the separate csv file for further processing. In `indicator_functions.ipynb` we consider vector converter functions and assess their accuracy.

We evaluated the accuracy of both models based on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "CelebA dataset source") and [lfw](http://vis-www.cs.umass.edu/lfw/) datasets.

To make code more readable in the *Jupyter* files, we moved all the helper functions in the separate folder `internal`. If interested, you might check it out as well.
