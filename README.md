# Binary Encoder
##### Created by Dmytro Zakharov under the supervision of Aleksandr Kuznetsov

**Binary encoder** is a part of the research project that extracts a feature vector from the face and converts it to the binary string. 

![Binary Encoder Example](https://github.com/ZamDimon/Binary-Encoder/blob/main/Images/presentation/binary_encoder.png "Binary Encoder Example") 

### Structure of a project

In project we mainly considered two models: *Keras Facenet* and *Face Recognition*. Corresponding *Jupyter*  files with the detailed descriptions of calculations and implementation (based on the [lfw people](http://vis-www.cs.umass.edu/lfw/ "Lfw people dataset source") dataset) are located in the root folder:
- `encoding_keras_facenet.ipynb` for *Keras Facenet*,
- `encoding_face_recognition.ipynb` for *Face Recognition*.

We also evaluated the accuracy of both models based on the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "CelebA dataset source") dataset. Corresponding file, though without such detailed explanations as for `encoding_X.ipynb` files, is called `celeba_estimator.ipynb`.  

To make code more readable in the *Jupyter* files, we moved all the helper functions in the separate folder `helpers`. If interested, you might check it out as well.
