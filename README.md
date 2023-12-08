# Fast and Accurate EM Image Registration

**[Fast and Accurate Electron Microscopy Image Registration with 3D Convolution](https://link.springer.com/chapter/10.1007/978-3-030-32239-7_53)**

Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2019, Oral Presentation

Shenglong Zhou, Zhiwei Xiong, Chang Chen, Dong Liu, Yueyi Zhang, Zheng-Jun Zha, Feng Wu

University of Science and Technology of China (USTC)


## Introduction

![framework](https://github.com/Blcony/FAEMReg/assets/26156941/b3956c8d-14f7-4af4-9f4d-402c168e6acf)

We propose an unsupervised deep learning method for serial electron microscopy (EM) image registration with fast speed and high accuracy. 
Current registration methods are time-consuming in practice due to the iterative optimization procedure. 
We model the registration process as a parametric function in the form of convolutional neural networks, and optimize its parameters based on features extracted from training serial EM images in a training set. Given a new series of EM
images, the deformation field of each serial image can be rapidly generated through the learned function. 
Specifically, we adopt a spatial transformer layer to reconstruct features in the subject image from the reference ones while constraining smoothness on the deformation field.
Moreover, for the first time, we introduce the 3D convolution layer to learn the relationship between several adjacent images, which effectively reduces error accumulation in serial EM image registration.


## Requirements
The packages and their corresponding version we used in this repository are listed below.
- Python 3
- Tensorflow 1.7.0
- Numpy
- SimpleITK

## Note

Considering that the code in the repo relies on Tensorflow and is relatively old, we highly recommend that you follow our latest EM registration work [Electron Microscopy Image Registration Using Correlation Volume (ISBI 2023)](https://ieeexplore.ieee.org/abstract/document/10230498) along with the [source code](https://github.com/llliuxz/EMReg).

Please be free to contact us by e-mail (slzhou96@mail.ustc.edu.cn) or WeChat (slzhou96) if you have any questions.

## Citation
If you find this work or code is helpful in your research, please cite:
```
@inproceedings{zhou2019fast,
  title={Fast and accurate electron microscopy image registration with 3D convolution},
  author={Zhou, Shenglong and Xiong, Zhiwei and Chen, Chang and Chen, Xuejin and Liu, Dong and Zhang, Yueyi and Zha, Zheng-Jun and Wu, Feng},
  booktitle={Medical Image Computing and Computer Assisted Intervention--MICCAI 2019: 22nd International Conference, Shenzhen, China, October 13--17, 2019, Proceedings, Part I 22},
  pages={478--486},
  year={2019},
  organization={Springer}
}
```
