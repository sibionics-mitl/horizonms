<div align="center">
 <center><h1>Welcome to HorizonMS</h1></center>
</div>

## Introduction

HorizonMS is a deep learning framework developed by PyTorch. It was developed by **Medical Imaging Technology Lab (MITL)** at [**Sibionics**](https://www.sibionicscgm.com/) and was released on December 12th, 2023. Its documentation is [here](https://horizonms.readthedocs.io/en/latest/). 

## Features

HorizonMS is a foundational library for computer vision research and it provides the following functionalities:

- It supports image classification, image segmentation, and object detection.

- It supports data augmentation by PyTorch, OpenCV, or mixture of OpenCV and PyTorch

- It configures the experiments by dictionary.

- It discriminates softmax, sigmoid, and linear activation for the network output in both losses and metrics.

- It supports well-known deep learning algorithms.

## Installation

HorizonMS supports  both pip installation and local installation. The command for installations are as follows:

**1. pip installation:**

```bash
pip install horizonms
```

**2. local installation:**

```bash
git clone https://github.com/sibionics-mitl/horizonms.git 
cd horizonms
pip install . # method 1, or
python steup.py install # method 2
```

## Contributors

[Juan Wang](https://github.com/wangjuan313), [Xiaochun Pan](), [Peihai Huang](), [Wanduo Zheng]()

## Citation

```latex
@misc{hms,
 title={HorizonMS},
 author={HorizonMS Contributors},
 howpublished = {\url{https://github.com/sibionics-mitl/horizonms}},
 year={2023}
}
```
