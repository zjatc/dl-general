* Check version of CUDA and cuDNN:

 CUDA: 
```
cat /usr/local/cuda/version.txt
nvcc --version
```
 cuDNN: 
```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
cat /usr/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

* CUDA docker images:
 
 'runtime' doesn't contain nvcc and valid cuDNN; 
 'devel' contains nvcc and valid cuDNN

* Build Tensorlfow Docker
Dockerfile
```
FROM tensorflow/tensorflow:1.10.0-gpu-py3
MAINTAINER Jason Zhang

RUN apt-get update && apt-get install -y curl vim git make
RUN mkdir -p /jason
EXPOSE 8888
EXPOSE 6006
WORKDIR /jason
```
