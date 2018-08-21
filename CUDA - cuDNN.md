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
