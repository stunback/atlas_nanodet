# atlas_nanodet

## Introduction
This is a project to deploy Nanodet on Atlas200DK.  
  
The code contains two achiving method at opencv and CANN(Compute Architecture for Neural Networks) in Python.  
  
我很喜欢Nanodet这个算法，因此把它部署到了Atlas200DK上，分别使用python opencv和华为CANN框架实现了嵌入上nanodet模型的推理。  

## Dependents:  
Before trying, you should have prepared for the running environments as below, **especially the pyACL for CANN**  
```
opencv_python>=4.0
opencv_contrib_python>=4.0
pyACL
```

## Usage:  

```  
git clone  https://github.com/stunback/atlas_nanodet.git

# your '~/.bashrc' shoule have the CANN environments setup
source ~/.bashrc  
cd atlas_nanodet/src

# inference at CANN
python3 acl_nanodet.py

# inference at opencv
python3 main_nanodet.py
```  

## Results
The onnx model I use is from the origin nanodet repo,
```
https://github.com/RangiLyu/nanodet.git
```
So I just test the model inference time without pre and post process,  

**nanodet_s(320)**    
onnx=170.5ms      
cann=8.5ms     

**nanodet_m(416)**   
onnx=280.0ms     
cann=11.0ms     

## More About
My Blog:    
https://blog.csdn.net/qq_41035283/article/details/119150751   