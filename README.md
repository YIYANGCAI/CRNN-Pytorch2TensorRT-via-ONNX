# CRNN-Pytorch2TensorRT-via-ONNX
the repository is about the conversion of CRNN model, which is widely used for text recognition. the CRNN model is converted from PyTorch to TensorRT via ONNX

## Installation
To run this project, some packages with special version will be installed:
|Package Name|Version|Description|
|:-:|:-:|:-:|
|PyTorch|1.3.1|Deep learning tool|
|onnx|1.6.0|Conversion medium for different frames|
|onnxruntime|1.3.0|Do the inference of an onnx file|
|TensorRT|7.0.0.11|Plan to Optimize the inference efficiency on Nvidia GPUs|
|pillow|7.1.2|Image processing|
|opencv-python|4.2.0.34|Image Processing|
|pycuda|2019.1.2|Help to allocate memory on cpu and gpu when using TensorRT for inference|
If other packages are to installed, please follow the information in CMD

The project is greatly helped by the project of [CRNN-Pytorch](https://github.com/meijieru/crnn.pytorch)
Thanks for meijieru's contributions!

## Usage
### Get the .pth model
```
git clone https://github.com/YIYANGCAI/CRNN-Pytorch2TensorRT-via-ONNX
cd CRNN-Pytorch2TensorRT-via-ONNX
```
Find the pretrained model from meijieru's project mentioned above
Copy the pth model into ./data in the project

### Run conversion of .pth to .onnx
```
python pytorch2onnx.py
```
Then you can find onnx model **./new_crnn.onnx** is created.
You can test the input and output of pth and onnx model by doing their inferences.

### Run conversoin of .onnx to TensorRT engine
```
python onnx2tensorrt.py
```

I have test the inference time on TITAN-RTX and the inference time can be fast as 3 ms, however, the inference by INT8 is not applied in this project, I will do this later.
