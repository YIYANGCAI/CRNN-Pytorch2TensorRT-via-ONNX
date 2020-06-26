
# This sample uses an CRNN text recognition  Model to create a TensorRT Inference Engine
from datetime import datetime
import random
from PIL import Image
import numpy as np
import copy

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

import sys, os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

import onnxruntime
import dataset

class ModelData(object):
    MODEL_PATH = "./new_crnn.onnx"
    INPUT_SHAPE = (1, 1, 32, 100)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        #self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        #img = self.toTensor(img)
        #img.sub_(0.5).div_(0.5)
        return img

# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# The Onnx path is used for Onnx models.
def build_engine_onnx(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = common.GiB(1)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        return builder.build_cuda_engine(network)

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        #n, c, h, w = ModelData.INPUT_SHAPE
        #transformer = resizeNormalize((w, h))
        #image = transformer(image)
        #image_np = (np.asarray(image)-0.5)/0.5
        #image_np_with_batch = np.expand_dims(image_np, 0)
        transformer = dataset.resizeNormalize((100, 32))
        image = transformer(image)
        image = image.view(1, *image.size())

        #image_arr = np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(ModelData.DTYPE)).ravel()
        # This particular ResNet50 model requires some preprocessing, specifically, mean normalization.
        return image

    # Normalize the image and copy to pagelocked memory.
    #print("the pagelocked_buffer:{}, the shape:{}".format(pagelocked_buffer, pagelocked_buffer.shape))
    image =  normalize_image(Image.open(test_image).convert('L'))
    trt_inputs = np.array(image).astype(np.float32)
    print('##########################the tensorrt input##########################\n{}'.format(trt_inputs))
    #print("the images:",image_host)
    np.copyto(pagelocked_buffer, trt_inputs)
    return test_image

def runOnnxModel(onnx_path = './onnx2trt/data/crnn_data/new_crnn.onnx', image_path = './data/demo.py'):
    # loading the image
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(image_path).convert('L')
    image = transformer(image)
    image = image.view(1, *image.size())
    onnx_inputs = copy.deepcopy(image)
    print('***************************the onnx inputs***************************\n{}'.format(onnx_inputs))
    # loading the onnx model
    session = onnxruntime.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    #print('\t>>input: {}, {}, {}'.format(session.get_inputs()[0].name, session.get_inputs()[0].shape, session.get_inputs()[0].type))
    _outputs = session.get_outputs()
    for kk in range(len(_outputs)):
        _out = _outputs[kk]
        #print('\t>>out-{}: {}, {}, {}'.format(kk, _out.name, _out.shape, _out.type))

    x = np.array(onnx_inputs).astype(np.float32)

    p = session.run(None, {input_name: x})
    out1 = p[0]
    #print('============================================================================')
    #print('>>summary:')
    #print('onnx out: {} \n{}'.format(np.shape(out1), out1))

def runTensorRTModel(onnx_path = './new_crnn.onnx', image_path = './data/demo.png'):
    # Set the data path to the directory that contains the trained models and test images for inference.
    #_, data_files = common.find_sample_data(
    #                description="Runs a ResNet50 network with a TensorRT inference engine.", 
    #                subfolder="crnn_data", 
    #                find_files=["demo.png", ModelData.MODEL_PATH])
    # Get test images, models and labels.
    test_image = image_path
    onnx_model_file = onnx_path
    #labels = open(labels_file, 'r').read().split('\n')

    # Build a TensorRT engine.
    with build_engine_onnx(onnx_model_file) as engine:
        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:
            # Load a normalized test case into the host input page-locked buffer.
            #test_image = random.choice(test_images)
            test_case = load_normalized_test_case(test_image, inputs[0].host)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            i = 0
            while i < 100:
                i = i + 1
                start = datetime.now()
                trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
                inference_time = datetime.now() - start
                print("the inference time:\t{}".format(inference_time))
            print("the final out put:{}".format(trt_outputs[0]))
            #print("the final shape:{}".format(trt_outputs[0].shape))

def main():
    runOnnxModel('./new_crnn.onnx', './data/demo.png')
    runTensorRTModel('./new_crnn.onnx', './data/demo.png')

if __name__ == '__main__':
    main()
