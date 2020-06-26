from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn

import os
import copy
import numpy as np

import onnx, onnxruntime
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

#onnx_file_path = 'crnn0622.onnx'

model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
#print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path)) #load the pytorch model

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
image = image.view(1, *image.size())
onnx_inputs = copy.deepcopy(image)
#print("onnx model's input shape:\t{}".format(onnx_inputs.shape))
if torch.cuda.is_available():
    image = image.cuda()
image = Variable(image)

onnx_file_path = './new_crnn.onnx'

model.eval()

model_cnn = model.cnn
#
rnn_input = torch.ones(26, 1, 512).cuda()
rnn_input = Variable(rnn_input)

# onnx
if not os.path.exists(onnx_file_path):
    print('\t>>write onnx: {}'.format(onnx_file_path))
    torch_out = torch.onnx.export(
            model,
            image,
            onnx_file_path,
            verbose=True,
            export_params=True)
torch_cnn = torch.onnx.export(model.cnn, image, "./new_crnn_cnn.onnx")
torch_rnn = torch.onnx.export(model.rnn, rnn_input, "./new_crnn_rnn.onnx")


if 1:
    session = onnxruntime.InferenceSession("./onnx2trt/data/crnn_data/new_crnn.onnx")
    input_name = session.get_inputs()[0].name
    print('\t>>input: {}, {}, {}'.format(session.get_inputs()[0].name, session.get_inputs()[0].shape, session.get_inputs()[0].type))
    _outputs = session.get_outputs()
    for kk in range(len(_outputs)):
        _out = _outputs[kk]
        print('\t>>out-{}: {}, {}, {}'.format(kk, _out.name, _out.shape, _out.type))

    x = np.array(onnx_inputs).astype(np.float32)

    p = session.run(None, {input_name: x})
    out1 = p[0]
    print('============================================================================')
    print('>>summary:')
    print('onnx out: {} \n{}'.format(np.shape(out1), out1))

    #print("x's type:", type(x))


#Forward if pytorch
out0 = model(image)
print('pytorch out: {} \n{}'.format(np.shape(out0), out0))

# test the first cnn part
"""
if 1:
    session = onnxruntime.InferenceSession("new_crnn_rnn.onnx")
    input_name = session.get_inputs()[0].name
    #print('\t>>input: {}, {}, {}'.format(session.get_inputs()[0].name, session.get_inputs()[0].shape, session.get_inputs()[0].type))
    _outputs = session.get_outputs()
    for kk in range(len(_outputs)):
        _out = _outputs[kk]
        #print('\t>>out-{}: {}, {}, {}'.format(kk, _out.name, _out.shape, _out.type))

    #x = np.array(onnx_inputs).astype(np.float32)
    x = np.ones((26, 1, 512)).astype(np.float32)
    

    p = session.run(None, {input_name: x})
    out1 = p[0]
    print('============================================================================')
    print('>>summary:')
    print('onnx out: {} \n{}'.format(np.shape(out1), out1))

"""
