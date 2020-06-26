
from itertools import chain
import argparse
import os

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt

try:
    # Sometimes python2 does not understand FileNotFoundError
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30


def add_help(description):
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    args, _ = parser.parse_known_args()


def find_sample_data(description="Runs a TensorRT Python sample", subfolder="", find_files=[]):
    '''
    Parses sample arguments.

    Args:
        description (str): Description of the sample.
        subfolder (str): The subfolder containing data relevant to this sample
        find_files (str): A list of filenames to find. Each filename will be replaced with an absolute path.

    Returns:
        str: Path of data directory.
    '''

    # Standard command-line arguments for all samples.
    kDEFAULT_DATA_ROOT = os.path.join(os.sep, "mnt", "crnn_yychai", "debug", "version1_6_0", "onnx2trt" ,"data")
    print("the default path:{}".format(kDEFAULT_DATA_ROOT))
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--datadir", help="Location of the TensorRT sample data directory, and any additional data directories.", action="append", default=[kDEFAULT_DATA_ROOT])
    args, _ = parser.parse_known_args()

    def get_data_path(data_dir):
        # If the subfolder exists, append it to the path, otherwise use the provided path as-is.
        data_path = os.path.join(data_dir, subfolder)
        if not os.path.exists(data_path):
            print("WARNING: " + data_path + " does not exist. Trying " + data_dir + " instead.")
            data_path = data_dir
        # Make sure data directory exists.
        if not (os.path.exists(data_path)):
            print("WARNING: {:} does not exist. Please provide the correct data path with the -d option.".format(data_path))
        return data_path
    data_paths = [get_data_path(data_dir) for data_dir in args.datadir]
    print("bug1\t", data_paths)
    return data_paths, locate_files(data_paths, find_files)

def locate_files(data_paths, filenames):
    """
    Locates the specified files in the specified data directories.
    If a file exists in multiple data directories, the first directory is used.

    Args:
        data_paths (List[str]): The data directories.
        filename (List[str]): The names of the files to find.

    Returns:
        List[str]: The absolute paths of the files.

    Raises:
        FileNotFoundError if a file could not be located.
    """
    found_files = [None] * len(filenames)
    for data_path in data_paths:
        # Find all requested files.
        for index, (found, filename) in enumerate(zip(found_files, filenames)):
            if not found:
                file_path = os.path.abspath(os.path.join(data_path, filename))
                if os.path.exists(file_path):
                    found_files[index] = file_path

    # Check that all files were found
    for f, filename in zip(found_files, filenames):
        if not f or not os.path.exists(f):
            raise FileNotFoundError("Could not find {:}. Searched in data paths: {:}".format(filename, data_paths))
    return found_files

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    #print('')
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@allocate_buffers(engine):....")
    for binding in engine:
        #print("\tbinding: {}, {}".format(binding, engine.get_binding_shape(binding)))

        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        #print('\tsize: ', size)
        #print('\tengine.max_batch_size: ', engine.max_batch_size)

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        #print('\tdtype: ', dtype)

        #print('\t# Allocate host and device buffers.........')
        if engine.binding_is_input(binding):
            host_mem = cuda.pagelocked_empty((1 ,1 , 32, 100), dtype)
        else:
            host_mem = cuda.pagelocked_empty((26, 1, 37), dtype)
        #print('\tAllocate host buffer: host_mem -> {}, {}'.format(host_mem, host_mem.nbytes))

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        #print('\tAllocate device buffer: device_mem -> {}, {}'.format(device_mem, int(device_mem)))

        #print('\t# Append the device buffer to device bindings.......')
        bindings.append(int(device_mem))
        #print('\tbindings: ', bindings)

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            #print("this is the input!")
            #print('____HostDeviceMem(host_mem, device_mem)): {}, {}'.format(HostDeviceMem(host_mem, device_mem),type(HostDeviceMem(host_mem, device_mem))))
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            #print("This is the output!")
            outputs.append(HostDeviceMem(host_mem, device_mem))
        #print('inputs: ', inputs)
        #print('outputs: ', outputs)
        #print('bindings: ', bindings)
        #print('stream: ', stream)
        #print("----------------------end allocating one binding in the onnx model-------------------------")

    return inputs, outputs, bindings, stream
"""
# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers_cyy(engine):
    inputs_host = []
    inputs_device= []
    outputs_host = []
    outputs_device = []

    bindings = []
    stream = cuda.Stream()
    print('')
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@allocate_buffers(engine):....")
    for binding in engine:
        print("\tbinding: {}, {}".format(binding, engine.get_binding_shape(binding)))

        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        print('\t\tsize: ', size)
        print('\t\tengine.max_batch_size: ', engine.max_batch_size)

        dtype = trt.nptype(engine.get_binding_dtype(binding))
        print('\t\tdtype: ', dtype)

        print('\t# Allocate host and device buffers.........')
        host_mem = cuda.pagelocked_empty(size, dtype)
        print('\t\tAllocate host buffer: host_mem -> {}, {}'.format(host_mem, host_mem.nbytes))

        device_mem = cuda.mem_alloc(host_mem.nbytes)
        print('\t\tAllocate device buffer: device_mem -> {}, {}'.format(device_mem, int(device_mem)))

        print('\t# Append the device buffer to device bindings.......')
        bindings.append(int(device_mem))
        print('\t\tbindings: ', bindings)

        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            print('______________________________________________HostDeviceMem(host_mem, device_mem)): {}, {}'.format(HostDeviceMem(host_mem, device_mem),type(HostDeviceMem(host_mem, device_mem))))
            inputs_host.append(host_mem)
            inputs_device.append(device_mem)
        else:
            outputs_host.append(host_mem)
            outputs_device.append(device_mem)

    return inputs_host, inputs_device, outputs_host, outputs_device, bindings, stream
"""
# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
"""
def do_inference_cyy(context, bindings, inputs_host, inputs_device, outputs_host, outputs_device, stream, batch_size=1):
    print('')
    print('@_@ do_inference(context, bindings, inputs, outputs, stream, batch_size=1):')

    # Transfer input data to the GPU.
    for k in range(len(inputs_host)):
        host_mem = inputs_host[k]
        host_device = inputs_device[k]
        cuda.memcpy_htod_async(host_device, host_mem, stream)

    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    for k in range(len(outputs_host)):
        host_mem = outputs_host[k]
        host_device = outputs_device[k]
        cuda.memcpy_dtoh_async(host_device, host_mem, stream)

    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
"""
# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    #print('')
    #print('@_@ do_inference(context, bindings, inputs, outputs, stream, batch_size=1):')
    #print('\tinputs: {}'.format(len(inputs)))
    #for inp in inputs:
    #    print("\tinput: {}".format(inp))
    #    print('\t\t____device: {}, {}'.format(inp.device, type(inp.device)))
    #    print('\t\t____host: {}, {}'.format(inp.host, type(inp.host)))
    #    print('\t\t____stream: {}, {}'.format(stream, type(stream)))

    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]

"""
# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]

    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
"""
