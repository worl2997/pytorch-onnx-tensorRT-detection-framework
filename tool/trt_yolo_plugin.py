import ctypes
import sys
import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
from tool.utils import *
import pycuda.autoinit

# Simple helper data class that's a little nicer to use than a 2-tuple.
def GiB(val):
    return val * 1 << 30


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host

def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class Trt_yolo(object):

    def __init__(self, engine_path, num_classes, img_size):
        self.engine = engine_path
        self.num_classes = num_classes
        #self.letter_box = letter_box
        self.IN_IMAGE_H,self.IN_IMAGE_W = img_size
        # 추후에 multi-batch를 지원하려면 나중에 확장하기
        self.inference_fn = do_inference
        self.trt_logger = trt.Logger()
        self.engine = self.get_engine()

        #self.input_shape = get_input_shape(self.engine)
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, 1)
            self.context.set_binding_shape(0, (1, 3, self.IN_IMAGE_W, self.IN_IMAGE_W))
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        # 쿠다 메모리 free
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, image_src, conf_thresh=0.4, nms_thresh=0.6):
        resized = cv2.resize(image_src, (self.IN_IMAGE_W, self.IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        self.inputs[0].host = img_in # 네트워크 입력에 맞게 resize된 입력

        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)

        #print('Len of outputs: ', len(trt_outputs))
        trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)
        trt_outputs[1] = trt_outputs[1].reshape(1, -1, self.num_classes)


        #뭔가 잘못되었음
        boxes = post_processing(conf_thresh, nms_thresh, trt_outputs)
        # nms를 거친 boxes, scores, classes를 반환해야함
        return boxes

    def get_engine(self):
        print("Reading engine from file {}".format(self.engine))
        with open(self.engine, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
