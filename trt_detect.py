import  sys
import os
import time
import numpy as np

import argparse
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from tool.utils import *

TRT_LOGGER = trt.Logger()

def get_engine(engine_file):
    # serialized 된 engine 파일을 입력받아 enigne을 빌드
    print("reading engine file {}".format(engine_file))
    # build할 때와 마찬가지로 runtime에도 logger 인스턴스가 필요함
    with open(engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


# 그냥 단순한 데이터 저장 클래스
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffer(engine, batch):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # 불러온 TensorRT의 engine 모듈을 분해하여 stream을 묶고, 각 텐서들을 GPU로 할당함
    for binding in engine:
        size = trt.volume(engine.get_bindin_shape(binding)) * batch
        dim =  engine.get_binding_shape(binding)
        # # dynamic 배치를 위한 작업
        if dim[0] <0:
            size*=-1

        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # host (CPU) 및 device (GPU) 버퍼들을 할당
        host_mem = cuda.pagelocked_empty(size,dtype) # size만큼의 dtype 데이터를위한 메모리 초기화
        device_mem = cuda.mem_alloc(host_mem.nbytes) # host에 할당한 만큼의 nbytes를 할당

        # device binding들에 device buffer를 추가
        bindings.append(int(device_mem))

        #각각 적절한 리스트에 append
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem,device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem,device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream):
    # input 을 GPU로  (target, source, stream)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs ]
    # inference 수행
    context.execute_async(bindings=bindings, stream_handle=stream.handle)
    # GPU로 부터 추론결과 다시 받기
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchornize the stream
    stream.synchronize()
    # 최종적으로 host outputs (GPU로부터 받은 최종 추론 결과)를 반환
    return [out.host for out in outputs]


def detect(context, buffers, img, img_size, num_classes):
    h,w = img_size
    # input 전처리 -> 학습할 때 수행한 전처리 참고하여 알맞게 적용하기
    resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    img_in = np.ascontiguousarray(img_in)
    print("Shape of the network input: ", img_in.shape)

    inputs, outputs, bindings, stream = buffers
    # batch가 1이기 때문에 인풋 길이는 1
    inputs[0].host = img_in # cpu에 이미지 로드

    trt_outputs = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    # output은 2개 ->bbox, classification
    trt_outputs[0] = trt_outputs[0].reshape(1, -1, 1, 4)  # 1개의 배치에 대해서 몇개의 box가 나왔는지모름 + objectness socre + 4개좌표
    trt_outputs[1] = trt_outputs[1].reshape(1, -1, num_classes)  #

    boxes = post_processing(img_in, 0.4, 0.6, trt_outputs)

    return boxes


def main(engine_path, img_path, img_size):
    with get_engine(engine_path) as engine, engine.create_execution_context() as context:
        '''
        추론을  수행하려면 텐서를 통과해야하는데,
        (Tensor)RT에서는 입력 및 출력에 대해서 GPU 포인터 목록에 명시해야 한다. 
        입력 및 출력 텐서에 제공한 이름을 사용하여 엔진을 쿼리함으로써 어레이에서 올바른 위치를 찾는다
        '''
        # 입출력 텐서에 대한 버퍼 할당
        buffers = allocate_buffer(engine, 1) # 엔진과 batch 사이즈
        img_h, img_w = img_size
        context.set_binding_shape(0, (1,3,img_h,img_w))

        # 여기서 부터 전처리
        img = cv2.imread(img_path)
        num_classes = 80
        boxes = detect(context, buffers, img, img_size, num_classes)  # get detected_boxes

        names_file = 'data/coco.names'
        class_names = load_class_names(names_file)
        plot_boxes_cv2(img , boxes[0], savename='predictions_trt.jpg', class_names=class_names)



