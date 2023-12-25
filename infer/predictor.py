from calendar import c
from dis import dis
from os.path import abspath, dirname
from typing import IO, Dict

import numpy as np
import torch
import yaml

from train.config.model_config import network_cfg
import tensorrt as trt
import pycuda.driver as pdd
import pycuda.autoinit

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

class DenosingConfig:

    def __init__(self, test_cfg):
        # 配置文件
        self.patch_size = test_cfg.get('patch_size')
        self.stride = test_cfg.get('stride')

    def __repr__(self) -> str:
        return str(self.__dict__)


class DenosingModel:

    def __init__(self, model_f: IO, config_f):
        # TODO: 模型文件定制
        self.model_f = model_f 
        self.config_f = config_f
        self.network_cfg = network_cfg


class DenosingPredictor:

    def __init__(self, device: str, model: DenosingModel):
        self.device = torch.device(device)
        self.model = model
        self.tensorrt_flag = False 

        with open(self.model.config_f, 'r') as config_f:
            self.test_cfg = DenosingConfig(yaml.safe_load(config_f))
        self.network_cfg = model.network_cfg
        self.load_model()

    def load_model(self) -> None:
        if isinstance(self.model.model_f, str):
            # 根据后缀判断类型
            if self.model.model_f.endswith('.pth'):
                self.load_model_pth()
            elif self.model.model_f.endswith('.pt'):
                self.load_model_jit()
            elif self.model.model_f.endswith('.engine'):
                self.tensorrt_flag = True
                self.load_model_engine()

    def load_model_jit(self) -> None:
        # 加载静态图
        from torch import jit
        self.net = jit.load(self.model.model_f, map_location=self.device)
        self.net.eval()
        self.net.to(self.device)

    def load_model_pth(self) -> None:
        # 加载动态图
        self.net = self.network_cfg.network
        checkpoint = torch.load(self.model.model_f, map_location=self.device)
        self.net.load_state_dict(checkpoint)
        self.net.eval()
        self.net.to(self.device)

    def load_model_engine(self) -> None:
        TRT_LOGGER = trt.Logger()
        runtime = trt.Runtime(TRT_LOGGER)
        with open(self.model.model_f, 'rb') as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_buffers(self, engine, context):
        inputs = []
        outputs = []
        bindings = []
        stream = pdd.Stream()
        for i, binding in enumerate(engine):
            size = trt.volume(context.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = pdd.pagelocked_empty(size, dtype)
            device_mem = pdd.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_inference(self, context, bindings, inputs, outputs, stream, batch_size):
        # Transfer input data to the GPU.
        [pdd.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [pdd.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def predict(self, input):
        patch_size = self.test_cfg.patch_size
        stride = self.test_cfg.stride
        _shape = input.shape
        output = np.zeros(_shape)
        counter = np.zeros(_shape)
        # 将图片切块进行测试
        start_index_x = 0
        while start_index_x < _shape[0]:
            start_index_y = 0
            while start_index_y < _shape[1]:
                temp_patch = np.zeros(patch_size)
                end_index_x = start_index_x + patch_size[0]
                end_index_y = start_index_y + patch_size[1]
                input_patch = input[start_index_x:end_index_x, start_index_y:end_index_y]
                temp_patch[:input_patch.shape[0],:input_patch.shape[1]] = input_patch 
                counter[start_index_x:end_index_x, start_index_y:end_index_y] += 1  
                pred = self._forward(temp_patch)
                temp_size = output[start_index_x:end_index_x, start_index_y:end_index_y].shape
                output[start_index_x:end_index_x, start_index_y:end_index_y] += pred[:temp_size[0],:temp_size[1]]
                start_index_y += stride[1]
            start_index_x += stride[0]
        # import matplotlib.pyplot as plt
        # plt.imshow(counter)
        # plt.show()
        output = output / counter
        return output

    def _forward(self, img):
        img_t = torch.from_numpy(img)[None,None].to(self.device)
        # pytorch预测
        with torch.no_grad():
            output = self.net(img_t.float())
            output = output.squeeze().cpu().detach().numpy()
        return output
