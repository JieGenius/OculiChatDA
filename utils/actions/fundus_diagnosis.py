import os

import cv2
import numpy as np
import onnxruntime as ort
from lagent.actions.base_action import BaseAction, tool_api
from streamlit.logger import get_logger

from utils.transform import center_crop, resized_edge

logger = get_logger(__name__)


class FundusDiagnosis(BaseAction):

    def __init__(self, model_path=None, enable: bool = True) -> None:
        super().__init__(description=None, enable=enable)

        if model_path is not None:
            assert os.path.exists(
                model_path), f'model_path: {model_path} not exists'
            assert model_path[-5:] == '.onnx', \
                f'model_path: {model_path} is not a onnx model'
            self.model_path = model_path
            providers = ['CUDAExecutionProvider']

            self.model = ort.InferenceSession(
                model_path,
                providers=providers,
            )

    @tool_api(explode_return=True)
    def fundus_diagnosis(self, fundus_path: str) -> dict:
        """运行眼底疾病诊断，可实现青光眼二分类和糖尿病视网膜病变5分级.

        Args:
            fundus_path (str): 眼底图像的路径

        Returns:
            :class:`dict`: 诊断结果
              * msg (str): 工具调用是否成功的说明
              * glaucoma (int): 1代表可疑青光眼，0代表非青光眼
              * dr_level (int): 糖尿病视网膜病变的等级，0代表健康，4代表患病程度最严重
                                0 为非糖尿病视网膜病变
                                1 代表轻度非增生性的糖尿病视网膜病变
                                2 中度
                                3 重度
                                4 增生性糖尿病视网膜病变
              * amd (int): 1代表可疑年龄相关性黄斑变性，0代表非黄斑变性
              * pm (int): 1代表病理性近视，0代表非病理性近视
        """
        image_path = fundus_path
        logger.info('查询是: ' + fundus_path)
        if not os.path.exists(image_path):
            return {'msg': '由于图片路径错误，该工具运行失败'}
        img = cv2.imread(image_path)

        img = resized_edge(img, 448, edge='long')
        img = center_crop(img, 448)
        mean = [0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
        std = [0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
        img = (img - mean) / std
        img = img[..., ::-1]  # bgr to rgb
        img = img.transpose((2, 0, 1))
        img = img.astype('float32')
        img = img[np.newaxis, ...]

        output = self.model.run(None, {'input': img})

        glaucoma = output[0][0].argmax()
        dr = output[1][0].argmax()
        amd = output[2][0].argmax()
        pm = output[3][0].argmax()
        return dict(
            glaucoma=int(glaucoma),
            dr_level=int(dr),
            amd=int(amd),
            pm=int(pm),
            msg='运行成功')
