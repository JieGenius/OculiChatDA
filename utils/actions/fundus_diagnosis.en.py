import os
import numpy as np
import cv2
import onnxruntime as ort
from utils.transform import resized_edge, center_crop
from streamlit.logger import get_logger
from lagent.actions.base_action import BaseAction, tool_api

logger = get_logger(__name__)


class FundusDiagnosis(BaseAction):
    def __init__(self,
                 model_path=None,
                 description: str = None,
                 enable: bool = True) -> None:
        super().__init__(description=None, enable=enable)

        if model_path is not None:
            assert os.path.exists(model_path), f"model_path: {model_path} not exists"
            assert model_path[-5:] == ".onnx", f"model_path: {model_path} is not a onnx model"
            self.model_path = model_path
            providers = ['CUDAExecutionProvider']

            self.model = ort.InferenceSession(model_path, providers=providers, )

    @tool_api(explode_return=True)
    def fundus_diagnosis(self, fundus_path: str) -> dict:
        """Run fundus disease diagnostics and return diagnostic results.
        Diagnosable diseases include diabetic retinopathy and glaucoma.
        This tool is only available after the user has uploaded an image

        Args:
            fundus_path (str): the path of fundus

        Returns:
            :class:`dict`: the diagnostic results
              * msg (str): An illustration about state of tool process
              * glaucoma (int): 0 denotes a non-glaucoma patient, 1 denotes a suspected glaucoma
              * dr_level (int): the diabetic retinopathy level.
                                0 is healthy
                                1 denotes Mild Nonproliferative Retinopathy
                                2 denotes Moderate Nonproliferative Retinopathy
                                3 denotes Severe Nonproliferative Retinopathy
                                4 denotes Proliferative Retinopathy

        """
        image_path = fundus_path
        logger.info("查询是: " + fundus_path)
        if not os.path.exists(image_path):
            return {
                "msg": "由于图片路径错误，该工具运行失败"
            }
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
        return dict(glaucoma=glaucoma, dr_level=dr, msg="运行成功")
