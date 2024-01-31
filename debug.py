from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.agents import ReAct, ReActProtocol, CALL_PROTOCOL_CN
from lagent.llms import HFTransformer
from lagent.llms.meta_template import INTERNLM2_META as META
import os
from modelscope import snapshot_download
from utils.actions.fundus_diagnosis import FundusDiagnosis

llm = HFTransformer(path='/share/model_repos/internlm2-chat-7b', meta_template=META)

cache_dir = "glaucoma_cls_dr_grading"
model_path = os.path.join(cache_dir, "flyer123/GlauClsDRGrading", "model.onnx")
if not os.path.exists(model_path):
    snapshot_download("flyer123/GlauClsDRGrading", cache_dir=cache_dir)


chatbot = ReAct(
    llm=llm,  # Provide the Language Model instance.
    action_executor=ActionExecutor(actions=[FundusDiagnosis(model_path=model_path)]),
    protocol=ReActProtocol(
        call_protocol=CALL_PROTOCOL_CN)
)

res = chatbot.chat(
    '我上传了一张图片，图片地址为/root/GlauClsDRGrading/data/refuge/images/g0001.jpg, 请看看我是否患有青光眼'
)

print(res.response)