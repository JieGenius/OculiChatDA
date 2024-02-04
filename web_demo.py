import copy
import os
import re

import streamlit as st
from streamlit.logger import get_logger

from lagent.actions import ActionExecutor
from lagent.agents.react import ReAct, ReActProtocol
from lagent.llms.huggingface import HFTransformerCasualLM
from utils.actions.fundus_diagnosis import FundusDiagnosis
from lagent.llms.meta_template import INTERNLM2_META as META
from utils.agent import MyReAct


# MODEL_DIR = "/share/model_repos/internlm2-chat-7b-4bits"
MODEL_DIR = "./OpenLMLab/InternLM-chat-7b"
CALL_PROTOCOL_CN = """你是一名眼科专家，可以通过文字和图片来帮助用户诊断眼睛的状态。（请不要在回复中透露你的个人信息和工作单位)。
你可以调用外部工具来帮助你解决问题。
可以使用的工具包括：
{tool_description}
如果使用工具请遵循以下格式回复：
```
{thought}思考你当前步骤需要解决什么问题，是否需要使用工具
{action}工具名称，你的工具必须从 [{action_names}] 选择
{action_input}工具输入参数
```
工具返回按照以下格式回复：
```
{response}调用工具后的结果
```
如果你已经知道了答案，或者你不需要工具，请遵循以下格式回复
```
{thought}给出最终答案的思考过程
{finish}最终答案
```
开始!"""
class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []


        cache_dir = "glaucoma_cls_dr_grading"
        model_path = os.path.join(cache_dir, "flyer123/GlauClsDRGrading", "model.onnx")
        if not os.path.exists(model_path):
            from modelscope import snapshot_download
            snapshot_download("flyer123/GlauClsDRGrading", cache_dir=cache_dir)

        action_list = [FundusDiagnosis(model_path=model_path)]
        st.session_state['plugin_map'] = {
            action.name: action
            for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()
        st.session_state["turn"] = 0 # 记录当前会话的轮次，第一轮需要添加system


    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state["turn"] = 0
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.set_page_config(
            layout='wide',
            page_title='眼科问诊大模型',
            page_icon='./assets/page_icon.png')
        st.header(':male-doctor: :blue[OculiChatDA]', divider='rainbow')
        st.sidebar.title('')

    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        model_name = "internlm2"
        if model_name != st.session_state['model_selected']:
            model = self.init_model(model_name)
            self.session_state.clear_state()
            st.session_state['model_selected'] = model_name
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']
        else:
            model = st.session_state['model_map'][model_name]

        plugin_name = list(st.session_state['plugin_map'].keys())

        plugin_action = [
            st.session_state['plugin_map'][name] for name in plugin_name
        ]
        if 'chatbot' in st.session_state:
            st.session_state['chatbot']._action_executor = ActionExecutor(
                actions=plugin_action)

        st.sidebar.header("自我揭秘")
        st.sidebar.markdown("你好！我是您的眼科问诊机器人，专业且贴心。我知道广泛的眼科知识，可以帮助您了解和诊断各种眼科疾病。")
        st.sidebar.markdown("另外，我还具备**识别眼底图**的能力，这对于判断一些重要眼科疾病非常重要。通过分析眼底图，我能够帮助您了解是否存在青光眼或糖尿病视网膜病变等情况。")
        st.sidebar.markdown("请随时向我提问，我将尽力为您提供专业的眼科建议和信息。您的眼健康，是我的首要关注点！")
        # st.sidebar.write("---")
        if st.sidebar.button('清空对话', key='clear', use_container_width=True):
            self.session_state.clear_state()
        if "file_upload_key" not in st.session_state:
            st.session_state.file_upload_key = 0
        uploaded_file = st.sidebar.file_uploader(
            '眼底图文件', type=['png', 'jpg', 'jpeg'], key=st.session_state.file_upload_key)
        return model_name, model, plugin_action, uploaded_file

    def init_model(self, option):
        """Initialize the model based on the selected option."""
        if option not in st.session_state['model_map']:
            st.session_state['model_map'][option] = self.load_internlm2()
        return st.session_state['model_map'][option]

    @staticmethod
    @st.cache_resource
    def load_internlm2():
        return HFTransformerCasualLM(
        MODEL_DIR, meta_template=META)

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return MyReAct(
            llm=model, action_executor=ActionExecutor(actions=plugin_action), protocol=ReActProtocol(call_protocol=CALL_PROTOCOL_CN))

    def render_user(self, prompt: str):
        with st.chat_message('user', avatar="👦"):
            st.markdown('''<style>
             .stChatMessage img {
                 width: 60%; 
                 display: block;
             } 
             </style>''', unsafe_allow_html=True)

            st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message('assistant', avatar="👨‍⚕️"): # 医生的avatar
            for action in agent_return.actions:
                if (action) and action.type == "FundusDiagnosis":
                    self.render_action(action)
            st.markdown(agent_return.response)

    def render_action(self, action):
        with st.expander(action.type, expanded=True):
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>插    件</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.type + '</span></p>',
                unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>思考步骤</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.thought + '</span></p>',
                unsafe_allow_html=True)
            if (isinstance(action.args, dict) and 'text' in action.args):
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行内容</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",
                    # noqa E501
                    unsafe_allow_html=True)
                st.markdown(action.args['text'])
            self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            st.markdown(
                "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行结果</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",
                # noqa E501
                unsafe_allow_html=True)
            if 'text' in action.result:
                st.markdown(
                    "<p style='text-align: left;'>" + action.result['text'] +
                    '</p>',
                    unsafe_allow_html=True)
            if 'image' in action.result:
                image_path = action.result['image']
                image_data = open(image_path, 'rb').read()
                st.image(image_data, caption='Generated Image')
            if 'video' in action.result:
                video_data = action.result['video']
                video_data = open(video_data, 'rb').read()
                st.video(video_data)
            if 'audio' in action.result:
                audio_data = action.result['audio']
                audio_data = open(audio_data, 'rb').read()
                st.audio(audio_data)


def main():
    logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)
        # st.session_state.ui = StreamlitUI(session_state)
    else:
        st.set_page_config(
            layout='wide',
            page_title='眼科问诊大模型',
            page_icon='./assets/page_icon.png')
        st.header(':male-doctor: :blue[OculiChatDA]', divider='rainbow')
    model_name, model, plugin_action, uploaded_file = st.session_state[
        'ui'].setup_sidebar()

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    if 'chatbot' not in st.session_state or model != st.session_state[
        'chatbot']._llm:
        st.session_state['chatbot'] = st.session_state[
            'ui'].initialize_chatbot(model, plugin_action)

    for prompt, agent_return in zip(st.session_state['user'],
                                    st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)
    # User input form at the bottom (this part will be at the bottom)
    # with st.form(key='my_form', clear_on_submit=True):

    if user_input := st.chat_input(''):
        # Add file uploader to sidebar
        if uploaded_file:
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type

            # Save the file to a temporary location and get the path
            if not os.path.exists("static"):
                os.makedirs("static")
            file_path = os.path.join("static", uploaded_file.name)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            print(f'File saved at: {file_path}')
            user_input_with_image_info = '我上传了一个图像，路径为: {file_path}. {user_input}'.format(
                file_path=file_path, user_input=user_input)
            user_input_render = "{} \n![{}]({})".format(user_input, "眼底图", "app/" + file_path)
            st.session_state.file_upload_key += 1 # 用于清除已经选择的文件
        else:
            user_input_with_image_info = user_input
            user_input_render = user_input

        st.session_state['ui'].render_user(user_input_render)
        st.session_state['user'].append(user_input_render)

        agent_return = st.session_state['chatbot'].chat(user_input_with_image_info)
        st.session_state['assistant'].append(copy.deepcopy(agent_return))
        logger.info("agent_return:",agent_return.inner_steps)
        st.session_state['ui'].render_assistant(agent_return)
        st.session_state["turn"] += 1


if __name__ == '__main__':

    root_dir = "tmp_dir"
    os.makedirs(root_dir, exist_ok=True)

    if not os.path.exists(MODEL_DIR):
        from openxlab.model import download

        download(model_repo='OpenLMLab/internlm2-chat-7b', output=MODEL_DIR)

        print("解压后目录结果如下：")
        print(os.listdir(MODEL_DIR))

    main()
