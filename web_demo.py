import copy
import hashlib
import json
import os
import re

import streamlit as st
from lagent.actions import ActionExecutor
from lagent.llms.lmdepoly_wrapper import LMDeployClient
from lagent.llms.meta_template import INTERNLM2_META as META
from lagent.schema import AgentStatusCode

from utils.actions.fundus_diagnosis import FundusDiagnosis
# from lagent.agents.internlm2_agent import Internlm2Protocol
from utils.internlm2_agent import Internlm2Agent, Internlm2Protocol

# from streamlit.logger import get_logger
LMDEPLOY_IP = '0.0.0.0:23333'
MODEL_NAME = 'internlm2-chat-7b'

OculiChatDA_META_CN = (
    '你是一名眼科专家，可以通过文字和图片来帮助用户诊断眼睛的状态。\n'
    '你有以下三种能力:\n'
    '1. 诊断眼底疾病，包括青光眼、糖尿病视网膜病变、年龄相关性黄斑变性和病理性近视\n'
    '2. 眼科常见疾病诊断，疾病解答，疾病预防等\n'
    '3. 眼科药品信息查询\n'
    '你的工作单位为**某三甲医院**\n'
    '你可以调用外部工具来帮助用户解决问题，如果工具调用显示用户眼睛存在问题，你需要为用户解释该病的病因，早期和晚期的症状，以及可能的治疗方案，同时提醒用户，这仅仅为初步诊断结果，需要用户到医院做进一步的检查'
)
OculiChatDA_META_CN = OculiChatDA_META_CN  # + "\n".join(ReActCALL_PROTOCOL_CN.split("\n")[1:])
PLUGIN_CN = """你可以使用如下工具：
{prompt}
**如果你已经获得足够信息，请直接给出答案. 避免重复或不必要的工具调用!**
如果使用工具请遵循以下格式回复：
```
开始执行工具<|action_start|><|plugin|>
{{
    name: tool_name,
    parameters: tool_parameters in dict format
}}
<|action_end|>
```
其中<|action_start|><|plugin|>必须原样复制，表示开始执行工具
同时注意你可以使用的工具，不要随意捏造！
"""

FUNDUS_DIAGNOSIS_MODEL_PATH = 'glaucoma_cls_dr_grading'


class SessionState:

    def init_state(self):
        """Initialize session state variables."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []

        model_path = os.path.join(FUNDUS_DIAGNOSIS_MODEL_PATH,
                                  'flyer123/GlauClsDRGrading', 'model.onnx')
        if not os.path.exists(model_path):
            from modelscope import snapshot_download
            snapshot_download(
                'flyer123/GlauClsDRGrading',
                cache_dir=FUNDUS_DIAGNOSIS_MODEL_PATH)

        action_list = [
            FundusDiagnosis(model_path=model_path),
        ]
        st.session_state['plugin_map'] = {
            action.name: action
            for action in action_list
        }
        st.session_state['model_map'] = {}
        st.session_state['model_selected'] = None
        st.session_state['plugin_actions'] = set()
        st.session_state['history'] = []

    def clear_state(self):
        """Clear the existing session state."""
        st.session_state['assistant'] = []
        st.session_state['user'] = []
        st.session_state['model_selected'] = None
        st.session_state['file'] = set()
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
        # st.sidebar.title('模型控制')
        st.session_state['file'] = set()
        st.session_state['ip'] = None

    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""

        if MODEL_NAME != st.session_state[
                'model_selected'] or st.session_state['ip'] != LMDEPLOY_IP:
            st.session_state['ip'] = LMDEPLOY_IP
            model = self.init_model(MODEL_NAME, LMDEPLOY_IP)
            self.session_state.clear_state()
            st.session_state['model_selected'] = MODEL_NAME
            if 'chatbot' in st.session_state:
                del st.session_state['chatbot']
        else:
            model = st.session_state['model_map'][MODEL_NAME]

        plugin_action = list(st.session_state['plugin_map'].values())

        if 'chatbot' in st.session_state:
            if len(plugin_action) > 0:
                st.session_state['chatbot']._action_executor = ActionExecutor(
                    actions=plugin_action)
            else:
                st.session_state['chatbot']._action_executor = None
            st.session_state['chatbot']._interpreter_executor = None

        st.sidebar.header('自我揭秘')
        st.sidebar.markdown(
            '我是您的眼科问诊机器人，你可以问我所有的眼科疾病和眼科药品信息。'
            '如果有需要的话，我可以通过识别眼底图来帮助诊断 **青光眼**、 **糖尿病视网膜病变**、**年龄相关性黄斑变性**和**病理性近视** 。'
        )
        if st.sidebar.button('清空对话', key='clear'):
            self.session_state.clear_state()
        uploaded_file = st.sidebar.file_uploader('上传文件')
        st.sidebar.download_button(
            label='下载眼底图测试用例',
            data=open('assets/test_case.zip', 'rb').read(),
            file_name='test_case.zip',
            mime='application/zip')
        return MODEL_NAME, model, plugin_action, uploaded_file, LMDEPLOY_IP

    def init_model(self, model_name, ip=None):
        """Initialize the model based on the input model name."""
        model_url = f'http://{ip}'
        # model_url = model_name
        st.session_state['model_map'][model_name] = LMDeployClient(
            model_name=model_name,
            url=model_url,
            meta_template=META,
            max_new_tokens=1024,
            top_p=0.8,
            top_k=100,
            temperature=0,
            repetition_penalty=1.0,
            stop_words=['<|im_end|>'])
        return st.session_state['model_map'][model_name]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return Internlm2Agent(
            llm=model,
            protocol=Internlm2Protocol(
                meta_prompt=OculiChatDA_META_CN,
                plugin_prompt=PLUGIN_CN,
                tool=dict(
                    begin='{start_token}{name}\n',
                    start_token='<|action_start|>',
                    name_map=dict(
                        plugin='<|plugin|>', interpreter='<|interpreter|>'),
                    belong='assistant',
                    end='<|action_end|>\n',
                ),
            ),
            max_turn=2,
        )

    def render_user(self, prompt: str):
        with st.chat_message('user', avatar='👦'):
            img_paths = re.findall(r'\!\[.*?\]\((.*?)\)', prompt,
                                   re.DOTALL)  # 允许皮配\n等空字符
            if len(img_paths):
                st.markdown(re.sub(r'!\[.*\]\(.*\)', '', prompt))  # 先渲染非图片部分
                # 再渲染图片
                img_path = img_paths[0]
                st.write(
                    f'<img src="app/{img_path}" style="width: 40%;">',
                    unsafe_allow_html=True)
                # if os.path.exists(img_path):
                #     st.image(open(img_path, 'rb').read(),
                #              caption='Uploaded Image', width=400)
            else:
                st.markdown(prompt)

    def render_assistant(self, agent_return):
        with st.chat_message('assistant', avatar='👨‍⚕️'):
            for action in agent_return.actions:
                if (action) and (action.type != 'FinishAction'):
                    self.render_action(action)
            st.markdown(agent_return.response.replace('\\n', ' \\n '))

    def render_plugin_args(self, action):
        action_name = action.type
        args = action.args
        import json
        parameter_dict = dict(name=action_name, parameters=args)
        parameter_str = '```json\n' + json.dumps(
            parameter_dict, indent=4, ensure_ascii=False) + '\n```'
        st.markdown(parameter_str)

    def render_interpreter_args(self, action):
        st.info(action.type)
        st.markdown(action.args['text'])

    def render_action(self, action):
        st.markdown(action.thought)
        if action.type == 'IPythonInterpreter':
            self.render_interpreter_args(action)
        elif action.type == 'FinishAction':
            pass
        else:
            self.render_plugin_args(action)
        self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if (isinstance(action.result, dict)):
            if 'text' in action.result:
                st.markdown('```\n' + action.result['text'] + '\n```')
            if 'image' in action.result:
                # image_path = action.result['image']
                for image_path in action.result['image']:
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
        elif isinstance(action.result, list):
            for item in action.result:
                if item['type'] == 'text':
                    st.markdown('```\n' + item['content'] + '\n```')
                elif item['type'] == 'image':
                    image_data = open(item['content'], 'rb').read()
                    st.image(image_data, caption='Generated Image')
                elif item['type'] == 'video':
                    video_data = open(item['content'], 'rb').read()
                    st.video(video_data)
                elif item['type'] == 'audio':
                    audio_data = open(item['content'], 'rb').read()
                    st.audio(audio_data)
        if action.errmsg:
            st.error(action.errmsg)


def main():
    # logger = get_logger(__name__)
    # Initialize Streamlit UI and setup sidebar
    if 'ui' not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state['ui'] = StreamlitUI(session_state)

    else:
        st.set_page_config(
            layout='wide',
            page_title='眼科问诊大模型',
            page_icon='./assets/page_icon.png')
        st.header(':male-doctor: :blue[OculiChatDA]', divider='rainbow')
    _, model, plugin_action, uploaded_file, _ = st.session_state[
        'ui'].setup_sidebar()

    # Initialize chatbot if it is not already initialized
    # or if the model has changed
    if 'chatbot' not in st.session_state or model != st.session_state[
            'chatbot']._llm:
        st.session_state['chatbot'] = st.session_state[
            'ui'].initialize_chatbot(model, plugin_action)
        st.session_state['session_history'] = []

    for prompt, agent_return in zip(st.session_state['user'],
                                    st.session_state['assistant']):
        st.session_state['ui'].render_user(prompt)
        st.session_state['ui'].render_assistant(agent_return)

    if user_input := st.chat_input(''):
        with st.container():
            st.session_state['ui'].render_user(user_input)
        st.session_state['user'].append(user_input)
        # Add file uploader to sidebar
        if (uploaded_file
                and uploaded_file.name not in st.session_state['file']):

            st.session_state['file'].add(uploaded_file.name)
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            if 'image' in file_type:
                st.image(file_bytes, caption='Uploaded Image', width=600)
            elif 'video' in file_type:
                st.video(file_bytes, caption='Uploaded Video')
            elif 'audio' in file_type:
                st.audio(file_bytes, caption='Uploaded Audio')
            # Save the file to a temporary location and get the path

            postfix = uploaded_file.name.split('.')[-1]
            # prefix = str(uuid.uuid4())
            prefix = hashlib.md5(file_bytes).hexdigest()
            filename = f'{prefix}.{postfix}'
            file_path = os.path.join(root_dir, filename)
            with open(file_path, 'wb') as tmpfile:
                tmpfile.write(file_bytes)
            file_size = os.stat(file_path).st_size / 1024 / 1024
            file_size = f'{round(file_size, 2)} MB'
            # st.write(f'File saved at: {file_path}')
            user_input = [
                dict(role='user', content=user_input),
                dict(
                    role='user',
                    content=json.dumps(dict(path=file_path, size=file_size)),
                    name='眼底图')
            ]
            st.session_state['user'][-1] = st.session_state['user'][
                -1] + f'\n ![眼底图图像路径]({file_path})'
        if isinstance(user_input, str):
            user_input = [dict(role='user', content=user_input)]
        st.session_state['last_status'] = AgentStatusCode.SESSION_READY
        for agent_return in st.session_state['chatbot'].stream_chat(
                st.session_state['session_history'] + user_input):
            if agent_return.state == AgentStatusCode.PLUGIN_RETURN:
                with st.container():
                    st.session_state['ui'].render_plugin_args(
                        agent_return.actions[-1])
                    st.session_state['ui'].render_action_results(
                        agent_return.actions[-1])
            elif agent_return.state == AgentStatusCode.CODE_RETURN:
                with st.container():
                    st.session_state['ui'].render_action_results(
                        agent_return.actions[-1])
            elif (agent_return.state == AgentStatusCode.STREAM_ING
                  or agent_return.state == AgentStatusCode.CODING):
                # st.markdown(agent_return.response)
                # 清除占位符的当前内容，并显示新内容
                with st.container():
                    if agent_return.state != st.session_state['last_status']:
                        st.session_state['temp'] = ''
                        placeholder = st.empty()
                        st.session_state['placeholder'] = placeholder
                    if isinstance(agent_return.response, dict):
                        action = f"\n\n {agent_return.response['name']}: \n\n"
                        action_input = agent_return.response['parameters']
                        if agent_return.response[
                                'name'] == 'IPythonInterpreter':
                            action_input = action_input['command']
                        response = action + action_input
                    else:
                        response = agent_return.response
                    st.session_state['temp'] = response
                    st.session_state['placeholder'].markdown(
                        st.session_state['temp'])
            elif agent_return.state == AgentStatusCode.END:
                st.session_state['session_history'] += (
                    user_input + agent_return.inner_steps)
                agent_return = copy.deepcopy(agent_return)
                agent_return.response = st.session_state['temp']
                st.session_state['assistant'].append(
                    copy.deepcopy(agent_return))
            st.session_state['last_status'] = agent_return.state


if __name__ == '__main__':
    root_dir = 'static'
    os.makedirs(root_dir, exist_ok=True)
    main()
