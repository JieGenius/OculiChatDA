from lagent.agents import ReAct
from lagent import AgentReturn, ActionReturn
import copy
from transformers import GenerationConfig
import time
from streamlit.logger import get_logger

logger = get_logger(__name__)
class MyReAct(ReAct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def chat(self, message: str) -> AgentReturn:
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()
        default_response = 'Sorry that I cannot answer your question.'
        start = time.time()
        gen_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            repetition_penalty=1.002,
        )
        logger.info("开始对话，message:" + message)
        for turn in range(self.max_turn):
            for_start = time.time()
            prompt = self._protocol.format(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                force_stop=(turn == self.max_turn - 1))
            s1 = time.time()
            response = self._llm.generate_from_template(prompt, 512, generation_config=gen_config)
            logger.info(f"response生成用时：{time.time() - s1}秒")
            self._inner_history.append(
                dict(role='assistant', content=response))
            thought, action, action_input = self._protocol.parse(
                response, self._action_executor)
            s2 = time.time()

            action_return: ActionReturn = self._action_executor(
                action, action_input)
            logger.info(f"exectue action用时：{time.time() - s2}秒")
            if action_return.type == "NoAction":
                # 没有获取到action的情况
                action_return.thought = "该回答不需要调用任何Action"
                agent_return.response = response
                logger.info("模型输出异常，未按照指定模板生成Action，直接返回原始输出")
                break
            action_return.thought = thought
            agent_return.actions.append(action_return)
            if action_return.type == self._action_executor.finish_action.name:
                agent_return.response = action_return.result['text']
                break
            self._inner_history.append(
                dict(
                    role='system',
                    content=self._protocol.format_response(action_return)))
            logger.info(f"第{turn + 1}轮对话用时：{time.time() - for_start}秒")
        else:
            agent_return.response = default_response
        agent_return.inner_steps = copy.deepcopy(self._inner_history)
        # only append the user and final response
        self._session_history.append(dict(role='user', content=message))
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        logger.info(f"总用时：{time.time() - start}秒")
        return agent_return