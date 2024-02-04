from lagent.agents import ReAct
from lagent import AgentReturn, ActionReturn
import copy
from transformers import GenerationConfig
class MyReAct(ReAct):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def chat(self, message: str) -> AgentReturn:
        self._inner_history = []
        self._inner_history.append(dict(role='user', content=message))
        agent_return = AgentReturn()
        default_response = 'Sorry that I cannot answer your question.'
        gen_config = GenerationConfig(
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            repetition_penalty=1.002,
        )

        for turn in range(self.max_turn):
            prompt = self._protocol.format(
                chat_history=self.session_history,
                inner_step=self._inner_history,
                action_executor=self._action_executor,
                force_stop=(turn == self.max_turn - 1))
            response = self._llm.generate_from_template(prompt, 512, generation_config=gen_config)
            self._inner_history.append(
                dict(role='assistant', content=response))
            thought, action, action_input = self._protocol.parse(
                response, self._action_executor)
            action_return: ActionReturn = self._action_executor(
                action, action_input)

            if action_return.type == "NoAction":
                # 没有获取到action的情况
                action_return.thought = "该回答不需要调用任何Action"
                agent_return.response = response
                print("模型输出异常，未按照指定模板生成Action，直接返回原始输出")
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
        else:
            agent_return.response = default_response
        agent_return.inner_steps = copy.deepcopy(self._inner_history)
        # only append the user and final response
        self._session_history.append(dict(role='user', content=message))
        self._session_history.append(
            dict(role='assistant', content=agent_return.response))
        return agent_return