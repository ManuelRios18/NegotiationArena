import os
from openai import OpenAI
from negotiationarena.agents.base_chatgpt import BaseOpenAIAgent
from negotiationarena.agents.agent_behaviours import SelfCheckingAgent


class ChatGPTAgent(BaseOpenAIAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class SelfCheckingChatGPTAgent(ChatGPTAgent, SelfCheckingAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
