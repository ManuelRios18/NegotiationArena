from openai import AzureOpenAI
from negotiationarena.agents.base_chatgpt import BaseOpenAIAgent
from negotiationarena.agents.agent_behaviours import SelfCheckingAgent

class AzureChatGPTAgent(BaseOpenAIAgent):
    def __init__(self, api_key: str, azure_endpoint: str, api_version: str,
                 **kwargs):
        super().__init__(**kwargs)

        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

class SelfCheckingChatGPTAgent(AzureChatGPTAgent, SelfCheckingAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)