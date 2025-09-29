from openai import AzureOpenAI
from ratbench.agents.base_chatgpt import BaseOpenAIAgent
from ratbench.agents.agent_behaviours import SelfCheckingAgent

class AzureChatGPTAgent(BaseOpenAIAgent):
    def __init__(self, **kwargs):
        api_key = kwargs.pop("api_key", "")
        azure_endpoint = kwargs.pop("azure_endpoint", "")
        api_version = kwargs.pop("api_version", "")
        super().__init__(**kwargs)
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

class SelfCheckingChatGPTAgent(AzureChatGPTAgent, SelfCheckingAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)