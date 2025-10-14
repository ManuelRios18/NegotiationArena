import boto3
from copy import deepcopy
from typing import List, Optional
from ratbench.agents.agents import Agent
from ratbench.constants import AGENT_ONE, AGENT_TWO

class BedrockAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
        region: str = "us-east-1",
        **kwargs
    ) -> None:
        super().__init__(agent_name)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._system_blocks: List[dict] = []
        self.conversation: List[dict] = []

        self.client = boto3.client("bedrock-runtime", region_name=region)

    def init_agent(self, system_prompt: str, role: str):
        """
        Builds system prompt + initial user role message.
        In Bedrock, system is passed separately via `system=[...]`.
        """
        if AGENT_ONE in self.agent_name:
            self._system_blocks = [{"text": system_prompt}]
            self.update_conversation_tracking("user", role)

        elif AGENT_TWO in self.agent_name:
            self._system_blocks = [{"text": system_prompt + role}]
        else:
            raise ValueError("Agent name must contain "
                             "'agent_one' or 'agent_two'")

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "client" and not isinstance(v, str):
                v = v.__class__.__name__
            setattr(result, k, deepcopy(v, memo))
        return result

    def chat(self) -> str:
        resp = self._converse(
            messages=self.conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system=self._system_blocks or None,
        )
        return resp["output"]["message"]["content"][0]["text"]

    def _converse(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: Optional[int],
        system: Optional[List[dict]] = None,
    ):
        inference_config = {"temperature": temperature}
        if max_tokens is not None:
            inference_config["maxTokens"] = max_tokens

        kwargs = {
            "modelId": self.model,
            "messages": messages,
            "inferenceConfig": inference_config,
        }
        if system:
            kwargs["system"] = system

        return self.client.converse(**kwargs)

    def update_conversation_tracking(self, role: str, message: str):
        if role == "system":
            self._system_blocks = [{"text": message}]
        else:
            self.conversation.append({"role": role,
                                      "content": [{"text": message}]})