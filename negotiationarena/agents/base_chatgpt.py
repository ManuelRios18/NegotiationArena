import time
import random
from copy import deepcopy
from negotiationarena.agents.agents import Agent
from negotiationarena.constants import AGENT_ONE, AGENT_TWO


class BaseOpenAIAgent(Agent):
    """
    Contains all the shared logic for any OpenAI-compatible agent.
    """
    def __init__(
            self,
            agent_name: str,
            model: str,
            temperature=0.7,
            max_tokens=400,
            seed=None,
    ):
        super().__init__(agent_name)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.seed = (
            int(self.run_epoch_time_ms) + random.randint(0, 2 ** 16)
            if seed is None
            else seed
        )
        self.client = None

    def init_agent(self, system_prompt, role):
        if AGENT_ONE in self.agent_name:
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
            self.update_conversation_tracking("user", role)
        elif AGENT_TWO in self.agent_name:
            system_prompt = system_prompt + role
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
        else:
            raise ValueError(
                "Agent name must contain 'agent_one' or 'agent_two'")

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "client" and not isinstance(v, str):
                v = v.__class__.__name__
            setattr(result, k, deepcopy(v, memo))
        return result

    def chat(self):
        if self.client is None:
            raise NotImplementedError(
                "Client has not been initialized by the child class.")

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        return chat.choices[0].message.content

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})