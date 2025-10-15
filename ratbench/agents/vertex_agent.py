import time
from copy import deepcopy
from typing import List, Tuple
from ratbench.agents.agents import Agent
from google.oauth2 import service_account
from vertexai import init as vertexai_init
from ratbench.constants import AGENT_ONE, AGENT_TWO
from vertexai.generative_models import Content, Part
from vertexai.generative_models import Content, ChatSession
from vertexai.generative_models import HarmCategory, HarmBlockThreshold
from vertexai.generative_models import GenerationResponse, FinishReason
from vertexai.generative_models import GenerativeModel, GenerationConfig


def gemini_translator(prompt: List[dict]) -> Tuple[str, List[Content], str]:
    prompt_copy = deepcopy(prompt)
    last_message = prompt_copy.pop()
    system = prompt_copy.pop(0)["content"]
    history = []
    for message in prompt_copy:
        role = "model" if message["role"] == "assistant" else "user"
        parts = [Part.from_text(message["content"])]
        content = Content(role=role, parts=parts)
        history.append(content)

    return system, history, last_message["content"]


class VertexAgent(Agent):

    def __init__(
        self,
        agent_name: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
        google_json_creds: dict = None,
        **kwargs
    ) -> None:
        super().__init__(agent_name)
        service_cred = service_account.Credentials.from_service_account_info(
            google_json_creds
        )
        vertexai_init(project=service_cred.project_id,
                      credentials=service_cred)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: \
                HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: \
                HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: \
                HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: \
                HarmBlockThreshold.BLOCK_NONE
        }

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

    def __get_chat(self,
                   system: str,
                   history: List[Content]) -> ChatSession:
        """Method to get chat object for the Gemini Model"""
        model = GenerativeModel(
            model_name=self.model,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            ),
            system_instruction=system
        )
        chat = model.start_chat(
            history=history
        )
        return chat

    def chat(self):
        system, history, message = gemini_translator(self.conversation)
        chat = self.__get_chat(system, history)
        completion = chat.send_message(
            message,
            stream=False,
            safety_settings=self.safety_settings
        )
        return completion.text


    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})
