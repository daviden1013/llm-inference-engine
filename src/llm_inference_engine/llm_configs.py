import abc
import re
import warnings
from typing import List, Dict, Union, Generator, Any, AsyncGenerator


class LLMConfig(abc.ABC):
    def __init__(self, **kwargs):
        """
        This is an abstract class to provide interfaces for LLM configuration. 
        Children classes that inherts this class can be used in extrators and prompt editor.
        Common LLM parameters: max_new_tokens, temperature, top_p, top_k, min_p.
        """
        self.params = kwargs.copy()


    @abc.abstractmethod
    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        This method preprocesses the input messages before passing them to the LLM.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        return NotImplemented

    @abc.abstractmethod
    def postprocess_response(self, response:Union[str, Dict[str, Any], Generator[str, None, None], AsyncGenerator[str, None]]) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None], AsyncGenerator[Dict[str, Any], None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Dict[str, Any], Generator[Dict[str, Any], None, None], AsyncGenerator[str, None]]
            the LLM response. Can be a dict, a generator, or an async generator. 
        
        Returns:
        -------
        response : Union[Dict[str, Any], Generator[Dict[str, Any], None, None], AsyncGenerator[Dict[str, Any], None]]
            the postprocessed LLM response
        """
        return NotImplemented


class BasicLLMConfig(LLMConfig):
    def __init__(self, max_new_tokens:int=2048, temperature:float=0.0, **kwargs):
        """
        The basic LLM configuration for most non-reasoning models.
        """
        super().__init__(**kwargs)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.params["max_new_tokens"] = self.max_new_tokens
        self.params["temperature"] = self.temperature

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        This method preprocesses the input messages before passing them to the LLM.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        return messages.copy()

    def postprocess_response(self, response:Union[str, Dict[str, Any], Generator[str, None, None], AsyncGenerator[str, None]]) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None], AsyncGenerator[Dict[str, Any], None]]:
        """
        This method postprocesses the LLM response after it is generated.

        Parameters:
        ----------
        response : Union[str, Dict[str, Any], Generator[str, None, None], AsyncGenerator[str, None]]
            the LLM response. Can be a string, a generator, or an async generator.
        
        Returns: Union[Dict[str, Any], Generator[Dict[str, Any], None, None], AsyncGenerator[Dict[str, Any], None]]
            the postprocessed LLM response. 
            if input is a string, the output will be a dict {"response": <response>}. 
            if input is a generator, the output will be a generator {"type": "response", "data": <content>}.
            if input is an async generator, the output will be an async generator {"type": "response", "data": <content>}.
        """
        if isinstance(response, str):
            return {"response": response}
        
        elif isinstance(response, dict):
            if "response" in response or "tool_calls" in response:
                return response
            else:
                warnings.warn(f"Invalid response dict keys: {response.keys()}. Returning default empty dict.", UserWarning)
                return {"response": ""}

        elif isinstance(response, Generator):
            def _process_stream():
                for chunk in response:
                    if isinstance(chunk, dict):
                        yield chunk
                    elif isinstance(chunk, str):
                        yield {"type": "response", "data": chunk}

            return _process_stream()

        elif isinstance(response, AsyncGenerator):
            async def _process_async_stream():
                async for chunk in response:
                    if isinstance(chunk, dict):
                        yield chunk
                    elif isinstance(chunk, str):
                        yield {"type": "response", "data": chunk}
            
            return _process_async_stream()

        else:
            warnings.warn(f"Invalid response type: {type(response)}. Returning default empty dict.", UserWarning)
            return {"response": ""}

class ReasoningLLMConfig(LLMConfig):
    def __init__(self, thinking_token_start="<think>", thinking_token_end="</think>", **kwargs):
        """
        The general LLM configuration for reasoning models.
        """
        super().__init__(**kwargs)
        self.thinking_token_start = thinking_token_start
        self.thinking_token_end = thinking_token_end

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        This method preprocesses the input messages before passing them to the LLM.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        return messages.copy()

    def postprocess_response(self, response:Union[str, Dict[str, str], Generator[str, None, None], AsyncGenerator[str, None]]) -> Union[Dict[str,str], Generator[Dict[str,str], None, None], AsyncGenerator[Dict[str,str], None]]:
        """
        This method postprocesses the LLM response after it is generated.
        1. If input is a string, it will extract the reasoning and response based on the thinking tokens.
        2. If input is a dict, it should contain keys "reasoning", "response", or "tool_calls". This is for inference engines that already parse reasoning, response, and tool calls.
        3. If input is a generator, 
            a. if the chunk is a dict, it should contain keys "type" and "data". This is for inference engines that already parse reasoning, response, and tool calls.
            b. if the chunk is a string, it will yield dicts with keys "type" and "data" based on the thinking tokens.

        Parameters:
        ----------
        response : Union[str, Generator[str, None, None]]
            the LLM response. Can be a string or a generator.
        
        Returns:
        -------
        response : Union[str, Generator[str, None, None]]
            the postprocessed LLM response as a dict {"reasoning": <reasoning>, "response": <content>}
            if input is a generator, the output will be a generator {"type": <reasoning or response>, "data": <content>}.
        """
        if isinstance(response, str):
            # get contents between thinking_token_start and thinking_token_end
            pattern = f"{re.escape(self.thinking_token_start)}(.*?){re.escape(self.thinking_token_end)}"
            match = re.search(pattern, response, re.DOTALL)
            reasoning = match.group(1) if match else ""
            # get response AFTER thinking_token_end
            response = re.sub(f".*?{self.thinking_token_end}", "", response, flags=re.DOTALL).strip()
            return {"reasoning": reasoning, "response": response}

        elif isinstance(response, dict):
            if "reasoning" in response and "response" in response:
                return response
            else:
                warnings.warn(f"Invalid response dict keys: {response.keys()}. Returning default empty dict.", UserWarning)
                return {"reasoning": "", "response": ""}

        elif isinstance(response, Generator):
            def _process_stream():
                think_flag = False
                buffer = ""
                for chunk in response:
                    if isinstance(chunk, dict):
                        yield chunk

                    elif isinstance(chunk, str):
                        buffer += chunk
                        # switch between reasoning and response
                        if self.thinking_token_start in buffer:
                            think_flag = True
                            buffer = buffer.replace(self.thinking_token_start, "")
                        elif self.thinking_token_end in buffer:
                            think_flag = False
                            buffer = buffer.replace(self.thinking_token_end, "")
                        
                        # if chunk is in thinking block, tag it as reasoning; else tag it as response
                        if chunk not in [self.thinking_token_start, self.thinking_token_end]:
                            if think_flag:
                                yield {"type": "reasoning", "data": chunk}
                            else:
                                yield {"type": "response", "data": chunk}

            return _process_stream()

        elif isinstance(response, AsyncGenerator):
            async def _process_async_stream():
                think_flag = False
                buffer = ""
                async for chunk in response:
                    if isinstance(chunk, dict):
                        yield chunk

                    elif isinstance(chunk, str):
                        buffer += chunk
                        # switch between reasoning and response
                        if self.thinking_token_start in buffer:
                            think_flag = True
                            buffer = buffer.replace(self.thinking_token_start, "")
                        elif self.thinking_token_end in buffer:
                            think_flag = False
                            buffer = buffer.replace(self.thinking_token_end, "")
                        
                        # if chunk is in thinking block, tag it as reasoning; else tag it as response
                        if chunk not in [self.thinking_token_start, self.thinking_token_end]:
                            if think_flag:
                                yield {"type": "reasoning", "data": chunk}
                            else:
                                yield {"type": "response", "data": chunk}
            
            return _process_async_stream()
        
        else:
            warnings.warn(f"Invalid response type: {type(response)}. Returning default empty dict.", UserWarning)
            return {"reasoning": "", "response": ""}

class Qwen3LLMConfig(ReasoningLLMConfig):
    def __init__(self, thinking_mode:bool=True, **kwargs):
        """
        The Qwen3 **hybrid thinking** LLM configuration. 
        For Qwen3 thinking 2507, use ReasoningLLMConfig instead; for Qwen3 Instruct, use BasicLLMConfig instead.

        Parameters:
        ----------
        thinking_mode : bool, Optional
            if True, a special token "/think" will be placed after each system and user prompt. Otherwise, "/no_think" will be placed.
        """
        super().__init__(**kwargs)
        self.thinking_mode = thinking_mode

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Append a special token to the system and user prompts.
        The token is "/think" if thinking_mode is True, otherwise "/no_think".

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        thinking_token = "/think" if self.thinking_mode else "/no_think"
        new_messages = []
        for message in messages:
            if message['role'] in ['system', 'user']:
                new_message = {'role': message['role'], 'content': f"{message['content']} {thinking_token}"}
            else:
                new_message = {'role': message['role'], 'content': message['content']}

            new_messages.append(new_message)

        return new_messages


class OpenAIReasoningLLMConfig(ReasoningLLMConfig):
    def __init__(self, reasoning_effort:str=None, **kwargs):
        """
        The OpenAI "o" series configuration.
        1. The reasoning effort as one of {"low", "medium", "high"}.
            For models that do not support setting reasoning effort (e.g., o1-mini, o1-preview), set to None.
        2. The temperature parameter is not supported and will be ignored.
        3. The system prompt is not supported and will be concatenated to the next user prompt.

        Parameters:
        ----------
        reasoning_effort : str, Optional
            the reasoning effort. Must be one of {"low", "medium", "high"}. Default is "low".
        """
        super().__init__(**kwargs)
        if reasoning_effort is not None:
            if reasoning_effort not in ["low", "medium", "high"]:
                raise ValueError("reasoning_effort must be one of {'low', 'medium', 'high'}.")

            self.reasoning_effort = reasoning_effort
            self.params["reasoning_effort"] = self.reasoning_effort

        if "temperature" in self.params:
            warnings.warn("Reasoning models do not support temperature parameter. Will be ignored.", UserWarning)
            self.params.pop("temperature")

    def preprocess_messages(self, messages:List[Dict[str,str]]) -> List[Dict[str,str]]:
        """
        Concatenate system prompts to the next user prompt.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        
        Returns:
        -------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        """
        system_prompt_holder = ""
        new_messages = []
        for i, message in enumerate(messages):
            # if system prompt, store it in system_prompt_holder
            if message['role'] == 'system':
                system_prompt_holder = message['content']
            # if user prompt, concatenate it with system_prompt_holder
            elif message['role'] == 'user':
                if system_prompt_holder:
                    new_message = {'role': message['role'], 'content': f"{system_prompt_holder} {message['content']}"}
                    system_prompt_holder = ""
                else:
                    new_message = {'role': message['role'], 'content': message['content']}

                new_messages.append(new_message)
            # if assistant/other prompt, do nothing
            else:
                new_message = {'role': message['role'], 'content': message['content']}
                new_messages.append(new_message)

        return new_messages