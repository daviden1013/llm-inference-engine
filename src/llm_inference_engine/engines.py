import abc
import os
import warnings
import importlib.util
from typing import Any, List, Dict, Union, Generator
from llm_inference_engine.utils import MessagesLogger, ConcurrencyLimiter, SlideWindowRateLimiter
from llm_inference_engine.llm_configs import LLMConfig, BasicLLMConfig


class InferenceEngine:
    def __init__(self, config:LLMConfig=None, max_concurrent_requests:int=None, max_requests_per_minute:int=None):
        """
        This is an abstract class to provide interfaces for LLM inference engines. 
        Children classes that inherts this class can be used in extrators. Must implement chat() method.

        Parameters:
        ----------
        config : LLMConfig
            the LLM configuration. Must be a child class of LLMConfig.
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        # Initialize LLM configuration
        self.config = config if config else BasicLLMConfig()

        # Format LLM configuration parameters
        self.formatted_params = self._format_config()

        # Initialize concurrency limiter
        self.max_concurrent_requests = max_concurrent_requests
        if self.max_concurrent_requests:
            self.concurrency_limiter = ConcurrencyLimiter(self.max_concurrent_requests)
        else:
            self.concurrency_limiter = None

        # Initialize rate limiter
        self.max_requests_per_minute = max_requests_per_minute
        if self.max_requests_per_minute:
            self.rate_limiter = SlideWindowRateLimiter(self.max_requests_per_minute)
        else:
            self.rate_limiter = None


    @abc.abstractmethod
    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, 
             messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, LLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.  
        Messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        return NotImplemented

    @abc.abstractmethod
    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 

        Return : Dict[str, Any]
            the config parameters.
        """
        return NotImplemented


class OllamaInferenceEngine(InferenceEngine):
    def __init__(self, model_name:str, num_ctx:int=4096, keep_alive:int=300, config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        The Ollama inference engine.

        Parameters:
        ----------
        model_name : str
            the model name exactly as shown in >> ollama ls
        num_ctx : int, Optional
            context length that LLM will evaluate.
        keep_alive : int, Optional
            seconds to hold the LLM after the last API call.
        config : LLMConfig, Optional
            the LLM configuration. 
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        if importlib.util.find_spec("ollama") is None:
            raise ImportError("ollama-python not found. Please install ollama-python (```pip install ollama```).")
        
        from ollama import Client, AsyncClient
        super().__init__(config=config, max_concurrent_requests=max_concurrent_requests, max_requests_per_minute=max_requests_per_minute)
        self.client = Client(**kwrs)
        self.async_client = AsyncClient(**kwrs)
        self.model_name = model_name
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
    
    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["num_predict"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, 
             messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs VLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        Messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)

        options={'num_ctx': self.num_ctx, **self.formatted_params}
        if stream:
            def _stream_generator():
                response_stream = self.client.chat(
                    model=self.model_name, 
                    messages=processed_messages, 
                    options=options,
                    stream=True, 
                    keep_alive=self.keep_alive
                )
                res = {"reasoning": "", "response": ""}
                for chunk in response_stream:
                    if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
                        content_chunk = getattr(getattr(chunk, 'message', {}), 'thinking', '')
                        res["reasoning"] += content_chunk
                        yield {"type": "reasoning", "data": content_chunk}
                    else:
                        content_chunk = getattr(getattr(chunk, 'message', {}), 'content', '')
                        res["response"] += content_chunk
                        yield {"type": "response", "data": content_chunk}

                    if chunk.done_reason == "length":
                        warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
                
                # Postprocess response
                res_dict = self.config.postprocess_response(res)
                # Write to messages log
                if messages_logger:
                    # replace images content with a placeholder "[image]" to save space
                    if not messages_logger.store_images:
                        for messages in processed_messages:
                            if "images" in messages:
                                messages["images"] = ["[image]" for _ in messages["images"]]

                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.client.chat(
                            model=self.model_name, 
                            messages=processed_messages, 
                            options=options,
                            stream=True,
                            keep_alive=self.keep_alive
                        )
            
            res = {"reasoning": "", "response": ""}
            phase = ""
            for chunk in response:
                if hasattr(chunk.message, 'thinking') and chunk.message.thinking:
                    if phase != "reasoning":
                        print("\n--- Reasoning ---")
                        phase = "reasoning"

                    content_chunk = getattr(getattr(chunk, 'message', {}), 'thinking', '')
                    res["reasoning"] += content_chunk
                else:
                    if phase != "response":
                        print("\n--- Response ---")
                        phase = "response"
                    content_chunk = getattr(getattr(chunk, 'message', {}), 'content', '')
                    res["response"] += content_chunk

                print(content_chunk, end='', flush=True)

                if chunk.done_reason == "length":
                    warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
            print('\n')

        else:
            response = self.client.chat(
                                model=self.model_name, 
                                messages=processed_messages, 
                                options=options,
                                stream=False,
                                keep_alive=self.keep_alive
                            )
            res = {"reasoning": getattr(getattr(response, 'message', {}), 'thinking', ''),
                   "response": getattr(getattr(response, 'message', {}), 'content', '')}
        
            if response.done_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            # replace images content with a placeholder "[image]" to save space
            if not messages_logger.store_images:
                for messages in processed_messages:
                    if "images" in messages:
                        messages["images"] = ["[image]" for _ in messages["images"]]

            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
        

    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
        """
        Async version of chat method. Streaming is not supported.
        """
        if self.concurrency_limiter:
            await self.concurrency_limiter.acquire()
        
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            processed_messages = self.config.preprocess_messages(messages)

            response = await self.async_client.chat(
                                model=self.model_name, 
                                messages=processed_messages, 
                                options={'num_ctx': self.num_ctx, **self.formatted_params},
                                stream=False,
                                keep_alive=self.keep_alive
                            )
            
            res = {"reasoning": getattr(getattr(response, 'message', {}), 'thinking', ''),
                   "response": getattr(getattr(response, 'message', {}), 'content', '')}
            
            if response.done_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
            # Postprocess response
            res_dict = self.config.postprocess_response(res)
            # Write to messages log
            if messages_logger:
                # replace images content with a placeholder "[image]" to save space
                if not messages_logger.store_images:
                    for messages in processed_messages:
                        if "images" in messages:
                            messages["images"] = ["[image]" for _ in messages["images"]]

                processed_messages.append({"role": "assistant", 
                                            "content": res_dict.get("response", ""), 
                                            "reasoning": res_dict.get("reasoning", "")})
                messages_logger.log_messages(processed_messages)

            return res_dict
        finally:
            if self.concurrency_limiter:
                self.concurrency_limiter.release()


class HuggingFaceHubInferenceEngine(InferenceEngine):
    def __init__(self, model:str=None, token:Union[str, bool]=None, base_url:str=None, api_key:str=None, config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        The Huggingface_hub InferenceClient inference engine.
        For parameters and documentation, refer to https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client

        Parameters:
        ----------
        model : str
            the model name exactly as shown in Huggingface repo
        token : str, Optional
            the Huggingface token. If None, will use the token in os.environ['HF_TOKEN'].
        base_url : str, Optional
            the base url for the LLM server. If None, will use the default Huggingface Hub URL.
        api_key : str, Optional
            the API key for the LLM server. 
        config : LLMConfig, Optional
            the LLM configuration. 
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        if importlib.util.find_spec("huggingface_hub") is None:
            raise ImportError("huggingface-hub not found. Please install huggingface-hub (```pip install huggingface-hub```).")
        
        from huggingface_hub import InferenceClient, AsyncInferenceClient
        super().__init__(config=config, max_concurrent_requests=max_concurrent_requests, max_requests_per_minute=max_requests_per_minute)
        self.model = model
        self.base_url = base_url
        self.client = InferenceClient(model=model, token=token, base_url=base_url, api_key=api_key, **kwrs)
        self.client_async = AsyncInferenceClient(model=model, token=token, base_url=base_url, api_key=api_key, **kwrs)


    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params


    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, 
             messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.
            
        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                response_stream = self.client.chat.completions.create(
                                    messages=processed_messages,
                                    stream=True,
                                    **self.formatted_params
                                )
                res_text = ""
                for chunk in response_stream:
                    content_chunk = chunk.get('choices')[0].get('delta').get('content')
                    if content_chunk:
                        res_text += content_chunk
                        yield content_chunk

                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    # replace images content with a placeholder "[image]" to save space
                    if not messages_logger.store_images:
                        for messages in processed_messages:
                            if "content" in messages and isinstance(messages["content"], list):
                                for content in messages["content"]:
                                    if isinstance(content, dict) and content.get("type") == "image_url":
                                        content["image_url"]["url"] = "[image]"

                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

            return self.config.postprocess_response(_stream_generator())
        
        elif verbose:
            response = self.client.chat.completions.create(
                            messages=processed_messages,
                            stream=True,
                            **self.formatted_params
                        )
            
            res = ''
            for chunk in response:
                content_chunk = chunk.get('choices')[0].get('delta').get('content')
                if content_chunk:
                    res += content_chunk
                    print(content_chunk, end='', flush=True)

        
        else:
            response = self.client.chat.completions.create(
                                messages=processed_messages,
                                stream=False,
                                **self.formatted_params
                            )
            res = response.choices[0].message.content

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            # replace images content with a placeholder "[image]" to save space
            if not messages_logger.store_images:
                for messages in processed_messages:
                    if "content" in messages and isinstance(messages["content"], list):
                        for content in messages["content"]:
                            if isinstance(content, dict) and content.get("type") == "image_url":
                                content["image_url"]["url"] = "[image]"

            processed_messages.append({"role": "assistant", 
                                       "content": res_dict.get("response", ""), 
                                       "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    
    
    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
        """
        Async version of chat method. Streaming is not supported.
        """
        if self.concurrency_limiter:
            await self.concurrency_limiter.acquire()
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            processed_messages = self.config.preprocess_messages(messages)

            response = await self.client_async.chat.completions.create(
                        messages=processed_messages,
                        stream=False,
                        **self.formatted_params
                    )
        
            res = response.choices[0].message.content
            # Postprocess response
            res_dict = self.config.postprocess_response(res)
            # Write to messages log
            if messages_logger:
                # replace images content with a placeholder "[image]" to save space
                if not messages_logger.store_images:
                    for messages in processed_messages:
                        if "content" in messages and isinstance(messages["content"], list):
                            for content in messages["content"]:
                                if isinstance(content, dict) and content.get("type") == "image_url":
                                    content["image_url"]["url"] = "[image]"

                processed_messages.append({"role": "assistant", 
                                           "content": res_dict.get("response", ""), 
                                           "reasoning": res_dict.get("reasoning", "")})
                messages_logger.log_messages(processed_messages)

            return res_dict
        finally:
            if self.concurrency_limiter:
                self.concurrency_limiter.release()

class OpenAIInferenceEngine(InferenceEngine):
    def __init__(self, model:str, config:LLMConfig=None, max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        The OpenAI API inference engine. 
        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as described in https://platform.openai.com/docs/models
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import OpenAI, AsyncOpenAI
        super().__init__(config=config, max_concurrent_requests=max_concurrent_requests, max_requests_per_minute=max_requests_per_minute)
        self.client = OpenAI(**kwrs)
        self.async_client = AsyncOpenAI(**kwrs)
        self.model = model

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_completion_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, messages_logger:MessagesLogger=None, **kwargs) -> Union[Dict[str, str], Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                response_stream = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=processed_messages,
                                        stream=True,
                                        **self.formatted_params,
                                        **kwargs
                                    )
                formatted_response = {"response": "", "tool_calls": []}
                for chunk in response_stream:
                    if len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
            
                        # Tool call chunks
                        if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls") is not None:
                            if isinstance(delta.tool_calls, list) and len(delta.tool_calls) > 0:
                                tool_call = delta.tool_calls[0]
                                function = tool_call.function
                                function_name = function.name
                                function_arguments = function.arguments
                                chunk_dict = {"type": "tool_calls", "data": {"name": function_name, "arguments": function_arguments}}
                        
                        # Response content chunks
                        else:
                            chunk_text = getattr(delta, "content", "")
                            if chunk_text is None:
                                chunk_text = ""
                            chunk_dict = {"type": "response", "data": chunk_text}
                        
                        yield chunk_dict

                        if chunk_dict.get("type") == "response":
                            formatted_response["response"] += chunk_dict["data"]
                        if chunk_dict.get("type") == "tool_calls":
                            # if first chunk of a tool call
                            if chunk_dict['data']['name']:
                                formatted_response["tool_calls"].append({"name": chunk_dict['data']['name'], "arguments": chunk_dict['data']['arguments']})
                            else:
                                formatted_response["tool_calls"][-1]["arguments"] += chunk_dict['data']['arguments']

                        if chunk.choices[0].finish_reason == "length":
                            warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

                # Postprocess response
                res_dict = self.config.postprocess_response(formatted_response)
                # Write to messages log
                if messages_logger:
                    # replace images content with a placeholder "[image]" to save space
                    if not messages_logger.store_images:
                        for messages in processed_messages:
                            if "content" in messages and isinstance(messages["content"], list):
                                for content in messages["content"]:
                                    if isinstance(content, dict) and content.get("type") == "image_url":
                                        content["image_url"]["url"] = "[image]"

                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "tool_calls": res_dict.get("tool_calls", None)})
                    messages_logger.log_messages(processed_messages)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=True,
                **self.formatted_params,
                **kwargs
            )
            formatted_response = {"response": "", "tool_calls": []}
            for chunk in response_stream:
                if len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
            
                    # Tool call chunks
                    if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls") is not None:
                        if isinstance(delta.tool_calls, list) and len(delta.tool_calls) > 0:
                            tool_call = delta.tool_calls[0]
                            function = tool_call.function
                            function_name = function.name
                            function_arguments = function.arguments
                            chunk_dict = {"type": "tool_calls", "data": {"name": function_name, "arguments": function_arguments}}
                    
                    # Response content chunks
                    else:
                        chunk_text = getattr(delta, "content", "")
                        if chunk_text is None:
                            chunk_text = ""
                        chunk_dict = {"type": "response", "data": chunk_text}
                        
                    # Verbose printing
                    phase = ""
                    if chunk_dict.get("type") == "tool_calls":
                        # if first chunk of a tool call
                        if chunk_dict['data']['name']:
                            print(f"\n--- Tool Call: {chunk_dict['data']['name']} ---")
                            formatted_response["tool_calls"].append({"name": chunk_dict['data']['name'], "arguments": chunk_dict['data']['arguments']})
                        else:
                            formatted_response["tool_calls"][-1]["arguments"] += chunk_dict['data']['arguments']
                        print(chunk_dict['data']['arguments'], end="", flush=True)

                    else:
                        chunk_text = chunk_dict["data"]
                        formatted_response[chunk_dict["type"]] += chunk_text
                        if phase != chunk_dict["type"] and chunk_text != "":
                            print(f"\n--- {chunk_dict['type'].capitalize()} ---")
                            phase = chunk_dict["type"]

                        print(chunk_text, end="", flush=True)

                    if chunk.choices[0].finish_reason == "length":
                        warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            print('\n')

        else:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params,
                **kwargs
            )

            if chat_completion.choices[0].finish_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            # Apply OpenAI-compatible engine-specific response formatting
            message = chat_completion.choices[0].message
            # Response extraction
            response = message.content if hasattr(message, "content") else ""
            response = response if response is not None else ""
            # Reasoning extraction
            reasoning = message.reasoning if hasattr(message, "reasoning") else ""
            reasoning = reasoning if reasoning is not None else ""
            # Tool calls extraction
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    function_name = function.name
                    function_arguments = function.arguments
                    tool_calls.append({"name": function_name, "arguments": function_arguments})

            formatted_response = {"response": response, "reasoning": reasoning, "tool_calls": tool_calls}
            
        # Postprocess response by config
        res_dict = self.config.postprocess_response(formatted_response)
        # Write to messages log
        if messages_logger:
            # replace images content with a placeholder "[image]" to save space
            if not messages_logger.store_images:
                for messages in processed_messages:
                    if "content" in messages and isinstance(messages["content"], list):
                        for content in messages["content"]:
                            if isinstance(content, dict) and content.get("type") == "image_url":
                                content["image_url"]["url"] = "[image]"

            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", ""),
                                    "tool_calls": res_dict.get("tool_calls", None)})
            messages_logger.log_messages(processed_messages)

        return res_dict
    

    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None, **kwargs) -> Dict[str,str]:
        """
        Async version of chat method. Streaming is not supported.
        """
        if self.concurrency_limiter:
            await self.concurrency_limiter.acquire()
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            processed_messages = self.config.preprocess_messages(messages)

            chat_completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params,
                **kwargs
            )
            
            if chat_completion.choices[0].finish_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            # Apply OpenAI-compatible engine-specific response formatting
            message = chat_completion.choices[0].message
            # Response extraction
            response = message.content if hasattr(message, "content") else ""
            response = response if response is not None else ""
            # Reasoning extraction
            reasoning = message.reasoning if hasattr(message, "reasoning") else ""
            reasoning = reasoning if reasoning is not None else ""
            # Tool calls extraction
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    function_name = function.name
                    function_arguments = function.arguments
                    tool_calls.append({"name": function_name, "arguments": function_arguments})

            formatted_response = {"response": response, "reasoning": reasoning, "tool_calls": tool_calls}

            # Postprocess response by config
            res_dict = self.config.postprocess_response(formatted_response)
            # Write to messages log
            if messages_logger:
                # replace images content with a placeholder "[image]" to save space
                if not messages_logger.store_images:
                    for messages in processed_messages:
                        if "content" in messages and isinstance(messages["content"], list):
                            for content in messages["content"]:
                                if isinstance(content, dict) and content.get("type") == "image_url":
                                    content["image_url"]["url"] = "[image]"

                processed_messages.append({"role": "assistant", 
                                            "content": res_dict.get("response", ""), 
                                            "reasoning": res_dict.get("reasoning", ""),
                                            "tool_calls": res_dict.get("tool_calls", None)})
                messages_logger.log_messages(processed_messages)

            return res_dict
        finally:
            if self.concurrency_limiter:
                self.concurrency_limiter.release()
             

class AzureOpenAIInferenceEngine(OpenAIInferenceEngine):
    def __init__(self, model:str, api_version:str, config:LLMConfig=None, max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        The Azure OpenAI API inference engine.
        For parameters and documentation, refer to 
        - https://azure.microsoft.com/en-us/products/ai-services/openai-service
        - https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart
        
        Parameters:
        ----------
        model : str
            model name as described in https://platform.openai.com/docs/models
        api_version : str
            the Azure OpenAI API version
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        InferenceEngine.__init__(self, config=config, 
                                 max_concurrent_requests=max_concurrent_requests, 
                                 max_requests_per_minute=max_requests_per_minute)

        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import AzureOpenAI, AsyncAzureOpenAI
        self.api_version = api_version
        self.client = AzureOpenAI(api_version=api_version, **kwrs)
        self.async_client = AsyncAzureOpenAI(api_version=api_version, **kwrs)
        self.model = model


class LiteLLMInferenceEngine(InferenceEngine):
    def __init__(self, model:str=None, base_url:str=None, api_key:str=None, config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None):
        """
        The LiteLLM inference engine. 
        For parameters and documentation, refer to https://github.com/BerriAI/litellm?tab=readme-ov-file

        Parameters:
        ----------
        model : str
            the model name
        base_url : str, Optional
            the base url for the LLM server
        api_key : str, Optional
            the API key for the LLM server
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        if importlib.util.find_spec("litellm") is None:
            raise ImportError("litellm not found. Please install litellm (```pip install litellm```).")
        
        import litellm 
        super().__init__(config=config, max_concurrent_requests=max_concurrent_requests, max_requests_per_minute=max_requests_per_minute)
        self.litellm = litellm
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, messages_logger:MessagesLogger=None) -> Union[Dict[str,str], Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"} 
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        messages_logger: MessagesLogger, Optional
            a messages logger that logs the messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)
        
        if stream:
            def _stream_generator():
                response_stream = self.litellm.completion(
                    model=self.model,
                    messages=processed_messages,
                    stream=True,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    **self.formatted_params
                )
                res_text = ""
                for chunk in response_stream:
                    chunk_content = chunk.get('choices')[0].get('delta').get('content')
                    if chunk_content:
                        res_text += chunk_content
                        yield chunk_content

                # Postprocess response
                res_dict = self.config.postprocess_response(res_text)
                # Write to messages log
                if messages_logger:
                    # replace images content with a placeholder "[image]" to save space
                    if not messages_logger.store_images:
                        for messages in processed_messages:
                            if "content" in messages and isinstance(messages["content"], list):
                                for content in messages["content"]:
                                    if isinstance(content, dict) and content.get("type") == "image_url":
                                        content["image_url"]["url"] = "[image]"

                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", "")})
                    messages_logger.log_messages(processed_messages)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.litellm.completion(
                model=self.model,
                messages=processed_messages,
                stream=True,
                base_url=self.base_url,
                api_key=self.api_key,
                **self.formatted_params
            )

            res = ''
            for chunk in response:
                chunk_content = chunk.get('choices')[0].get('delta').get('content')
                if chunk_content:
                    res += chunk_content
                    print(chunk_content, end='', flush=True)
        
        else:
            response = self.litellm.completion(
                    model=self.model,
                    messages=processed_messages,
                    stream=False,
                    base_url=self.base_url,
                    api_key=self.api_key,
                    **self.formatted_params
                )
            res = response.choices[0].message.content

        # Postprocess response
        res_dict = self.config.postprocess_response(res)
        # Write to messages log
        if messages_logger:
            # replace images content with a placeholder "[image]" to save space
            if not messages_logger.store_images:
                for messages in processed_messages:
                    if "content" in messages and isinstance(messages["content"], list):
                        for content in messages["content"]:
                            if isinstance(content, dict) and content.get("type") == "image_url":
                                content["image_url"]["url"] = "[image]"

            processed_messages.append({"role": "assistant", 
                                        "content": res_dict.get("response", ""), 
                                        "reasoning": res_dict.get("reasoning", "")})
            messages_logger.log_messages(processed_messages)

        return res_dict
    
    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None) -> Dict[str,str]:
        """
        Async version of chat method. Streaming is not supported.
        """
        if self.concurrency_limiter:
            await self.concurrency_limiter.acquire()
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire()

            processed_messages = self.config.preprocess_messages(messages)

            response = await self.litellm.acompletion(
                model=self.model,
                messages=processed_messages,
                stream=False,
                base_url=self.base_url,
                api_key=self.api_key,
                **self.formatted_params
            )
            
            res = response.get('choices')[0].get('message').get('content')

            # Postprocess response
            res_dict = self.config.postprocess_response(res)
            # Write to messages log
            if messages_logger:
                # replace images content with a placeholder "[image]" to save space
                if not messages_logger.store_images:
                    for messages in processed_messages:
                        if "content" in messages and isinstance(messages["content"], list):
                            for content in messages["content"]:
                                if isinstance(content, dict) and content.get("type") == "image_url":
                                    content["image_url"]["url"] = "[image]"

                processed_messages.append({"role": "assistant", 
                                        "content": res_dict.get("response", ""), 
                                        "reasoning": res_dict.get("reasoning", "")})
                messages_logger.log_messages(processed_messages)
            return res_dict
        finally:
            if self.concurrency_limiter:
                self.concurrency_limiter.release()


class OpenAICompatibleInferenceEngine(InferenceEngine):
    def __init__(self, model:str, api_key:str, base_url:str, config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        General OpenAI-compatible server inference engine.
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str
            the API key for the vLLM server.
        base_url : str
            the base url for the vLLM server. 
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            maximum number of concurrent requests to the LLM.
        max_requests_per_minute : int, Optional
            maximum number of requests per minute to the LLM.
        """
        if importlib.util.find_spec("openai") is None:
            raise ImportError("OpenAI Python API library not found. Please install OpanAI (```pip install openai```).")
        
        from openai import OpenAI, AsyncOpenAI
        from openai.types.chat import ChatCompletionChunk, ChatCompletion
        self.ChatCompletion = ChatCompletion
        self.ChatCompletionChunk = ChatCompletionChunk
        super().__init__(config=config, max_concurrent_requests=max_concurrent_requests, max_requests_per_minute=max_requests_per_minute)
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwrs)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwrs)
        self.model = model

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_completion_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params
    
    @abc.abstractmethod
    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response from OpenAI API to a dict with keys "type" and "data".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a dict, generator, or object.
        """
        return NotImplemented

    def chat(self, messages:List[Dict[str,str]], verbose:bool=False, stream:bool=False, messages_logger:MessagesLogger=None, **kwargs) -> Union[Dict[str, str], Generator[Dict[str, str], None, None]]:
        """
        This method inputs chat messages and outputs LLM generated text.

        Parameters:
        ----------
        messages : List[Dict[str,str]]
            a list of dict with role and content. role must be one of {"system", "user", "assistant"}
        verbose : bool, Optional
            if True, VLM generated text will be printed in terminal in real-time.
        stream : bool, Optional
            if True, returns a generator that yields the output in real-time.
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>} or Generator {"type": <reasoning or response>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                response_stream = self.client.chat.completions.create(
                                        model=self.model,
                                        messages=processed_messages,
                                        stream=True,
                                        **self.formatted_params,
                                        **kwargs
                                    )
                formatted_response = {"reasoning": "", "response": "", "tool_calls": []}
                for chunk in response_stream:
                    if len(chunk.choices) > 0:
                        chunk_dict = self._format_response(chunk)
                        yield chunk_dict

                        if chunk_dict.get("type") == "reasoning":
                            formatted_response["reasoning"] += chunk_dict["data"]
                        elif chunk_dict.get("type") == "response":
                            formatted_response["response"] += chunk_dict["data"]
                        elif chunk_dict.get("type") == "tool_calls":
                            formatted_response["tool_calls"].extend(chunk_dict["data"])

                        if chunk.choices[0].finish_reason == "length":
                            warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

                # Postprocess response
                res_dict = self.config.postprocess_response(formatted_response)
                # Write to messages log
                if messages_logger:
                    # replace images content with a placeholder "[image]" to save space
                    if not messages_logger.store_images:
                        for messages in processed_messages:
                            if "content" in messages and isinstance(messages["content"], list):
                                for content in messages["content"]:
                                    if isinstance(content, dict) and content.get("type") == "image_url":
                                        content["image_url"]["url"] = "[image]"

                    processed_messages.append({"role": "assistant",
                                                "content": res_dict.get("response", ""),
                                                "reasoning": res_dict.get("reasoning", ""),
                                                "tool_calls": res_dict.get("tool_calls", None)})
                    messages_logger.log_messages(processed_messages)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=True,
                **self.formatted_params,
                **kwargs
            )
            formatted_response = {"reasoning": "", "response": "", "tool_calls": []}
            phase = ""
            for chunk in response:
                if len(chunk.choices) > 0:
                    chunk_dict = self._format_response(chunk)
                    chunk_text = chunk_dict["data"]
                    
                    if chunk_dict.get("type") == "tool_calls":
                        formatted_response["tool_calls"].extend(chunk_dict["data"])
                    else:
                        formatted_response[chunk_dict["type"]] += chunk_text

                    if phase != chunk_dict["type"] and chunk_text != "":
                        print(f"\n--- {chunk_dict['type'].capitalize()} ---")
                        phase = chunk_dict["type"]

                        print(chunk_text, end="", flush=True)

                    if chunk.choices[0].finish_reason == "length":
                        warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            print('\n')

        else:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params,
                **kwargs
            )

            if chat_completion.choices[0].finish_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            # Apply OpenAI-compatible engine-specific response formatting
            formatted_response = self._format_response(chat_completion)
            
        # Postprocess response by config
        res_dict = self.config.postprocess_response(formatted_response)
        # Write to messages log
        if messages_logger:
            # replace images content with a placeholder "[image]" to save space
            if not messages_logger.store_images:
                for messages in processed_messages:
                    if "content" in messages and isinstance(messages["content"], list):
                        for content in messages["content"]:
                            if isinstance(content, dict) and content.get("type") == "image_url":
                                content["image_url"]["url"] = "[image]"

            processed_messages.append({"role": "assistant", 
                                    "content": res_dict.get("response", ""), 
                                    "reasoning": res_dict.get("reasoning", ""),
                                    "tool_calls": res_dict.get("tool_calls", None)})
            messages_logger.log_messages(processed_messages)

        return res_dict
    

    async def chat_async(self, messages:List[Dict[str,str]], messages_logger:MessagesLogger=None, **kwargs) -> Dict[str,str]:
        """
        Async version of chat method. Streaming is not supported.
        """
        if self.concurrency_limiter:
            await self.concurrency_limiter.acquire()
        try:
            if self.rate_limiter:
                await self.rate_limiter.acquire()
            processed_messages = self.config.preprocess_messages(messages)

            chat_completion = await self.async_client.chat.completions.create(
                model=self.model,
                messages=processed_messages,
                stream=False,
                **self.formatted_params,
                **kwargs
            )
            
            if chat_completion.choices[0].finish_reason == "length":
                warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)

            # Apply OpenAI-compatible engine-specific response formatting
            formatted_response = self._format_response(chat_completion)

            # Postprocess response by config
            res_dict = self.config.postprocess_response(formatted_response)
            # Write to messages log
            if messages_logger:
                # replace images content with a placeholder "[image]" to save space
                if not messages_logger.store_images:
                    for messages in processed_messages:
                        if "content" in messages and isinstance(messages["content"], list):
                            for content in messages["content"]:
                                if isinstance(content, dict) and content.get("type") == "image_url":
                                    content["image_url"]["url"] = "[image]"

                processed_messages.append({"role": "assistant", 
                                            "content": res_dict.get("response", ""), 
                                            "reasoning": res_dict.get("reasoning", ""),
                                            "tool_calls": res_dict.get("tool_calls", None)})
                messages_logger.log_messages(processed_messages)

            return res_dict
        finally:
            if self.concurrency_limiter:
                self.concurrency_limiter.release()


class VLLMInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, model:str, api_key:str="", base_url:str="http://localhost:8000/v1", config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        vLLM OpenAI compatible server inference engine.
        https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

        For parameters and documentation, refer to https://platform.openai.com/docs/api-reference/introduction

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str, Optional
            the API key for the vLLM server.
        base_url : str, Optional
            the base url for the vLLM server. 
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            the maximum number of concurrent requests.
        max_requests_per_minute : int, Optional
            the maximum number of requests per minute.
        """
        super().__init__(model=model, 
                         api_key=api_key, 
                         base_url=base_url, 
                         config=config, 
                         max_concurrent_requests=max_concurrent_requests, 
                         max_requests_per_minute=max_requests_per_minute, 
                         **kwrs)


    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response from OpenAI API to a dict with keys "type" and "data".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a dict, generator, or object.
        """
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Check for tool calls in streaming
            if delta.tool_calls is not None:
                return {"type": "tool_calls", "data": delta.tool_calls}
            
            # Check for reasoning content
            if hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content") is not None:
                chunk_text = getattr(delta, "reasoning_content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "reasoning", "data": chunk_text}
            else:
                chunk_text = getattr(delta, "content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "response", "data": chunk_text}

        return {"reasoning": getattr(response.choices[0].message, "reasoning_content", ""),
                "response": getattr(response.choices[0].message, "content", ""),
                "tool_calls": getattr(response.choices[0].message, "tool_calls", None)}
        

class SGLangInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, model:str, api_key:str="", base_url:str="http://localhost:30000/v1", config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        SGLang OpenAI compatible API inference engine.
        https://docs.sglang.ai/basic_usage/openai_api.html

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str, Optional
            the API key for the vLLM server.
        base_url : str, Optional
            the base url for the vLLM server. 
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            the maximum number of concurrent requests.
        max_requests_per_minute : int, Optional
            the maximum number of requests per minute.
        """
        super().__init__(model=model, 
                         api_key=api_key, 
                         base_url=base_url, 
                         config=config, 
                         max_concurrent_requests=max_concurrent_requests, 
                         max_requests_per_minute=max_requests_per_minute, 
                         **kwrs)


    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response (ChatCompletion or ChatCompletionChunk) from OpenAI API to a dict:
        If streaming (ChatCompletionChunk), returns a dict with keys "type" and "data".
        If non-streaming (ChatCompletion), returns a dict with keys "response", "reasoning", and "tool_calls".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a ChatCompletion or ChatCompletionChunk.

        Returns:
        -------
        formatted_response : Dict[str, str]
            If streaming, a dict {"type": <reasoning, response, or tool_calls>, "data": <content>}.
            If non-streaming, a dict {"response": <response>, "reasoning": <reasoning>, "tool_calls": <tool_calls>}.
            Tool calls are returned as List[{"name": <function_name>, "arguments": <function_arguments>}].
        """
        # Streaming response
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Tool call chunks
            if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls") is not None:
                if isinstance(delta.tool_calls, list):
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        function = tool_call.function
                        function_name = function.name
                        function_arguments = function.arguments
                        tool_calls.append({"name": function_name, "arguments": function_arguments})
              
                    return {"type": "tool_calls", "data": tool_calls}
            
            # Reasoning content chunks
            elif hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content") is not None:
                chunk_text = getattr(delta, "reasoning_content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "reasoning", "data": chunk_text}
            
            # Response content chunks
            else:
                chunk_text = getattr(delta, "content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "response", "data": chunk_text}

        # Non-streaming response
        elif isinstance(response, self.ChatCompletion):
            message = response.choices[0].message
            # Response extraction
            response = message.content if hasattr(message, "content") else ""
            response = response if response is not None else ""
            # Reasoning extraction
            reasoning = message.reasoning if hasattr(message, "reasoning") else ""
            reasoning = reasoning if reasoning is not None else ""
            # Tool calls extraction
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    function_name = function.name
                    function_arguments = function.arguments
                    tool_calls.append({"name": function_name, "arguments": function_arguments})
            return {"response": response,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls}
    

class OpenRouterInferenceEngine(OpenAICompatibleInferenceEngine):
    def __init__(self, model:str, api_key:str=None, base_url:str="https://openrouter.ai/api/v1", config:LLMConfig=None, 
                 max_concurrent_requests:int=None, max_requests_per_minute:int=None, **kwrs):
        """
        OpenRouter OpenAI-compatible server inference engine.

        Parameters:
        ----------
        model_name : str
            model name as shown in the vLLM server
        api_key : str, Optional
            the API key for the vLLM server. If None, will use the key in os.environ['OPENROUTER_API_KEY'].
        base_url : str, Optional
            the base url for the vLLM server. 
        config : LLMConfig, Optional
            the LLM configuration.
        max_concurrent_requests : int, Optional
            the maximum number of concurrent requests.
        max_requests_per_minute : int, Optional
            the maximum number of requests per minute.
        """
        self.api_key = api_key
        if self.api_key is None:
            self.api_key = os.getenv("OPENROUTER_API_KEY")
        super().__init__(model=model, 
                         api_key=self.api_key, 
                         base_url=base_url, 
                         config=config, 
                         max_concurrent_requests=max_concurrent_requests, 
                         max_requests_per_minute=max_requests_per_minute, 
                         **kwrs)

    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        This method format the response (ChatCompletion or ChatCompletionChunk) from OpenAI API to a dict:
        If streaming (ChatCompletionChunk), returns a dict with keys "type" and "data".
        If non-streaming (ChatCompletion), returns a dict with keys "response", "reasoning", and "tool_calls".

        Parameters:
        ----------
        response : Any
            the response from OpenAI-compatible API. Could be a ChatCompletion or ChatCompletionChunk.

        Returns:
        -------
        formatted_response : Dict[str, str]
            If streaming, a dict {"type": <reasoning, response, or tool_calls>, "data": <content>}.
            If non-streaming, a dict {"response": <response>, "reasoning": <reasoning>, "tool_calls": <tool_calls>}.
            Tool calls are returned as List[{"name": <function_name>, "arguments": <function_arguments>}].
        """
        # Streaming response
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Tool call chunks
            if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls") is not None:
                if isinstance(delta.tool_calls, list):
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        function = tool_call.function
                        function_name = function.name
                        function_arguments = function.arguments
                        tool_calls.append({"name": function_name, "arguments": function_arguments})
              
                    return {"type": "tool_calls", "data": tool_calls}
            
            # Reasoning content chunks
            elif hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content") is not None:
                chunk_text = getattr(delta, "reasoning_content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "reasoning", "data": chunk_text}
            
            # Response content chunks
            else:
                chunk_text = getattr(delta, "content", "")
                if chunk_text is None:
                    chunk_text = ""
                return {"type": "response", "data": chunk_text}

        # Non-streaming response
        elif isinstance(response, self.ChatCompletion):
            message = response.choices[0].message
            # Response extraction
            response = message.content if hasattr(message, "content") else ""
            response = response if response is not None else ""
            # Reasoning extraction
            reasoning = message.reasoning if hasattr(message, "reasoning") else ""
            reasoning = reasoning if reasoning is not None else ""
            # Tool calls extraction
            tool_calls = []
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function = tool_call.function
                    function_name = function.name
                    function_arguments = function.arguments
                    tool_calls.append({"name": function_name, "arguments": function_arguments})
            return {"response": response,
                    "reasoning": reasoning,
                    "tool_calls": tool_calls}