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
        from openai.types.chat import ChatCompletionChunk, ChatCompletion
        super().__init__(config=config, max_concurrent_requests=max_concurrent_requests, max_requests_per_minute=max_requests_per_minute)
        self.client = OpenAI(**kwrs)
        self.async_client = AsyncOpenAI(**kwrs)
        self.model = model
        self.ChatCompletion = ChatCompletion
        self.ChatCompletionChunk = ChatCompletionChunk

    def _format_config(self) -> Dict[str, Any]:
        """
        This method format the LLM configuration with the correct key for the inference engine. 
        """
        formatted_params = self.config.params.copy()
        if "max_new_tokens" in formatted_params:
            formatted_params["max_completion_tokens"] = formatted_params["max_new_tokens"]
            formatted_params.pop("max_new_tokens")

        return formatted_params

    def _format_response(self, response: Any) -> Dict[str, str]:
        """
        Format OpenAI API response (ChatCompletion or ChatCompletionChunk) to a standardized dict.
        
        For streaming (ChatCompletionChunk), returns:
            {"type": "response" | "tool_call_delta", "data": <content>}
        For non-streaming (ChatCompletion), returns:
            {"response": str, "reasoning": str, "tool_calls": list}
        """
        # Streaming response
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Tool call chunks - OpenAI streams incrementally
            if hasattr(delta, "tool_calls") and delta.tool_calls is not None:
                if isinstance(delta.tool_calls, list) and len(delta.tool_calls) > 0:
                    tool_call_deltas = []
                    for tc in delta.tool_calls:
                        tool_call_deltas.append({
                            "index": tc.index,
                            "id": tc.id,
                            "name": tc.function.name if tc.function else None,
                            "arguments": tc.function.arguments if tc.function else ""
                        })
                    return {"type": "tool_call_delta", "data": tool_call_deltas}
            
            # Response content chunks (OpenAI doesn't have reasoning_content)
            chunk_text = getattr(delta, "content", "") or ""
            return {"type": "response", "data": chunk_text}

        # Non-streaming response
        message = response.choices[0].message
        
        # Response extraction
        response_text = message.content if hasattr(message, "content") else ""
        response_text = response_text if response_text is not None else ""
        
        # Reasoning extraction (OpenAI o1 models may have this)
        reasoning = ""
        if hasattr(message, "reasoning"):
            reasoning = message.reasoning if message.reasoning is not None else ""
        
        # Tool calls extraction
        tool_calls = []
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            for tc in message.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
        
        return {
            "response": response_text,
            "reasoning": reasoning,
            "tool_calls": tool_calls
        }

    def _log_messages(self, processed_messages: List[Dict[str, str]], res_dict: Dict[str, str], messages_logger: MessagesLogger):
        """Helper method to log messages."""
        if not messages_logger.store_images:
            for msg in processed_messages:
                if "content" in msg and isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and content.get("type") == "image_url":
                            content["image_url"]["url"] = "[image]"

        processed_messages.append({
            "role": "assistant",
            "content": res_dict.get("response", ""),
            "reasoning": res_dict.get("reasoning", ""),
            "tool_calls": res_dict.get("tool_calls", None)
        })
        messages_logger.log_messages(processed_messages)

    def _chat_stream(self, processed_messages: List[Dict[str, str]], **kwargs) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Helper method for streaming chat responses.
        Handles OpenAI's incremental tool call streaming.
        
        Yields:
            {"type": "response" | "tool_calls", "data": <content>}
        
        Returns:
            Aggregated response dict with keys: response, reasoning, tool_calls
        """
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=processed_messages,
            stream=True,
            **self.formatted_params,
            **kwargs
        )
        
        agg_response = {"reasoning": "", "response": "", "tool_calls": []}
        tool_call_accumulators = {}  # index -> {"id": str, "name": str, "arguments": str}
        
        for chunk in response_stream:
            if len(chunk.choices) > 0:
                chunk_dict = self._format_response(chunk)
                
                # Handle incremental tool call deltas - accumulate without yielding
                if chunk_dict.get("type") == "tool_call_delta":
                    for tc_delta in chunk_dict["data"]:
                        idx = tc_delta["index"]
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {"id": "", "name": "", "arguments": ""}
                        
                        # Accumulate non-None values
                        if tc_delta["id"] is not None:
                            tool_call_accumulators[idx]["id"] = tc_delta["id"]
                        if tc_delta["name"] is not None:
                            tool_call_accumulators[idx]["name"] = tc_delta["name"]
                        if tc_delta["arguments"]:
                            tool_call_accumulators[idx]["arguments"] += tc_delta["arguments"]
                    
                    # Don't yield incremental tool call deltas
                    continue
                
                # Yield non-tool-call chunks normally
                yield chunk_dict
                
                # Aggregate response
                if chunk_dict.get("type") == "response":
                    agg_response["response"] += chunk_dict["data"]
                
                if chunk.choices[0].finish_reason == "length":
                    warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
        
        # After streaming completes, convert accumulated tool calls to final format
        if tool_call_accumulators:
            sorted_tool_calls = sorted(tool_call_accumulators.items(), key=lambda x: x[0])
            agg_response["tool_calls"] = [
                {"name": tc["name"], "arguments": tc["arguments"]}
                for idx, tc in sorted_tool_calls
            ]
            # Yield the complete tool calls at the end
            yield {"type": "tool_calls", "data": agg_response["tool_calls"]}
        
        return agg_response

    def chat(self, messages: List[Dict[str, str]], verbose: bool = False, stream: bool = False, 
             messages_logger: MessagesLogger = None, **kwargs) -> Union[Dict[str, str], Generator[Dict[str, str], None, None]]:
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
        messages_logger : MessagesLogger, Optional
            the message logger that logs the chat messages.

        Returns:
        -------
        response : Union[Dict[str,str], Generator[Dict[str, str], None, None]]
            a dict {"reasoning": <reasoning>, "response": <response>, "tool_calls": <tool_calls>} 
            or Generator {"type": <response or tool_calls>, "data": <content>}
        """
        processed_messages = self.config.preprocess_messages(messages)

        if stream:
            def _stream_generator():
                stream_gen = self._chat_stream(processed_messages, **kwargs)
                agg_response = None
                
                try:
                    while True:
                        chunk_dict = next(stream_gen)
                        yield chunk_dict
                except StopIteration as e:
                    agg_response = e.value
                
                # Postprocess and log
                agg_response = agg_response if agg_response is not None else {"reasoning": "", "response": "", "tool_calls": []}
                res_dict = self.config.postprocess_response(agg_response)
                if messages_logger:
                    self._log_messages(processed_messages, res_dict, messages_logger)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            phase = ""
            agg_response = None
            
            stream_gen = self._chat_stream(processed_messages, **kwargs)
            
            try:
                while True:
                    chunk_dict = next(stream_gen)
                    chunk_type = chunk_dict["type"]
                    chunk_data = chunk_dict["data"]
                    
                    # Print with phase headers
                    if chunk_type == "tool_calls":
                        # Tool calls come as complete list at the end
                        for tool_call in chunk_data:
                            print(f"\n--- Tool Call: {tool_call['name']} ---")
                            print(f"Arguments: {tool_call['arguments']}")
                    else:
                        if phase != chunk_type and chunk_data != "":
                            print(f"\n--- {chunk_type.capitalize()} ---")
                            phase = chunk_type
                        print(chunk_data, end="", flush=True)
                        
            except StopIteration as e:
                agg_response = e.value
            
            print('\n')
            agg_response = agg_response if agg_response is not None else {"reasoning": "", "response": "", "tool_calls": []}
            formatted_response = agg_response

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

            formatted_response = self._format_response(chat_completion)
            
        # Postprocess response by config
        res_dict = self.config.postprocess_response(formatted_response)
        
        # Write to messages log
        if messages_logger:
            self._log_messages(processed_messages, res_dict, messages_logger)

        return res_dict

    async def chat_async(self, messages: List[Dict[str, str]], messages_logger: MessagesLogger = None, **kwargs) -> Dict[str, str]:
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

            formatted_response = self._format_response(chat_completion)

            # Postprocess response by config
            res_dict = self.config.postprocess_response(formatted_response)
            
            # Write to messages log
            if messages_logger:
                self._log_messages(processed_messages, res_dict, messages_logger)

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
        model : str
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
        This method format the response (ChatCompletion or ChatCompletionChunk) from OpenAI API to a dict.
        Must be implemented by child classes to handle backend-specific response formats.
        """
        return NotImplemented
    
    
    def _log_messages(self, processed_messages:List[Dict[str,str]], res_dict:Dict[str,str], messages_logger:MessagesLogger):
        """Helper method to log messages."""
        if not messages_logger.store_images:
            for msg in processed_messages:
                if "content" in msg and isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and content.get("type") == "image_url":
                            content["image_url"]["url"] = "[image]"

        processed_messages.append({
            "role": "assistant", 
            "content": res_dict.get("response", ""), 
            "reasoning": res_dict.get("reasoning", ""),
            "tool_calls": res_dict.get("tool_calls", None)
        })
        messages_logger.log_messages(processed_messages)

    def _chat_stream(self, processed_messages:List[Dict[str,str]], **kwargs) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Helper method for streaming chat responses. Can be overridden by child classes
        to handle backend-specific streaming behavior.
        """
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=processed_messages,
            stream=True,
            **self.formatted_params,
            **kwargs
        )
        
        agg_response = {"reasoning": "", "response": "", "tool_calls": []}
        
        for chunk in response_stream:
            if len(chunk.choices) > 0:
                chunk_dict = self._format_response(chunk)
                yield chunk_dict
                
                # Aggregate the response
                if chunk_dict.get("type") == "reasoning":
                    agg_response["reasoning"] += chunk_dict["data"]
                elif chunk_dict.get("type") == "response":
                    agg_response["response"] += chunk_dict["data"]
                elif chunk_dict.get("type") == "tool_calls":
                    agg_response["tool_calls"].extend(chunk_dict["data"])
                
                if chunk.choices[0].finish_reason == "length":
                    warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
        
        return agg_response

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
                stream_gen = self._chat_stream(processed_messages, **kwargs)
                agg_response = None
                
                try:
                    while True:
                        chunk_dict = next(stream_gen)
                        yield chunk_dict
                except StopIteration as e:
                    agg_response = e.value 
                
                # Postprocess and log
                agg_response = agg_response if agg_response is not None else {"reasoning": "", "response": "", "tool_calls": []}
                res_dict = self.config.postprocess_response(agg_response)
                if messages_logger:
                    self._log_messages(processed_messages, res_dict, messages_logger)

            return self.config.postprocess_response(_stream_generator())

        elif verbose:
            phase = ""
            agg_response = None
            
            stream_gen = self._chat_stream(processed_messages, **kwargs)
            
            try:
                while True:
                    chunk_dict = next(stream_gen)
                    chunk_type = chunk_dict["type"]
                    chunk_data = chunk_dict["data"]
                    
                    # Print with phase headers
                    if phase != chunk_type and chunk_data != "":
                        print(f"\n--- {chunk_type.capitalize()} ---")
                        phase = chunk_type
                    
                    if chunk_type == "tool_calls":
                        for tool_call in chunk_data:
                            print(f"Tool: {tool_call['name']}, Args: {tool_call['arguments']}")
                    else:
                        print(chunk_data, end="", flush=True)
            except StopIteration as e:
                agg_response = e.value  # Capture the return value here!
            
            print('\n')
            agg_response = agg_response if agg_response is not None else {"reasoning": "", "response": "", "tool_calls": []}
            formatted_response = agg_response

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

            formatted_response = self._format_response(chat_completion)
            
        # Postprocess response by config
        res_dict = self.config.postprocess_response(formatted_response)
        
        # Write to messages log
        if messages_logger:
            self._log_messages(processed_messages, res_dict, messages_logger)

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

            formatted_response = self._format_response(chat_completion)
            res_dict = self.config.postprocess_response(formatted_response)
            
            if messages_logger:
                self._log_messages(processed_messages, res_dict, messages_logger)

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
        super().__init__(model=model, api_key=api_key, base_url=base_url, config=config, 
                         max_concurrent_requests=max_concurrent_requests, 
                         max_requests_per_minute=max_requests_per_minute, **kwrs)

    def _format_response(self, response: Any) -> Dict[str, str]:
        """Format vLLM-specific responses."""
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Tool calls - vLLM streams incrementally, return delta for accumulation
            if delta.tool_calls is not None:
                tool_call_deltas = []
                for tc in delta.tool_calls:
                    tool_call_deltas.append({
                        "index": tc.index,
                        "id": tc.id,
                        "name": tc.function.name if tc.function else None,
                        "arguments": tc.function.arguments if tc.function else ""
                    })
                return {"type": "tool_call_delta", "data": tool_call_deltas}
            
            # Reasoning content
            if hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content") is not None:
                chunk_text = getattr(delta, "reasoning_content", "") or ""
                return {"type": "reasoning", "data": chunk_text}
            
            # Response content
            chunk_text = getattr(delta, "content", "") or ""
            return {"type": "response", "data": chunk_text}

        # Non-streaming
        message = response.choices[0].message
        tool_calls = None
        if hasattr(message, "tool_calls") and message.tool_calls is not None:
            tool_calls = []
            for tc in message.tool_calls:
                tool_calls.append({"name": tc.function.name, "arguments": tc.function.arguments})
        
        return {
            "reasoning": getattr(message, "reasoning_content", "") or "",
            "response": getattr(message, "content", "") or "",
            "tool_calls": tool_calls
        }

    def _chat_stream(self, processed_messages: List[Dict[str, str]], **kwargs) -> Generator[Dict[str, Any], None, Dict[str, Any]]:
        """
        Override parent's _chat_stream to handle vLLM's incremental tool call streaming.
        vLLM sends tool calls in multiple chunks that need to be accumulated.
        """
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=processed_messages,
            stream=True,
            **self.formatted_params,
            **kwargs
        )
        
        agg_response = {"reasoning": "", "response": "", "tool_calls": []}
        tool_call_accumulators = {}  # index -> {"id": str, "name": str, "arguments": str}
        
        for chunk in response_stream:
            if len(chunk.choices) > 0:
                chunk_dict = self._format_response(chunk)
                
                # Handle incremental tool call deltas - accumulate without yielding
                if chunk_dict.get("type") == "tool_call_delta":
                    for tc_delta in chunk_dict["data"]:
                        idx = tc_delta["index"]
                        if idx not in tool_call_accumulators:
                            tool_call_accumulators[idx] = {"id": "", "name": "", "arguments": ""}
                        
                        # Accumulate non-None values
                        if tc_delta["id"] is not None:
                            tool_call_accumulators[idx]["id"] = tc_delta["id"]
                        if tc_delta["name"] is not None:
                            tool_call_accumulators[idx]["name"] = tc_delta["name"]
                        if tc_delta["arguments"]:
                            tool_call_accumulators[idx]["arguments"] += tc_delta["arguments"]
                    
                    # Don't yield incremental tool call deltas
                    continue
                
                # Yield non-tool-call chunks normally
                yield chunk_dict
                
                # Aggregate reasoning and response
                if chunk_dict.get("type") == "reasoning":
                    agg_response["reasoning"] += chunk_dict["data"]
                elif chunk_dict.get("type") == "response":
                    agg_response["response"] += chunk_dict["data"]
                
                if chunk.choices[0].finish_reason == "length":
                    warnings.warn("Model stopped generating due to context length limit.", RuntimeWarning)
        
        # After streaming completes, convert accumulated tool calls to final format
        if tool_call_accumulators:
            # Sort by index to maintain order
            sorted_tool_calls = sorted(tool_call_accumulators.items(), key=lambda x: x[0])
            agg_response["tool_calls"] = [
                {"name": tc["name"], "arguments": tc["arguments"]}
                for idx, tc in sorted_tool_calls
            ]
            # Yield the complete tool calls at the end
            yield {"type": "tool_calls", "data": agg_response["tool_calls"]}
        
        return agg_response
        
    

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
        super().__init__(model=model, api_key=api_key, base_url=base_url, config=config, 
                         max_concurrent_requests=max_concurrent_requests, 
                         max_requests_per_minute=max_requests_per_minute, **kwrs)

    def _format_response(self, response: Any) -> Dict[str, str]:
        """Format SGLang-specific responses."""
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Tool calls - SGLang returns complete tool call in single chunk
            if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls") is not None:
                if isinstance(delta.tool_calls, list):
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        function = tool_call.function
                        tool_calls.append({"name": function.name, "arguments": function.arguments})
                    return {"type": "tool_calls", "data": tool_calls}
            
            # Reasoning content
            if hasattr(delta, "reasoning_content") and getattr(delta, "reasoning_content") is not None:
                chunk_text = getattr(delta, "reasoning_content", "") or ""
                return {"type": "reasoning", "data": chunk_text}
            
            # Response content
            chunk_text = getattr(delta, "content", "") or ""
            return {"type": "response", "data": chunk_text}

        # Non-streaming
        message = response.choices[0].message
        response_text = message.content if hasattr(message, "content") else ""
        response_text = response_text if response_text is not None else ""
        reasoning = message.reasoning if hasattr(message, "reasoning") else ""
        reasoning = reasoning if reasoning is not None else ""
        
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function = tool_call.function
                tool_calls.append({"name": function.name, "arguments": function.arguments})
        
        return {"response": response_text, "reasoning": reasoning, "tool_calls": tool_calls}
    
    

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
        """Format SGLang-specific responses."""
        if isinstance(response, self.ChatCompletionChunk):
            delta = response.choices[0].delta
            
            # Tool calls - SGLang returns complete tool call in single chunk
            if hasattr(delta, "tool_calls") and getattr(delta, "tool_calls") is not None:
                if isinstance(delta.tool_calls, list):
                    tool_calls = []
                    for tool_call in delta.tool_calls:
                        function = tool_call.function
                        tool_calls.append({"name": function.name, "arguments": function.arguments})
                    return {"type": "tool_calls", "data": tool_calls}
            
            # Reasoning content
            if hasattr(delta, "reasoning") and getattr(delta, "reasoning") is not None:
                chunk_text = getattr(delta, "reasoning", "") or ""
                return {"type": "reasoning", "data": chunk_text}
            
            # Response content
            chunk_text = getattr(delta, "content", "") or ""
            return {"type": "response", "data": chunk_text}

        # Non-streaming
        message = response.choices[0].message
        response_text = message.content if hasattr(message, "content") else ""
        response_text = response_text if response_text is not None else ""
        reasoning = message.reasoning if hasattr(message, "reasoning") else ""
        reasoning = reasoning if reasoning is not None else ""
        
        tool_calls = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                function = tool_call.function
                tool_calls.append({"name": function.name, "arguments": function.arguments})
        
        return {"response": response_text, "reasoning": reasoning, "tool_calls": tool_calls}
    