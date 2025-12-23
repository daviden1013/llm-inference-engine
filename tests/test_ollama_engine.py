import pytest
import os
from typing import Generator
from llm_inference_engine import OllamaInferenceEngine, BasicLLMConfig, ReasoningLLMConfig, Qwen3LLMConfig

# --- CONFIGURATION ---
BASIC_MODEL_NAME = "llama3.1:8b-instruct-q8_0" 
REASON_MODEL_NAME = "gpt-oss:20b"

@pytest.fixture
def basic_engine():
    """Returns an Ollama engine with Basic configuration."""
    config = BasicLLMConfig(max_new_tokens=100, temperature=0.7)
    return OllamaInferenceEngine(model_name=BASIC_MODEL_NAME, config=config)

def test_ollama_chat_sync_basic(basic_engine):
    """Test synchronous chat with Basic Config."""
    messages = [{"role": "user", "content": "Say hello!"}]
    response = basic_engine.chat(messages, verbose=True)
    
    assert isinstance(response, dict)
    assert "response" in response
    assert len(response["response"]) > 0
    print(f"\n[Sync] Response: {response['response']}")

def test_ollama_chat_stream_basic(basic_engine):
    """Test streaming chat with Basic Config."""
    messages = [{"role": "user", "content": "Count to 5."}]
    stream = basic_engine.chat(messages, stream=True)
    
    assert isinstance(stream, Generator)
    
    full_content = ""
    for chunk in stream:
        # Check structure of chunk based on BasicLLMConfig.postprocess_response logic
        # It yields {"type": "response", "data": "..."}
        assert isinstance(chunk, dict)
        assert "type" in chunk
        assert "data" in chunk
        full_content += chunk["data"]
        
    assert len(full_content) > 0
    print(f"\n[Stream] Full Content: {full_content}")

@pytest.mark.asyncio
async def test_ollama_chat_async_basic(basic_engine):
    """Test async chat. This also exercises the concurrency/rate limiters implicitly."""
    messages = [{"role": "user", "content": "What is 2+2?"}]
    response = await basic_engine.chat_async(messages)
    
    assert isinstance(response, dict)
    assert "response" in response
    assert "4" in response["response"] or "four" in response["response"].lower()
    print(f"\n[Async] Response: {response['response']}")

@pytest.mark.asyncio
async def test_ollama_with_reasoning_config():
    """
    Test using ReasoningLLMConfig with a standard model. 
    Even if the model doesn't output <think> tags, the regex should handle it gracefully 
    (empty reasoning, full response).
    """
    # We use a config that expects specific tags
    config = ReasoningLLMConfig(thinking_token_start="<think>", thinking_token_end="</think>")
    engine = OllamaInferenceEngine(model_name=REASON_MODEL_NAME, config=config)
    
    messages = [{"role": "user", "content": "Explain gravity briefly."}]
    response = await engine.chat_async(messages)
    
    assert "reasoning" in response
    assert "response" in response
    # Likely empty reasoning if using standard Llama3, but shouldn't crash
    print(f"\n[ReasoningConfig] Reasoning: {response['reasoning']}")
    print(f"\n[ReasoningConfig] Response: {response['response']}")
