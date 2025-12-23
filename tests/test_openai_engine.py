import pytest
import os
from llm_inference_engine import OpenAIInferenceEngine, BasicLLMConfig, OpenAIReasoningLLMConfig

# --- CONFIGURATION ---
MODEL_GPT4 = "gpt-4.1-mini" 
MODEL_REASONING = "o4-mini"

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAIEngine:
    
    @pytest.fixture
    def gpt_engine(self):
        config = BasicLLMConfig(max_new_tokens=1024, temperature=0.5)
        return OpenAIInferenceEngine(model=MODEL_GPT4, config=config)

    def test_openai_chat_sync(self, gpt_engine):
        messages = [{"role": "user", "content": "Hello OpenAI"}]
        response = gpt_engine.chat(messages)
        print(f"\n[Sync] Response: {response['response']}")
        assert len(response["response"]) > 0

    def test_openai_chat_stream(self, gpt_engine):
        messages = [{"role": "user", "content": "List 3 colors"}]
        stream = gpt_engine.chat(messages, stream=True)
        
        full_text = ""
        for chunk in stream:
            full_text += chunk["data"]
        
        print(f"\n[Stream] Full Text: {full_text}")
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_openai_chat_async(self, gpt_engine):
        messages = [{"role": "user", "content": "Async test"}]
        response = await gpt_engine.chat_async(messages)
        print(f"\n[Async] Response: {response['response']}")
        assert len(response["response"]) > 0

    @pytest.mark.asyncio
    async def test_openai_reasoning_model(self):
        """Test the o4 specific configuration."""
        # Note: o4 models do not support 'max_new_tokens' via standard params usually,
        # but OpenAIReasoningLLMConfig handles reasoning_effort.
        config = OpenAIReasoningLLMConfig(reasoning_effort="low")
        engine = OpenAIInferenceEngine(model=MODEL_REASONING, config=config)
        
        # o4 models usually don't support system prompt in the standard way, 
        # the config preprocessor handles concatenating it to user prompt.
        messages = [
            {"role": "system", "content": "You are a math helper."},
            {"role": "user", "content": "Solve 2x=4"}
        ]
        
        try:
            response = await engine.chat_async(messages)
            print(f"\n[Reasoning Model] Response: {response['response']}")
            assert "x = 2" in response["response"] or "2" in response["response"]
        except Exception as e:
            pytest.fail(f"OpenAI Reasoning model failed: {e}")