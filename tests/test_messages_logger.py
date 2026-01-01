import pytest
import sys
from unittest.mock import MagicMock, patch
from llm_inference_engine import OllamaInferenceEngine, OpenAIInferenceEngine, BasicLLMConfig
from llm_inference_engine.utils import MessagesLogger

class TestMessagesLogger:
    """
    Tests for MessagesLogger and its integration with Inference Engines.
    Note: The logic for 'store_images' (replacing images with placeholders) 
    is implemented in the Engines' chat methods, not the Logger itself.
    """

    @pytest.fixture(autouse=True)
    def mock_dependencies(self):
        """
        Mock external dependencies (ollama, openai) in sys.modules to ensure 
        tests run without needing the actual packages installed.
        """
        mock_ollama = MagicMock()
        mock_openai = MagicMock()
        
        # Setup mocks for Client/AsyncClient/OpenAI/AsyncOpenAI
        mock_ollama.Client = MagicMock()
        mock_ollama.AsyncClient = MagicMock()
        mock_openai.OpenAI = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()

        with patch.dict(sys.modules, {"ollama": mock_ollama, "openai": mock_openai}):
            yield

    def test_logger_initialization(self):
        """Test that the MessagesLogger initializes with the correct flags."""
        logger_true = MessagesLogger(store_images=True)
        assert logger_true.store_images is True
        
        logger_false = MessagesLogger(store_images=False)
        assert logger_false.store_images is False
        assert logger_false.get_messages_log() == []

    @patch("llm_inference_engine.engines.importlib.util.find_spec", return_value=True)
    def test_ollama_logging_placeholder(self, mock_find_spec):
        """
        Test that OllamaInferenceEngine replaces images with '[image]' 
        when store_images is False.
        """
        # Setup Mock Client and Response
        mock_client = sys.modules["ollama"].Client.return_value
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_response.message.thinking = None
        mock_response.done_reason = "stop"
        mock_client.chat.return_value = mock_response

        # Initialize Engine and Logger
        engine = OllamaInferenceEngine(model_name="test-model")
        logger = MessagesLogger(store_images=False)

        # Input message with image
        messages = [{"role": "user", "content": "Analyze this", "images": ["base64_image_data"]}]

        # Run Chat
        engine.chat(messages, messages_logger=logger)

        # Verify Log
        log = logger.get_messages_log()
        assert len(log) == 1
        last_interaction = log[0]
        
        user_msg = next(m for m in last_interaction if m["role"] == "user")
        # Should be replaced
        assert user_msg["images"] == ["[image]"]
        
        # Verify Assistant response logged
        asst_msg = next(m for m in last_interaction if m["role"] == "assistant")
        assert asst_msg["content"] == "Response"

    @patch("llm_inference_engine.engines.importlib.util.find_spec", return_value=True)
    def test_ollama_logging_store_images(self, mock_find_spec):
        """
        Test that OllamaInferenceEngine preserves images 
        when store_images is True.
        """
        mock_client = sys.modules["ollama"].Client.return_value
        mock_response = MagicMock()
        mock_response.message.content = "Response"
        mock_client.chat.return_value = mock_response

        engine = OllamaInferenceEngine(model_name="test-model")
        logger = MessagesLogger(store_images=True)

        messages = [{"role": "user", "content": "Analyze this", "images": ["base64_image_data"]}]

        engine.chat(messages, messages_logger=logger)

        log = logger.get_messages_log()
        user_msg = next(m for m in log[0] if m["role"] == "user")
        # Should NOT be replaced
        assert user_msg["images"] == ["base64_image_data"]

    @patch("llm_inference_engine.engines.importlib.util.find_spec", return_value=True)
    def test_openai_logging_placeholder(self, mock_find_spec):
        """
        Test that OpenAIInferenceEngine replaces image_url with '[image]' 
        when store_images is False.
        """
        mock_client = sys.modules["openai"].OpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].finish_reason = "stop"
        mock_client.chat.completions.create.return_value = mock_response

        engine = OpenAIInferenceEngine(model="gpt-4")
        logger = MessagesLogger(store_images=False)

        messages = [{
            "role": "user", 
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]
        }]

        engine.chat(messages, messages_logger=logger)

        log = logger.get_messages_log()
        user_msg = next(m for m in log[0] if m["role"] == "user")
        
        # Find image content part
        image_part = next(part for part in user_msg["content"] if part["type"] == "image_url")
        # Should be replaced
        assert image_part["image_url"]["url"] == "[image]"

    @patch("llm_inference_engine.engines.importlib.util.find_spec", return_value=True)
    def test_openai_logging_store_images(self, mock_find_spec):
        """
        Test that OpenAIInferenceEngine preserves image_url 
        when store_images is True.
        """
        mock_client = sys.modules["openai"].OpenAI.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response

        engine = OpenAIInferenceEngine(model="gpt-4")
        logger = MessagesLogger(store_images=True)

        messages = [{
            "role": "user", 
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
            ]
        }]

        engine.chat(messages, messages_logger=logger)

        log = logger.get_messages_log()
        user_msg = next(m for m in log[0] if m["role"] == "user")
        
        image_part = next(part for part in user_msg["content"] if part["type"] == "image_url")
        # Should NOT be replaced
        assert image_part["image_url"]["url"] == "https://example.com/image.png"