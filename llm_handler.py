import os
import logging
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handler for LLM text generation using OpenAI-compatible APIs."""
    
    def __init__(self, provider=None, api_key=None, base_url=None):
        """
        Initialize LLM Handler.
        
        Args:
            provider: "ollama" for local Ollama or "openai" for OpenAI API
            api_key: API key for OpenAI (optional for Ollama)
            base_url: Custom base URL for Ollama server
        """
        self.provider = provider or config.LLM_PROVIDER
        
        if self.provider == "ollama":
            self.base_url = base_url or config.OLLAMA_BASE_URL
            self.client = OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # Required but unused by Ollama
            )
            self.model = config.OLLAMA_MODEL
            logger.info(f"Initialized Ollama LLM with model: {self.model}")
        else:
            api_key = api_key or config.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OpenAI API key not provided")
            self.client = OpenAI(api_key=api_key)
            self.model = config.OPENAI_MODEL
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")

    def generate_text(self, prompt, system_prompt=None, max_tokens=None):
        """
        Generate text using the LLM.
        
        Args:
            prompt: User prompt for text generation
            system_prompt: System prompt to set behavior
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text or error message
        """
        try:
            logger.debug(f"Generating text with prompt: {prompt[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt or config.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens or config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            generated_text = response.choices[0].message.content
            logger.debug(f"Successfully generated {len(generated_text)} characters")
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg

if __name__ == "__main__":
    # Test
    # llm = LLMHandler(provider="ollama")
    # print(llm.generate_text("Say hello!"))
    pass
