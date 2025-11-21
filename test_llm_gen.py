from llm_handler import LLMHandler
import sys

def test_generation():
    print("Initializing LLM...")
    llm = LLMHandler(provider="ollama")
    print(f"Model: {llm.model}")
    
    prompt = "Say 'Hello, World!' and nothing else."
    print(f"Prompt: {prompt}")
    
    try:
        response = llm.generate_text(prompt)
        print(f"Response: {response}")
        if response:
            return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    if test_generation():
        print("SUCCESS")
    else:
        print("FAILURE")
