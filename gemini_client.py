"""
Gemini client for the AI Product Expert Bot.
"""

import logging
import json
from typing import Dict, List, Optional, Union, Any, Callable

import google.generativeai as genai
# Import safety settings without specific classes that might have changed
from google.generativeai.types import SafetySettingDict

# Import config
try:
    import config
except ModuleNotFoundError:
    print("ERROR: config.py not found. Please ensure it's in the same directory or PYTHONPATH is set.")
    # Fallback for critical configs if needed
    config = type('obj', (object,), {
        'GOOGLE_CLOUD_PROJECT': 'your-gcp-project-id-fallback',
        'GOOGLE_CLOUD_REGION': 'us-central1',
    })

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add API key to config
if not hasattr(config, 'GEMINI_API_KEY'):
    # Add this to config.py manually with your actual API key
    logger.warning("GEMINI_API_KEY not found in config. You'll need to add this to config.py.")
    config.GEMINI_API_KEY = None

# Initialize Gemini
try:
    # Configure with API key instead of project
    genai.configure(api_key=config.GEMINI_API_KEY)
    logger.info("Gemini initialized with API key")
except Exception as e:
    logger.error(f"Failed to initialize Gemini: {e}")

# Default safety settings - adjust as needed
# Using the dictionary format that's compatible with the current version
DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
]

# Cache for models
_gemini_model = None

def get_gemini_model(model_name: str = "gemini-2.5-flash-preview-05-20") -> Any:
    """
    Returns a Gemini model instance.
    
    Args:
        model_name: Name of the Gemini model to use.
        
    Returns:
        The Gemini model instance.
    """
    global _gemini_model
    if _gemini_model is None:
        try:
            logger.info(f"Loading Gemini model: {model_name}")
            _gemini_model = genai.GenerativeModel(
                model_name=model_name,
                safety_settings=DEFAULT_SAFETY_SETTINGS,
                generation_config={
                    "temperature": 0.2,  # Lower temperature for more deterministic outputs
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                }
            )
            logger.info(f"Gemini model {model_name} loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Gemini model: {e}")
            _gemini_model = None
    return _gemini_model

def generate_text(
    prompt: str,
    system_instruction: Optional[str] = None,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Generates text using the Gemini model.
    
    Args:
        prompt: The prompt to generate text from.
        system_instruction: Optional system instruction to guide the model.
        temperature: Optional temperature for generation.
        max_output_tokens: Optional maximum number of tokens to generate.
        chat_history: Optional chat history for multi-turn conversations.
        
    Returns:
        The generated text.
    """
    model = get_gemini_model()
    if not model:
        logger.error("Gemini model is not available.")
        return "Error: Gemini model is not available."
    
    try:
        # Create generation config with any overrides
        generation_config = {}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if max_output_tokens is not None:
            generation_config["max_output_tokens"] = max_output_tokens
        
        # Handle chat history if provided
        if chat_history:
            chat = model.start_chat(history=chat_history)
            response = chat.send_message(
                prompt,
                generation_config=generation_config if generation_config else None
            )
        else:
            # For single-turn generation
            if system_instruction:
                content = [
                    {"role": "system", "parts": [system_instruction]},
                    {"role": "user", "parts": [prompt]}
                ]
                response = model.generate_content(
                    content,
                    generation_config=generation_config if generation_config else None
                )
            else:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config if generation_config else None
                )
        
        logger.info("Text generated successfully.")
        return response.text
    
    except Exception as e:
        logger.error(f"Failed to generate text: {e}")
        return f"Error generating text: {str(e)}"

def function_calling(
    prompt: str,
    functions: List[Dict[str, Any]],
    function_executors: Dict[str, Callable],
    system_instruction: Optional[str] = None,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> str:
    """
    Uses Gemini's function calling capability to execute functions based on user input.
    
    Args:
        prompt: The user prompt.
        functions: List of function definitions.
        function_executors: Dictionary mapping function names to their executor functions.
        system_instruction: Optional system instruction to guide the model.
        chat_history: Optional chat history for multi-turn conversations.
        
    Returns:
        The final response after function execution.
    """
    model = get_gemini_model()
    if not model:
        logger.error("Gemini model is not available.")
        return "Error: Gemini model is not available."
    
    try:
        # Configure the model with the functions
        model_with_tools = genai.GenerativeModel(
            model_name=model._model_name,
            tools=functions,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
            generation_config=model._generation_config
        )
        
        # Prepare the content
        if system_instruction:
            content = [
                {"role": "system", "parts": [system_instruction]},
                {"role": "user", "parts": [prompt]}
            ]
        else:
            content = prompt
        
        # Generate the response
        if chat_history:
            chat = model_with_tools.start_chat(history=chat_history)
            response = chat.send_message(content)
        else:
            response = model_with_tools.generate_content(content)
        
        # Check if the model wants to call a function
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            function_name = function_call.name
                            
                            if function_name in function_executors:
                                # Parse the arguments
                                args = json.loads(function_call.args)
                                
                                # Execute the function
                                logger.info(f"Executing function: {function_name}")
                                function_result = function_executors[function_name](**args)
                                
                                # Send the function result back to the model
                                function_response = {
                                    "name": function_name,
                                    "response": function_result
                                }
                                
                                # Get the final response
                                if chat_history:
                                    final_response = chat.send_message(function_response)
                                else:
                                    final_response = model_with_tools.generate_content([
                                        {"role": "user", "parts": [content]},
                                        {"role": "model", "parts": [{"function_call": function_call}]},
                                        {"role": "function", "parts": [{"function_response": function_response}]}
                                    ])
                                
                                logger.info("Function calling completed successfully.")
                                return final_response.text
        
        # If no function call was made, return the original response
        logger.info("No function call was made.")
        return response.text
    
    except Exception as e:
        logger.error(f"Failed to execute function calling: {e}")
        return f"Error in function calling: {str(e)}"

if __name__ == "__main__":
    # Test the Gemini client
    logger.info("Testing Gemini client...")
    
    # Test text generation
    prompt = "What are the key features to consider when comparing smartphones?"
    system_instruction = "You are a helpful product expert who provides concise, accurate information about consumer products."
    
    logger.info(f"Generating text for prompt: '{prompt}'")
    response = generate_text(prompt, system_instruction=system_instruction)
    print(f"\nGenerated Text:\n{response}\n")
    
    # Test function calling
    # Define a simple function for demonstration
    def search_products(query: str, max_results: int = 5) -> Dict[str, Any]:
        return {
            "results": [
                {"id": "p1", "name": f"Product for {query} - 1", "price": 99.99},
                {"id": "p2", "name": f"Product for {query} - 2", "price": 149.99},
            ],
            "total_results": 2,
            "query": query,
            "max_results": max_results
        }
    
    # Define the function schema
    functions = [
        {
            "name": "search_products",
            "description": "Search for products in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    ]
    
    # Map function names to their implementations
    function_executors = {
        "search_products": search_products
    }
    
    # Test function calling
    function_prompt = "I'm looking for a good smartphone under $200"
    logger.info(f"Testing function calling with prompt: '{function_prompt}'")
    
    try:
        function_response = function_calling(
            function_prompt,
            functions,
            function_executors,
            system_instruction="You are a product search assistant. Use the search_products function to find relevant products."
        )
        print(f"\nFunction Calling Response:\n{function_response}\n")
    except Exception as e:
        logger.error(f"Function calling test failed: {e}")
        print(f"\nFunction calling test failed: {e}\n")
    
    logger.info("Gemini client tests completed.")