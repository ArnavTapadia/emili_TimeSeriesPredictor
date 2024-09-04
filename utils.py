import openai
import requests
import os
import json
import anthropic
from anthropic.types import Message, TextBlock
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

openai.api_key = os.environ["OPENAI_API_KEY"]
anthropic.api_key = os.environ.get("ANTHROPIC_API_KEY")
client = openai.OpenAI()

# Most models return a response object: To recover the generated text, use 
#   response.choices[0].message.content
#
# But gpt-4-vision-preview returns a "requests" object: To recover the generated text, use
#   (response.json())['choices'][0]['message']['content']

default_response_object = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1677858242,
    "model": "gpt-3.5-turbo-0613",
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    },
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": "Brain disconnect, sorry mate."
            },
            "logprobs": None,
            "finish_reason": "OpenAI API error",
            "index": 0
        }
    ]
}

class JSONToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, JSONToObject(value) if isinstance(value, dict) else value)

def json_to_object(data):
    return json.loads(data, object_hook=JSONToObject)

@retry(wait=wait_exponential(multiplier=1.5, min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def get_OAI_api_response(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
    try:
        full_response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens
        )
        if return_full_response:
            return full_response # the full response object
        else:
            return full_response.choices[0].message.content # just the generated text
    except Exception as e:
        print(f"(API error: {e}, retrying...)")
        raise e



def get_Claude_response(messages, model="claude-3-sonnet-20240229", temperature=1.0, max_tokens=1024, return_full_response=False):
    try:
        client = anthropic.Anthropic()

        system_messages = []
        api_messages = []

        print("Debug: Processing input messages:")
        for idx, message in enumerate(messages):
            role = message['role']
            content = message['content']

            if role == 'system':
                if isinstance(content, str):
                    system_messages.append(content)
                    #print(f"Debug: Found text-based system message {idx+1}: {content[:50]}...")
                else:
                    print(f"Debug: Skipping non-text system message {idx+1}: {type(content)}")
            else:
                api_messages.append({"role": role, "content": content})
                #print(f"Debug: Added {role} message {idx+1} to API messages")

        if model == "gpt-4-vision-preview": model = "claude-3-sonnet-20240229"
        kwargs = {
            "model": model,
            "messages": api_messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        if system_messages:
            kwargs["system"] = "\n".join(system_messages)
            print(f"\nDebug: Using all system messages for API call. Total system message length: {len(kwargs['system'])}")
        else:
            print("\nDebug: No text-based system messages to send to API")

        print("\nDebug: Preparing to make API call with the following parameters:")
        print(f"  Model: {kwargs['model']}")
        print(f"  Max tokens: {kwargs['max_tokens']}")
        print(f"  Temperature: {kwargs['temperature']}")
        print(f"  System message: {'Present' if 'system' in kwargs else 'Not present'}")
        print(f"  Number of messages: {len(kwargs['messages'])}")

        print("\nDebug: Making API call now...")
        response: Message = client.messages.create(**kwargs)
        print("Debug: API call completed")

        if return_full_response:
            return response
        else:
            # Extract text from the response
            text_content = ""
            for content in response.content:
                if isinstance(content, TextBlock):
                    text_content += content.text
            return text_content.strip()

    except Exception as e:
        error_message = f"Error in Claude API call: {str(e)}"
        if return_full_response:
            return {"error": error_message}
        else:
            return error_message


def get_OAI_response(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
    try:
        if model != "gpt-4-vision-preview":
            return get_OAI_text_response(messages, model, temperature, max_tokens, seed, return_full_response)
        else:
            return get_api_vision_response(messages, model, temperature, max_tokens, seed, return_full_response)
    except Exception: # all retries failed
        if(return_full_response):
            return default_response_object # todo: improve this!
        else:
            return "Brain disconnect, sorry mate."
        


def get_OAI_text_response(messages, model="gpt-3.5-turbo", temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):
    try:
        full_response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=messages,
            max_tokens=max_tokens
        )
        if return_full_response:
            return full_response # the full response object
        else:
            return full_response.choices[0].message.content # just the generated text
    except Exception as e:
        print(f"(API error: {e}, retrying...)")
        raise e


headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {openai.api_key}"
}
endpoint_url = "https://api.openai.com/v1/chat/completions"

@retry(wait=wait_exponential(multiplier=1.5, min=1, max=60), stop=stop_after_attempt(6), retry=retry_if_exception_type(Exception))
def get_api_vision_response(messages, model="gpt-4-vision-preview", temperature=1.0, max_tokens=64, seed=1331, return_full_response=False):

    try:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens
        }

        api_response = requests.post(endpoint_url, headers=headers, json=payload)
        json_response = api_response.json()

        if return_full_response:
            return json_response 
        else:
            return json_response['choices'][0]['message']['content'] # just the generated text
        
    except Exception as e:
        print(f"(API error: {e}, retrying...)")
        raise e
