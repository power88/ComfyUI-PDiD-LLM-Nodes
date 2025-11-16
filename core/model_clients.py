"""
Model clients.
Including OpenAI, OpenAI-Responses, Ollama and Mistral client. And many nodes that based on them.
Cannot support Anthropic because this client cannot be tested on my country.
"""

from dataclasses import dataclass
from typing import Callable, Literal
from openai import OpenAI

from ollama import Client as Ollama
from mistralai import Mistral


@dataclass
class DetailedArguments:
    """
    Detailed arguments for chat completion.
    """

    model: str
    messages: list[dict[str, str]] | list[str]
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40


@dataclass
class ClientInfo:
    """
    A dataclass that configure client.
    """

    client: OpenAI | Ollama | Mistral
    client_type: Literal["openai", "openai-responses", "ollama", "mistral"]
    chat_func: Callable
    arguments: DetailedArguments


def init_client(
    client_type: Literal["openai", "openai-responses", "ollama", "mistral"],
    base_url: str,
    api_key: str,
    model: str,
) -> ClientInfo:
    """
    Init client and send to next node.
    """
    # Configure the client.
    if client_type in ["openai", "openai-responses"]:
        base_client = OpenAI(base_url=base_url, api_key=api_key)
    elif client_type == "ollama":
        base_client = Ollama()
    elif client_type == "mistral":
        base_client = Mistral(api_key=api_key)
    else:
        raise ValueError("The client is not supported")

    arguments: DetailedArguments = DetailedArguments(
        model=model,
        messages=[],
        temperature=1.0,
        top_p=0.95,
        top_k=40,
    )
    # Configure the function
    if client_type == "openai":
        chat_func: Callable = base_client.chat.completions.create

    elif client_type == "openai-responses":
        chat_func: Callable = base_client.responses.create

    elif client_type == "mistral":
        chat_func: Callable = base_client.chat.complete

    elif client_type == "ollama":
        chat_func: Callable = base_client.chat

    else:
        raise ValueError("The client is not supported")

    result: ClientInfo = ClientInfo(
        client=base_client,
        client_type=client_type,
        chat_func=chat_func,
        arguments=arguments,
    )

    return result
