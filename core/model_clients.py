"""
Model clients.
Including OpenAI, OpenAI-Responses, Ollama and Mistral client. And many nodes that based on them.
Cannot support Anthropic because this client cannot be tested on my country.
"""

from dataclasses import dataclass
from typing import Callable, Literal, Optional
from openai import OpenAI

from ollama import Client as Ollama
from mistralai import Mistral


@dataclass
class ClientInfo:
    """
    A dataclass that configure client.
    """

    client: OpenAI | Ollama | Mistral
    client_type: Literal["openai", "openai-responses", "ollama", "mistral"]
    chat_func: Callable
    arguments: dict[str, str | list[dict[str, str]] | float | int]
    message_type: Literal["messages", "input"]


def init_client(
    client_type: Literal["openai", "openai-responses", "ollama", "mistral"],
    base_url: str,
    api_key: str,
    model: str,
) -> ClientInfo:
    """
    Init client and send to next node.
    """
    base_client: Optional[OpenAI, Ollama, Mistral] = None

    # Configure the client.
    match client_type:
        case "openai" | "openai-responses":
            base_client: OpenAI = OpenAI(base_url=base_url, api_key=api_key)
        case "ollama":
            base_client: Ollama = Ollama()
        case "mistral":
            base_client: Mistral = Mistral(api_key=api_key)
        case _:
            raise ValueError("The client is not supported")

    # Configure the function
    match client_type:
        case "openai":
            arguments: dict[str, str | list[dict[str, str]] | float | int] = {
                "model": model,
                "messages": [],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_func = base_client.chat.completions.create

        case "openai-responses":
            arguments: dict[str, str | list[dict[str, str]] | float | int] = {
                "model": model,
                "input": [],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_func = base_client.responses.create

        case "mistral":
            arguments: dict[str, str | list[dict[str, str]] | float | int] = {
                "model": model,
                "messages": [],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_func = base_client.chat.complete

        case "ollama":
            arguments: dict[str, str | list[dict[str, str]] | float | int] = {
                "model": model,
                "messages": [],
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 40,
            }
            chat_func = base_client.chat

        case _:
            raise ValueError("The client is not supported")

    result: ClientInfo = ClientInfo(
        client=base_client,
        client_type=client_type,
        chat_func=chat_func,
        arguments=arguments,
        message_type="input" if "responses" in client_type else "messages",
    )

    return result
