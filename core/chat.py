"""
General chat utility functions.
Function 1: Basic chat completion.
Function 2: Image grounding.
"""

import re
from logging import getLogger

from dataclasses import dataclass
from typing import Optional, Literal, Tuple
from PIL import Image, ImageDraw
from openai import OpenAIError

from .model_clients import ClientInfo, DetailedArguments
from .utils import pil_to_base64

Logger = getLogger(__name__)


@dataclass
class ExtraParameters:
    """
    Extra parameters for the chat completion.
    Useful for some models with CoT.
    """

    thinking: Optional[Literal["disabled", "enabled"]] = (
        None  # "thinking" : {"type": "disabled"} for doubao models in volcengine.
    )
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = (
        None  # "reasoning_effort" : "medium" for doubao-seed-1-6-lite-251015 or gpt-5
    )


def chat_completion(
    client_info: ClientInfo,
    system_prompt: str,
    user_prompt: str,
    images: list[Image.Image] | None = None,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 40,
    max_tokens: int = 1024,
    unload_after_chat: bool = True,
    extra_parameters: Optional[ExtraParameters] = None,
) -> str:
    """
    Basic chat completion.
    """
    # Basic check of the user's input.
    if images is None:
        payload_images: list = []
    elif len(images) > 4:
        Logger.warning(
            "Warning: The number of images is greater than 4. Only the first 4 images will be used."
        )
        payload_images: list = images[:4]
    else:
        payload_images: list = images

    base64_images: list[str] = [pil_to_base64(image) for image in payload_images]
    orig_args: DetailedArguments = client_info.arguments
    messages: list[dict] = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                *[
                    {"type": "image_url", "image_url": {"url": base64_image}}
                    for base64_image in base64_images
                ],
            ],
        },
    ]
    # prepare the arguments
    # Due to ollama issue (Cannot unload model using openai chat api.). We have to generate a single payload here.
    if client_info.client_type == "ollama":
        # ACPCat5173: WHY OLLAMA HAVE TO WRITE A SPECIAL FORMAT ON THEIR OWN??? I WILL NOT USE IT IN THE FUTURE ANYMORE.
        # WHY NOT USE OPENAI CHAT COMPTITABLE API? BECAUSE WE CANNOT UNLOAD MODEL ON CHAT COMPTITABLE API!
        payload = {
            "model": orig_args.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [base64_image for base64_image in base64_images],
                },
            ],
            "options": {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "num_predict": max_tokens,
            },
        }
        if extra_parameters:
            # "minimal" thinking effect is not supported by ollama.
            # but "minimal" thinking effect belong to no think is volcengine.
            need_think = extra_parameters.thinking == "enabled"
            payload["think"] = need_think
            if need_think and extra_parameters.reasoning_effort != "minimal":
                payload["think"] = extra_parameters.reasoning_effort
        if unload_after_chat:
            payload["keep_alive"] = "0"

    elif client_info.client_type == "openai":
        payload = {
            "model": orig_args.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if extra_parameters:
            if extra_parameters.thinking == "enabled":
                if "seed-1-6" in orig_args.model:
                    payload["extra_body"] = {}
                    payload["extra_body"]["thinking"] = {
                        "type": extra_parameters.thinking
                    }
                payload["reasoning_effort"] = extra_parameters.reasoning_effort
    elif client_info.client_type == "openai-responses":
        payload = {
            "model": orig_args.model,
            "input": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": user_prompt},
                        *[
                            {"type": "input_image", "image_url": base64_image}
                            for base64_image in base64_images
                        ],
                    ],
                },
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
        }
        if extra_parameters:
            model: str = orig_args.model
            if "seed-1-6" in model:
                payload["extra_body"] = {}
                payload["extra_body"]["thinking"] = {"type": extra_parameters.thinking}
            else:
                # volcengine does not support thinking effort in responses api
                payload["reasoning"] = {}
                payload["reasoning"]["effort"] = extra_parameters.reasoning_effort

    elif client_info.client_type == "mistral":
        payload = {
            "model": orig_args.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
        }
    else:
        raise ValueError("The client type is not supported. You are using Anthropic?")

    # Call the API
    try:
        response = client_info.chat_func(**payload)
    except OpenAIError as e:
        raise RuntimeError(f"Error in openai chat completion: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error in chat completion: {e}") from e

    # Parse the response
    if client_info.client_type in ["openai", "mistral"]:  # Chat Completion API
        result: str = response.choices[0].message.content
    elif client_info.client_type == "openai-responses":  # Responses API
        result: str = response.output_text
    elif client_info.client_type == "ollama":
        result: str = response.message.content
    else:
        raise ValueError("The message type is not supported. You are using Anthropic?")
    return result


def grounding(
    client_info: ClientInfo,
    item: Optional[str],
    image: Image.Image,
    thinking: Literal["disabled", "enabled"] = "disabled",
    mode: Literal["minimal", "low", "medium", "high"] = "minimal",
    unload_after_chat: bool = True,
) -> Tuple[list[list[int]], Image.Image]:
    """
    Image grounding with item.
    """
    # Basic check of the user's input.
    if item is None:
        raise ValueError("The item is None")

    # prepare the prompt
    system_prompt = "You are a professional image grounding assistant."
    user_prompt = (
        f"Please locate the item '{item}' in the image accurately. Response in coordinate of the bounding box. "
        + "The format is <bbox>x_min y_min x_max y_max</bbox> in percentage(0-1000). "
        + "If there are multiple items, please list all bounding boxes. "
    )

    # prepare the extra parameters
    extra_parameters = ExtraParameters(thinking=thinking, reasoning_effort=mode)

    # call the API
    response: str = chat_completion(
        client_info=client_info,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        images=[image],
        temperature=1.0,
        top_p=0.7,
        max_tokens=4096,
        unload_after_chat=unload_after_chat,
        extra_parameters=extra_parameters,
    )

    # responses: <bbox>0 55 63 250</bbox><bbox>0 300 51 427</bbox>...
    bboxes: list[list[int]] = []
    for bbox in re.findall(r"<bbox>(.*?)</bbox>", response):
        # seed 1.6
        bbox = [int(x) for x in bbox.split()]
        if len(bbox) != 4:
            raise ValueError("BBox is invaild. Please retry again.")
        x_min, y_min, x_max, y_max = tuple(bbox)
        w, h = image.size
        x_min_real = int(x_min * w / 1000)
        y_min_real = int(y_min * h / 1000)
        x_max_real = int(x_max * w / 1000)
        y_max_real = int(y_max * h / 1000)

        bbox_x = x_min_real
        bbox_y = y_min_real
        bbox_w = x_max_real - x_min_real
        bbox_h = y_max_real - y_min_real

        bboxes.append([bbox_x, bbox_y, bbox_w, bbox_h])

    # Create a copy of the image to draw on
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)

    # Define bbox style
    bbox_color = "red"  # Red color for bboxes
    bbox_width = 2  # Line width

    # Draw each bbox
    for bbox in bboxes:
        if len(bbox) == 4:  # Ensure bbox has 4 coordinates [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            # Draw rectangle
            # Pillow uses [x1, x2, x3, x4]
            draw.rectangle(
                [x1, y1, x1 + x2, y1 + y2], outline=bbox_color, width=bbox_width
            )

    return bboxes, draw_image
