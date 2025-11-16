"""
ComfyUI nodes for the LLM.
Converted to V3 schema
"""

from typing import Literal, Optional
from torch import Tensor
from PIL import Image
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io
from .core.chat import chat_completion, grounding, captioning, ExtraParameters
from .core.utils import tensor_to_pil, pil_to_tensor
from .core.model_clients import init_client, ClientInfo


class LLMLoader(io.ComfyNode):
    """
    Load the LLM client.

    Class methods
    -------------
    define_schema (io.Schema):
        Tell the main program the metadata, input, output parameters of nodes.
    execute:
        Load the LLM client.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Tell the main program the metadata, input, output parameters of nodes.
        """
        return io.Schema(
            node_id="APILLMLoader",
            display_name="API LLM Loader",
            category="LLM",
            inputs=[
                io.String.Input(
                    "base_url",
                    multiline=False,
                    default="https://api.openai.com/v1",
                    tooltip="The API URL of the provider.",
                ),
                io.String.Input(
                    "api_key",
                    multiline=False,
                    default="sk-1234567890",
                    tooltip="The API key of the provider.",
                ),
                io.String.Input(
                    "model_name",
                    multiline=False,
                    default="gpt-5-high",
                    tooltip="The name of the model.",
                ),
                io.Combo.Input(
                    "client_type",
                    options=["openai", "openai-responses", "mistral", "ollama"],
                    default="openai",
                    tooltip="The type of the LLM client (OpenAI chat comptitable? or Mistral?).",
                    optional=True,
                ),
            ],
            outputs=[
                io.Custom("CLIENT_INFO").Output(display_name="client_info"),
            ],
        )

    @classmethod
    def execute(
        cls,
        base_url: str,
        api_key: str,
        model_name: str,
        client_type: Literal["openai", "openai-responses", "mistral", "ollama"],
    ) -> io.NodeOutput:
        """
        Load the LLM client.
        """
        client_info: ClientInfo = init_client(
            client_type=client_type,
            base_url=base_url,
            api_key=api_key,
            model=model_name,
        )
        return io.NodeOutput(client_info)


class ExtraParametersComfy(io.ComfyNode):
    """
    Extra parameters for the chat completion.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Extra parameters for the chat completion.
        """
        return io.Schema(
            node_id="ExtraParameters",
            display_name="Extra Parameters",
            category="LLM",
            inputs=[
                io.Combo.Input(
                    "thinking",
                    options=["disabled", "enabled"],
                    default="disabled",
                    tooltip="Whether to enable the thinking mode for the LLM model.",
                ),
                io.Combo.Input(
                    "reasoning_effort",
                    options=["minimal", "low", "medium", "high"],
                    default="medium",
                    tooltip="The reasoning effort for the LLM model.",
                ),
            ],
            outputs=[
                io.Custom("EXTRA_PARAMETERS").Output(display_name="extra_parameters"),
            ],
        )

    @classmethod
    def execute(
        cls,
        thinking: Literal["disabled", "enabled"],
        reasoning_effort: Literal["minimal", "low", "medium", "high"],
    ) -> io.NodeOutput:
        """
        Extra parameters for the chat completion.
        """
        return io.NodeOutput(
            ExtraParameters(thinking=thinking, reasoning_effort=reasoning_effort)
        )


class ChatViaAPI(io.ComfyNode):
    """
    Chat with the LLM model.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Tell the main program the metadata, input, output parameters of nodes.
        """
        return io.Schema(
            node_id="ChatViaAPI",
            display_name="API Chat",
            category="LLM",
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images for the LLM model.",
                    optional=True,
                ),
                io.Custom("CLIENT_INFO").Input(
                    "client_info",
                    tooltip="The LLM client info.",
                ),
                io.String.Input(
                    "system_prompt",
                    multiline=True,
                    default="You are a helpful assistant.",
                    tooltip="The system prompt for the LLM model.",
                ),
                io.String.Input(
                    "user_prompt",
                    multiline=True,
                    default="Hello world!",
                    tooltip="The system prompt for the LLM model.",
                ),
                io.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="The temperature parameter for the LLM model.",
                ),
                io.Float.Input(
                    "top_p",
                    default=0.95,
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    tooltip="The top_p parameter for the LLM model.",
                ),
                io.Int.Input(
                    "top_k",
                    default=40,
                    min=1,
                    max=99,
                    step=1,
                    tooltip="The top_k parameter for the LLM model. Only Ollama is supported.",
                ),
                io.Int.Input(
                    "max_tokens",
                    default=1024,
                    min=1,
                    max=1_000_000,
                    step=1,
                    tooltip="The max_tokens for the LLM model.",
                ),
                io.Boolean.Input(
                    "unload_model_after_chat",
                    default=True,
                    tooltip=(
                        "Whether to unload the LLM model after the chat. "
                        + "Only Ollama is supported."
                    ),
                ),
                io.Custom("EXTRA_PARAMETERS").Input(
                    "extra_parameters",
                    tooltip="The extra parameters for the LLM model.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="response"),
            ],
        )

    @classmethod
    def execute(
        cls,
        client_info: ClientInfo,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        unload_model_after_chat: bool,
        extra_parameters: Optional[ExtraParameters] = None,
        images: Optional[Tensor] = None,
    ) -> io.NodeOutput:
        """
        Chat with the LLM model.
        """
        if images is not None and images.any():
            pil_images: Optional[list[Image.Image]] = []
            for image in images:
                pil_images.append(tensor_to_pil(image))

        response: str = chat_completion(
            client_info=client_info,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=pil_images,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            unload_after_chat=unload_model_after_chat,
            extra_parameters=extra_parameters,
        )
        return io.NodeOutput(response)


class GenerateBBOX(io.ComfyNode):
    """
    Generate bboxes based on LLM's grounding abitility.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Tell the main program the metadata, input, output parameters of nodes.
        """
        return io.Schema(
            node_id="GenerateBBOXViaAPI",
            display_name="Generate BBOXes",
            category="LLM",
            inputs=[
                io.Custom("CLIENT_INFO").Input(
                    "client_info",
                    tooltip="The LLM client info.",
                ),
                io.String.Input(
                    "items",
                    default="dog",
                    tooltip="The system prompt for the LLM model.",
                ),
                io.Boolean.Input(
                    "unload_model_after_chat",
                    default=True,
                    tooltip=(
                        "Whether to unload the LLM model after the chat. "
                        + "Only Ollama is supported."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip="The image for the LLM model.",
                ),
            ],
            outputs=[
                io.Custom("BBOX").Output(display_name="bbox"),
                io.Image.Output(display_name="BBoxPreviewImage"),
            ],
        )

    @classmethod
    def execute(
        cls,
        client_info: ClientInfo,
        items: str,
        unload_model_after_chat: bool,
        image: Tensor,
    ) -> io.NodeOutput:
        """
        Generate bboxes based on LLM's grounding abitility.
        """
        pil_image: Image.Image = tensor_to_pil(image)

        (response, bbox_image) = grounding(
            client_info=client_info,
            item=items,
            image=pil_image,
            unload_after_chat=unload_model_after_chat,
        )
        # Convert bbox images to Tensor
        result_image: Tensor = pil_to_tensor(bbox_image)
        return io.NodeOutput(response, result_image)


class Capotion(io.ComfyNode):
    """
    Generate bboxes based on LLM's grounding abitility.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        """
        Tell the main program the metadata, input, output parameters of nodes.
        """
        return io.Schema(
            node_id="GenerateBBOXViaAPI",
            display_name="Generate BBOXes",
            category="LLM",
            inputs=[
                io.Custom("CLIENT_INFO").Input(
                    "client_info",
                    tooltip="The LLM client info.",
                ),
                io.String.Input(
                    "language",
                    default="English",
                    tooltip="The language of the caption.",
                ),
                io.Boolean.Input(
                    "unload_model_after_chat",
                    default=True,
                    tooltip=(
                        "Whether to unload the LLM model after the chat. "
                        + "Only Ollama is supported."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip="The image for the LLM model.",
                ),
                io.Custom("EXTRA_PARAMETERS").Input(
                    "extra_parameters",
                    tooltip="The extra parameters for the LLM model.",
                    optional=True,
                ),
            ],
            outputs=[
                io.String.Output(display_name="Caption"),
            ],
        )

    @classmethod
    def execute(
        cls,
        client_info: ClientInfo,
        language: str,
        unload_model_after_chat: bool,
        image: Tensor,
        extra_parameters: Optional[ExtraParameters] = None,
    ) -> io.NodeOutput:
        """
        Generate bboxes based on LLM's grounding abitility.
        """
        pil_image: Image.Image = tensor_to_pil(image)

        payload = {
            "client_info": client_info,
            "language": language,
            "image": pil_image,
            "unload_after_chat": unload_model_after_chat,
        }

        if extra_parameters is not None:
            payload["thinking"] = extra_parameters.thinking
            payload["mode"] = extra_parameters.reasoning_effort

        response = captioning(**payload)

        return io.NodeOutput(response)


class PDIDLLMNodes(ComfyExtension):
    """
    Define ComfyUI nodes.
    """

    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:  # type: ignore[override]
        """
        Get the list of nodes.
        """
        return [LLMLoader, ExtraParametersComfy, ChatViaAPI, GenerateBBOX]


async def comfy_entrypoint() -> ComfyExtension:
    """
    Register ComfyUI nodes.
    """
    return PDIDLLMNodes()
