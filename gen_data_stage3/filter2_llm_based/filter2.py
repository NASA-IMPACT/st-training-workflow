import os
import sys
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider


class Quality(Enum):
    """Quality of the content."""

    VERY_GOOD = "VERY_GOOD"
    GOOD = "GOOD"
    POOR = "POOR"


class ContentQuality(BaseModel):
    """
    The quality of the text content
    for information retrieval.
    """

    quality: Quality = Field(
        ...,
        description="The quality of the content",
    )
    reasoning_traces: list[str] = Field(
        ...,
        description="Concise and accurate reasoning traces for assesing the quality of the content.",
    )


def main():
    # quality = ContentQuality(quality=Quality.VERY_GOOD, reasoning_traces=[])

    ollama_model = OpenAIModel(
        model_name="gemma3:4b",
        provider=OpenAIProvider(base_url="http://localhost:11434/v1"),
    )
    agent = Agent(
        ollama_model,
        output_type=ContentQuality,
        system_prompt="You are an expert in assessing the quality of scientific content",
    )
    result = agent.run_sync("I am paradox. I exist only in my own mind")
    print(result)


if __name__ == "__main__":
    main()
