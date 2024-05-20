from typing import Any, List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal


class CompletionLogprobs(TypedDict):
    text_offset: List[int]
    token_logprobs: List[Optional[float]]
    tokens: List[str]
    top_logprobs: List[Optional[Dict[str, float]]]


class CompletionChoice(TypedDict):
    text: str
    index: int
    logprobs: Optional[CompletionLogprobs]
    finish_reason: Optional[Literal["stop", "length"]]


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateCompletionResponse(TypedDict):
    id: str
    object: Literal["text_completion"]
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: NotRequired[CompletionUsage]


CreateCompletionStreamResponse = CreateCompletionResponse
