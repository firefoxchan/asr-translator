import dataclasses
import pathlib
from dataclasses import dataclass

from llm_types import *


@dataclass
class SakuraConfig:
    model_name_or_path: str

    use_gpu: bool = True
    text_length: int = 1024

    model_name: str = None
    model_version: str = None
    model_quant: str = None


@dataclass
class SakuraGenerationConfig:
    max_new_tokens: int = None
    temperature: float = None
    top_p: float = None
    min_p: float = None
    frequency_penalty: float = None
    presence_penalty: float = None
    repeat_penalty: float = None
    top_k: int = None
    stream: bool = None
    seed: int = None

    def load_sakura_config(self, sc: SakuraConfig):
        if self.max_new_tokens is None:
            self.max_new_tokens = 2*sc.text_length

    def llama_cpp(self):
        d = dataclasses.asdict(self, dict_factory=lambda x: {k: v for (k, v) in x if v is not None})
        d["max_tokens"] = d["max_new_tokens"]
        del d["max_new_tokens"]
        return d

    def asdict(self):
        return self.llama_cpp()


@dataclass
class SakuraCompletionResponse:
    text: str
    finish_reason: Literal["stop", "length"]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Sakura:
    def __init__(self, cfg: SakuraConfig):
        # init cfg
        filename = pathlib.Path(cfg.model_name_or_path).stem
        model_name, model_version, model_quant = filename.rsplit('-', 2)
        if cfg.model_name is None:
            cfg.model_name = model_name
        if cfg.model_version is None:
            cfg.model_version = model_version
        if cfg.model_quant is None:
            cfg.model_quant = model_quant
        self.cfg = cfg
        # load model
        self._load_model(cfg)

    def _load_model(self, cfg: SakuraConfig):
        self._load_llama_cpp(cfg)

    def _load_llama_cpp(self, cfg: SakuraConfig):
        import llama_cpp
        llama_config = {
            "n_ctx": 4*cfg.text_length,
        }
        if cfg.use_gpu:
            llama_config["n_gpu_layers"] = -1
            llama_config["offload_kqv"] = True
        else:
            llama_config["n_gpu_layers"] = 0
            llama_config["offload_kqv"] = False
        self._model = llama_cpp.Llama(
            cfg.model_name_or_path,
            **llama_config,
        )
        self._tokenizer = llama_cpp.LlamaTokenizer(self._model)

    def completion(self, prompt: str, cfg: SakuraGenerationConfig) -> SakuraCompletionResponse:
        cfg.load_sakura_config(self.cfg)
        cfg.stream = False
        resp: Optional[CreateCompletionResponse] = None
        for i in range(2):
            resp: CreateCompletionResponse = self._model(prompt, **cfg.asdict())
            if len(resp["choices"]) == 0:
                if i == 1:
                    cfg.temperature = 1.0
                    cfg.top_p = 1.0
                continue
            ret = SakuraCompletionResponse(
                text=resp["choices"][0]["text"],
                finish_reason="stop",
                **resp["usage"],
            )
            if resp["choices"][0]["finish_reason"]:
                ret.finish_reason = resp["choices"][0]["finish_reason"]
            if resp["usage"]:
                ret.prompt_tokens = resp["usage"]["prompt_tokens"]
                ret.completion_tokens = resp["usage"]["completion_tokens"]
                ret.total_tokens = resp["usage"]["total_tokens"]
            return ret
        ret = SakuraCompletionResponse(
            text="",
            finish_reason="stop",
        )
        if resp and resp["usage"]:
            ret.prompt_tokens = resp["usage"]["prompt_tokens"]
            ret.completion_tokens = resp["usage"]["completion_tokens"]
            ret.total_tokens = resp["usage"]["total_tokens"]
        return ret

