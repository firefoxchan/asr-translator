import copy
import collections
from pprint import pprint

import dicts
import subs
import llm


class Progress:
    def __init__(self, current: float, total: float, desc: str, data: any, finish: bool):
        self.current = float(current)
        self.total = float(total)
        self.desc = desc
        self.data = data
        self.finish = finish


class SakuraLLMTranslator:
    LINE_BREAK = "\n"

    def __init__(
            self,
            cfg: llm.SakuraConfig,
            gc: llm.SakuraGenerationConfig,
            show_progress=False,
            max_source_lines=30,
    ):
        self.model = llm.Sakura(cfg)
        self.generation_config = gc
        self.gpt_dict = dicts.gpt_dict
        self.history = collections.deque([])
        self.history_length = 0
        self.show_progress = show_progress
        self.max_source_lines = max_source_lines

    def history_append(self, src: str, trs: str):
        self.history.append((src, trs))
        self.history_length += len(src) + len(trs) + 2
        while self.history_length > self.model.cfg.text_length:
            l_src, l_trs = self.history.popleft()
            self.history_length -= (len(l_src) + len(l_trs) + 2)
        if self.show_progress:
            print(f"trs from: {src}")
            print(f"      to: {trs}")

    def translate_file(self, file):
        sub = subs.Sub.load_file(file)
        return self.translate(sub)

    def translate(self, sub: subs.Sub):
        sub = [event.clean_ja() for event in sub]
        grouped: [[subs.SubEvent]] = []
        current: [subs.SubEvent] = []
        current_chars: int = 0
        for line in sub:
            if current_chars + len(line.text) > self.model.cfg.text_length or len(current) >= self.max_source_lines:
                grouped.append(current)
                current, current_chars = [], 0
            current.append(line)
            current_chars += len(line.text)
        if len(current) > 0:
            grouped.append(current)
            current, current_chars = [], 0
        translated: [subs.SubEvent] = []
        yield Progress(len(translated), len(sub), '', translated, False)
        for current in grouped:
            non_empty = list(line.text for line in current if line.text != '')
            response = self._translate(self.LINE_BREAK.join(non_empty))
            contents = response.text.split(self.LINE_BREAK)
            if len(contents) == len(non_empty):
                i = 0
                for line in current:
                    if line.text == "":
                        translated.append(line)
                        continue
                    cpy: subs.SubEvent = copy.copy(line)
                    cpy.text = contents[i]
                    cpy.clean_zh(line.text)
                    translated.append(cpy)
                    self.history_append(line.text, cpy.text)
                    i += 1
                yield Progress(len(translated), len(sub), '', translated, False)
            else:
                self._warning_lines_mismatch(non_empty, contents)
                # retry line by line
                print("回退至逐行翻译模式")
                for line in current:
                    cpy: subs.SubEvent = copy.copy(line)
                    if line.text != "":
                        cpy.text = self._translate(line.text).text
                        cpy.clean_zh(line.text)
                    translated.append(cpy)
                    self.history_append(line.text, cpy.text)
                    yield Progress(len(translated), len(sub), '', translated, False)
        yield Progress(len(translated), len(sub), '', translated, True)

    def _warning_lines_mismatch(self, srcs, trss):
        print(f"多行翻译返回结果行数不匹配, 输入[{len(srcs)}]行, 返回[{len(trss)}]行:")
        print("对比:")
        i = 0
        result = []
        while i < len(srcs) or i < len(trss):
            src = srcs[i] if i < len(srcs) else ''
            trs = trss[i] if i < len(trss) else ''
            result.append(f"{src} / {trs}")
            i += 1
        pprint(result)

    def _translate(self, text: str):
        prompt = self.get_prompt(text)
        return self.model.completion(prompt, self.generation_config)

    def get_prompt(self, text: str):
        history = list(self.history)
        user = text
        assistant = ""
        if len(history) > 0:
            history_user = "\n".join([src for src, trs in history])
            history_assistant = "\n".join([trs for src, trs in history])
            user = f"{history_user}\n{text}"
            assistant = f"{history_assistant}\n"
        if "0.10" in self.model.cfg.model_version:
            gpt_dict_text_list = list(f"{row[0]}->{row[1]}" for row in self.gpt_dict)
            gpt_dict_raw_text = "\n".join(gpt_dict_text_list)
            prompt = "<|im_start|>system\n" \
                     "你是一个轻小说翻译模型，可以流畅通顺地使用给定的术语表以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，不要擅自添加原文中没有的代词，也不要擅自增加或减少换行。<|im_end|>\n" \
                     "<|im_start|>user\n" \
                     "根据以下术语表（可以为空）：\n" \
                      f"{gpt_dict_raw_text}\n\n" \
                      f"将下面的日文文本根据上述术语表的对应关系和备注翻译成中文：{user}<|im_end|>\n" \
                      f"<|im_start|>assistant\n{assistant}"
            return prompt
        if "0.9" in self.model.cfg.model_version:
            prompt = f"<|im_start|>system\n" \
                     "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。<|im_end|>\n" \
                     "<|im_start|>user\n" \
                     f"将下面的日文文本翻译成中文：{user}<|im_end|>\n" \
                     f"<|im_start|>assistant\n{assistant}"
            return prompt
        raise ValueError(
            f"Wrong model version{self.model.cfg.model_version}, please view https://huggingface.co/sakuraumi/Sakura-13B-Galgame")

