import re


def read_tsv(*files, length: int = None, row_processor=None):
    result = []
    for file in files:
        try:
            with open(file, mode='r', encoding='utf8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split("#", 1)[0]
                    line = line.strip()
                    if line == "":
                        continue
                    frags = line.split("\t")
                    if length is not None:
                        if len(frags) > length:
                            frags = frags[:length]
                        else:
                            while len(frags) < length:
                                frags.append("")
                    frags = list(frag.strip() for frag in frags)
                    if row_processor is not None:
                        frags = row_processor(frags)
                    result.append(frags)
        except OSError as e:
            if not file.endswith("private.tsv"):
                print(f"warning, dict {file} load failed: {e}")
    return result


# 调整whisper出来的日语文本
# 格式为 `转录后文本A, 替换文本B`
# 替换规则为, 如果转录后的文本包含A, 则将A替换为B
whisper_ja_replace = read_tsv("dicts/whisper.ja.tsv", "dicts/whisper.ja.private.tsv", length=2)
# 上面的正则版
whisper_ja_regex = read_tsv("dicts/whisper.ja.re.tsv", "dicts/whisper.ja.re.private.tsv", length=2,
                            row_processor=lambda x: [re.compile(x[0]), x[1]])

# sakura v0.10 模型可以使用
# 格式为 `原文, 译文`
gpt_dict = read_tsv("dicts/gpt.tsv", "dicts/gpt.private.tsv", length=2)

# 调整翻译出来的中文文本
# 格式为 `翻译前文本A, 翻译后文本B, 替换文本C`
# 替换规则为, 如果翻译前包含文本A, 翻译后包含文本B, 则将B替换为C
# 翻译前文本A为空的情况下, 不检查翻译前是否包含此文本
translate_zh_replace = read_tsv("dicts/translate.zh.tsv", "dicts/translate.zh.private.tsv", length=3)
# 上面的正则版
translate_zh_regex = read_tsv("dicts/translate.zh.re.tsv", "dicts/translate.zh.re.private.tsv", length=3,
                              row_processor=lambda x: [re.compile(x[0]) if x[0] else None, re.compile(x[1]), x[2]])


def reload():
    global whisper_ja_replace, whisper_ja_regex, gpt_dict, translate_zh_replace, translate_zh_regex
    whisper_ja_replace = read_tsv("dicts/whisper.ja.tsv", "dicts/whisper.ja.private.tsv", length=2)
    whisper_ja_regex = read_tsv("dicts/whisper.ja.re.tsv", "dicts/whisper.ja.re.private.tsv", length=2,
                                row_processor=lambda x: [re.compile(x[0]), x[1]])
    gpt_dict = read_tsv("dicts/gpt.tsv", "dicts/gpt.private.tsv", length=2)
    translate_zh_replace = read_tsv("dicts/translate.zh.tsv", "dicts/translate.zh.private.tsv", length=3)
    translate_zh_regex = read_tsv("dicts/translate.zh.re.tsv", "dicts/translate.zh.re.private.tsv", length=3,
                                  row_processor=lambda x: [re.compile(x[0]) if x[0] else None, re.compile(x[1]), x[2]])

