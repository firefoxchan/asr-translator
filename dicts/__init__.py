import csv


def read_csv(*files):
    result = []
    for file in files:
        file_result = []
        try:
            with open(file, mode='r', encoding='utf8') as csvfile:
                reader = csv.reader(csvfile)
                file_result = list(list(col.strip() for col in row) for row in reader)
        except OSError as e:
            if not file.endswith("private.csv"):
                print(f"warning, dict {file} load failed: {e}")
        if len(file_result) == 0:
            continue
        result.extend(file_result[1:])
    return result


# 调整whisper出来的日语文本
# 格式为 `转录后文本A, 替换文本B`
# 替换规则为, 如果转录后的文本包含A, 则将A替换为B
whisper_ja_replace = read_csv("dicts/whisper.ja.csv", "dicts/whisper.ja.private.csv")

# sakura v0.10 模型可以使用
# 格式为 `原文, 译文`
gpt_dict = read_csv("dicts/gpt.csv", "dicts/gpt.private.csv")

# 调整翻译出来的中文文本
# 格式为 `翻译前文本A, 翻译后文本B, 替换文本C`
# 替换规则为, 如果翻译前包含文本A, 翻译后包含文本B, 则将B替换为C
# 翻译前文本A为空的情况下, 不检查翻译前是否包含此文本
translate_zh_replace = read_csv("dicts/translate.zh.csv", "dicts/translate.zh.private.csv")


def reload():
    global whisper_ja_replace, gpt_dict, translate_zh_replace
    whisper_ja_replace = read_csv("dicts/whisper.ja.csv", "dicts/whisper.ja.private.csv")
    gpt_dict = read_csv("dicts/gpt.csv", "dicts/gpt.private.csv")
    translate_zh_replace = read_csv("dicts/translate.zh.csv", "dicts/translate.zh.private.csv")

