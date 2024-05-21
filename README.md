# asr-translator

音频转录+翻译前后端, 转录部分使用 [WhisperX](https://github.com/m-bain/whisperX) , 翻译部分使用 [SakuraLLM](https://github.com/SakuraLLM)

输出为 lrc / vtt 格式的字幕

目前是写死转录日文音频, 之后翻译到中文  
如果后续SakuraLLM支持更多语言到中文的翻译, 或者加了支持其他语言的翻译后端, 这部分也可以改成可配置或自适应识别的

## 起因

24G 显存只够刚好跑 32B iq4xs 的 SakuraLLM, 没法并行再跑一个 Whisper 了  
目前好像没看到有项目是自动跑完一个之后释放显存跑另一个的, 所以写了个轮子把他们缝起来  

# 部署

## 环境

目前没有打包好的一键环境, 需要手动部署  

请先参考 [SakuraLLM的部署教程](https://github.com/SakuraLLM/Sakura-13B-Galgame/wiki/Python%E9%83%A8%E7%BD%B2%E6%95%99%E7%A8%8B)  
使用llama-cpp-python后端, 配置好相关的环境 

之后安装本项目的依赖即可 ( `pip install -r requirements.txt` )

## 模型

### Whisper

默认使用 `large-v2`, 别的也是可以的, 具体参考 [WhisperX](https://github.com/m-bain/whisperX) 项目的文档

### SakuraLLM

支持 `v0.9`, `v0.10` 版本的模型  
如果显卡有 `24G` 显存, 可以使用 `32B` 的 `iq4xs` 量化模型

# WebUI

## 运行

参考 `run.sh` 文件, 样例: 

```bash
python app.py \
  --model_name_or_path ./models/sakura-32b-qwen2beta-v0.9-iq4xs.gguf \
  --use_gpu \
  --translate_show_progress
```

如果需要挂在反代后面, 比如挂在nginx后面, 需要加上参数 `--root_path 你的网站地址`, 比如 `--root_path https://whisper.example.com`

如果需要调整临时文件目录, 可以配置环境变量 `GRADIO_TEMP_DIR`, 具体可以参考 [gradio](https://www.gradio.app/) 的文档

## 使用

成功启动后, 窗口内应该会显示一个网页链接, 点击即可, 默认的地址是 `http://127.0.0.1:20233`

打开后类似下图, 选择需要的模式, 拖动文件进去之后点击运行即可

![main_0](/assets/main_0.jpg)  

目前有一个简单的进度说明, 处理完的文件会实时展示在下面, 最后会有一个压缩包, 点击蓝色链接可以下载  

![main_1](/assets/main_1.jpg)  

# CLI

TODO

# 字典

各个字典都有一个文件名部分增加 `.private` 后缀的版本  
比如 `whisper.ja.tsv` 对应的就是 `whisper.ja.private.tsv`  
功能跟主文件一样, 只是不会被git跟踪  

## 格式说明

字典文件中, 每行数据的各列之间使用 **Tab 分隔**, 不是空格, 注意不要打错了  
`#`号开始的内容为注释, 不会被解析, 可以有空行  

## 简单替换

基于文本替换  

### 调整转录结果

编辑文件 `dicts/whisper.ja.tsv`, 可以对 whisper 转录出来的文本做简单的替换  
每行格式为 `原词A	替换词B`  
替换规则为, 如果转录后的文本包含A, 则将A替换为B  

### 调整翻译结果

编辑文件 `dicts/translate.zh.tsv`, 可以对翻译出来的文本做简单的替换  
每行格式为 `翻译前文本A	翻译后文本B	替换文本C`  
替换规则为, 如果翻译前包含文本A, 翻译后包含文本B, 则将B替换为C  
翻译前文本A为空的情况下, 不检查翻译前是否包含此文本  

### 调整翻译词库

目前仅 `0.10` 版本的 SakuraLLM 模型支持此功能  
编辑文件 `dicts/gpt.tsv`, 可以将这个词汇表喂给翻译模型  
每行格式为 `原文A,译文B`  

## 正则

基于正则替换, 可以到 [regex101](https://regex101.com/) 测试, 选择 Python 的正则  

### 调整转录结果

编辑文件 `dicts/whisper.ja.re.tsv`, 可以对 whisper 转录出来的文本做正则替换  
每行格式为 `原文正则A	替换表达式B`  
替换方式为,  `re.sub(原文正则A, 替换表达式B, 一行文本)`

### 调整翻译结果

编辑文件 `dicts/translate.zh.re.tsv`, 可以对翻译出来的文本做正则替换  
每行格式为 `翻译前正则A	翻译后正则B	替换表达式C`  
替换方式为:
- 如果翻译前正则A为空, `re.sub(翻译后正则B, 替换表达式C, 翻译后文本)`  
- 如果翻译前正则A不为空,  `if re.match(翻译前正则A, 翻译前文本): re.sub(翻译后正则B, 替换表达式C, 翻译后文本)`
