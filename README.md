# asr-translator

音频转录+翻译前后端, 转录部分使用 [WhisperX](https://github.com/m-bain/whisperX) , 翻译部分使用 [SakuraLLM](https://github.com/SakuraLLM)

输出为 lrc / vtt 格式的字幕

目前是写死转录日文音频, 之后翻译到中文  
如果后续SakuraLLM支持更多语言到中文的翻译, 或者加了支持其他语言的翻译后端, 这部分也可以改成可配置或自适应识别的

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

# 使用

运行后, 窗口内应该会显示一个网页链接, 点击即可, 默认的地址是 `http://127.0.0.1:20233`

打开后类似下图, 选择需要的模式, 拖动文件进去之后点击运行即可

![main_0](/assets/main_0.jpg)  

![main_1](/assets/main_1.jpg)  

## 调整转录词汇

编辑文件 `dicts.whisper.ja.csv`, 可以对 whisper 转录出来的文本做简单的替换  
第一列为原词, 第二列为替换词  
