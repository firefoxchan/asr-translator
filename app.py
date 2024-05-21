import argparse
import gc
import json
import os
import datetime
import time

import gradio as gr
import torch
import whisperx
from pathvalidate import sanitize_filename

import py7zr

import dicts
import llm
import subs
from translate import SakuraLLMTranslator


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_name', type=str, default='127.0.0.1')
    parser.add_argument('--server_port', type=int, default='20233')
    parser.add_argument('--root_path', type=str, default=None)
    parser.add_argument('--username', type=str, default=None)
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--upload_dir', type=str, default=None)
    parser.add_argument('--model_name_or_path', type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true', default=False)
    parser.add_argument('--text_length', type=int, default=1024)
    parser.add_argument('--translate_show_progress', action='store_true', default=False)
    return parser.parse_args()


class Progress:
    def __init__(self, current: float, total: float, desc: str, data: any):
        self.current = float(current)
        self.total = float(total)
        self.desc = desc
        self.data = data


class App:
    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks()
        upload_dir = args.upload_dir
        if not upload_dir:
            upload_dir = gr.utils.get_upload_folder()
        self.output_dir = os.path.join(upload_dir, 'output')
        self.debug_dir = os.path.join(upload_dir, 'debug')
        self.concurrent_id = '__global__'
        self.transcribe_device = "cuda"
        self.transcribe_model = "large-v2"
        self.transcribe_compute_type = "float16"
        self.sakura_config = llm.SakuraConfig(
            model_name_or_path=args.model_name_or_path,
            use_gpu=args.use_gpu,
            text_length=args.text_length,
        )
        self.sakura_generation_config = llm.SakuraGenerationConfig(
            temperature=0.1,
            top_p=0.3,
            top_k=40,
            max_new_tokens=2*self.sakura_config.text_length,
            frequency_penalty=0.0,
        )
        self.translate_show_progress = args.translate_show_progress

    def current_output_dir(self) -> str:
        now = datetime.datetime.now()
        return os.path.join(self.output_dir, now.strftime("%Y%m"), now.strftime("%d"), now.strftime("%H%M%S"))

    def _transcribe_whisperx(self, files):
        yield Progress(0, len(files), f'初始化Whisper', None)
        transcribe_model = whisperx.load_model(self.transcribe_model, self.transcribe_device, compute_type=self.transcribe_compute_type)
        align_model, align_metadata = whisperx.load_align_model(language_code="ja", device=self.transcribe_device)
        i = 0
        for file in files:
            yield Progress(i, len(files), f'转录 ({i+1}/{len(files)})', None)
            audio = whisperx.load_audio(file)
            result = transcribe_model.transcribe(audio, language="ja", chunk_size=6, batch_size=8)
            result = whisperx.align(result["segments"], align_model, align_metadata, audio, self.transcribe_device, return_char_alignments=False)
            # debug out
            try:
                filename = sanitize_filename(os.path.basename(file))
                filepath = os.path.join(self.debug_dir, f'{filename}_{time.time()}.json')
                with open(filepath, "w", encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False)
            except Exception as e:
                print(e)
            yield Progress(i+1, len(files), f'转录 ({i+1}/{len(files)})', subs.SubEvent.from_fast_whisper(result))
            gc.collect()
            torch.cuda.empty_cache()
            i += 1
        del transcribe_model
        del align_model
        del align_metadata
        gc.collect()
        torch.cuda.empty_cache()

    def _translate(self, ss):
        yield Progress(0, len(ss), f'初始化SakuraLLM', None)
        translator = SakuraLLMTranslator(
            self.sakura_config,
            self.sakura_generation_config,
            show_progress=self.translate_show_progress,
        )
        i = 0
        for sub in ss:
            yield Progress(i, len(ss), f'翻译 ({i+1}/{len(ss)})', None)
            for progress in translator.translate(sub):
                if progress.finish:
                    yield Progress(
                        i + 1, len(ss),
                        f'翻译 ({i + 1}/{len(ss)}), 行 ({int(progress.current)}/{int(progress.total)})',
                        progress.data,
                    )
                else:
                    yield Progress(
                        i, len(ss),
                        f'翻译 ({i + 1}/{len(ss)}), 行 ({int(progress.current)}/{int(progress.total)})',
                        None,
                    )
            i += 1
        del translator
        gc.collect()
        torch.cuda.empty_cache()

    def transcribe(self, files, formats):
        if not files or len(files) == 0:
            return [], ''
        dicts.reload()
        output_dir = self.current_output_dir()
        output_transcribe_dir = os.path.join(output_dir, 'transcribe')
        os.makedirs(output_transcribe_dir, 0o755, exist_ok=True)
        transcribes = []
        i = 0
        for progress in self._transcribe_whisperx(files):
            if progress.data is None:
                yield transcribes, progress.desc
                continue
            file = files[i]
            curr_transcribes = subs.write_all(
                list(event.clean_ja() for event in progress.data), output_transcribe_dir, os.path.splitext(os.path.basename(file.name))[0], formats,
            )
            transcribes.extend(curr_transcribes)
            i += 1
            yield transcribes, progress.desc
        # archive
        transcribe_archive_path = os.path.join(output_dir, '转录打包.7z')
        with py7zr.SevenZipFile(transcribe_archive_path, 'w') as archive:
            archive.writeall(output_transcribe_dir, "")
        transcribes.append(transcribe_archive_path)
        yield transcribes, '结束'

    def translate(self, files, formats):
        if not files or len(files) == 0:
            return [], ''
        dicts.reload()
        output_dir = self.current_output_dir()
        output_translate_dir = os.path.join(output_dir, 'translate')
        os.makedirs(output_translate_dir, 0o755, exist_ok=True)
        translates = []
        i = 0
        for progress in self._translate(list(subs.SubEvent.load_file(file) for file in files)):
            if progress.data is None:
                yield translates, progress.desc
                continue
            sub = progress.data
            file = files[i]
            is_txt = os.path.splitext(file.name)[1].lstrip(".").lower() == 'txt'
            if is_txt:
                curr_transcribes = subs.write_all(
                    sub, output_translate_dir, os.path.splitext(os.path.basename(file.name))[0], ['txt'],
                )
            else:
                curr_transcribes = subs.write_all(
                    sub, output_translate_dir, os.path.splitext(os.path.basename(file.name))[0], formats,
                )
            translates.extend(curr_transcribes)
            i += 1
            yield translates, progress.desc
        # archive
        translate_archive_path = os.path.join(output_dir, '翻译打包.7z')
        with py7zr.SevenZipFile(translate_archive_path, 'w') as archive:
            archive.writeall(output_translate_dir, "")
        translates.append(translate_archive_path)
        yield translates, '结束'

    def transcribe_then_translate(self, files, formats):
        if not files or len(files) == 0:
            return [], [], ''
        dicts.reload()
        output_dir = self.current_output_dir()
        output_transcribe_dir = os.path.join(output_dir, 'transcribe')
        output_translate_dir = os.path.join(output_dir, 'translate')
        os.makedirs(output_transcribe_dir, 0o755, exist_ok=True)
        os.makedirs(output_translate_dir, 0o755, exist_ok=True)
        transcribes = []
        ss = []
        translates = []
        i = 0
        for progress in self._transcribe_whisperx(files):
            if progress.data is None:
                yield transcribes, translates, progress.desc
                continue
            ss.append(progress.data)
            curr_transcribes = subs.write_all(
                list(event.clean_ja() for event in progress.data), output_transcribe_dir, os.path.splitext(os.path.basename(files[i].name))[0], formats,
            )
            transcribes.extend(curr_transcribes)
            i += 1
            yield transcribes, translates, progress.desc
        # archive
        transcribe_archive_path = os.path.join(output_dir, '转录打包.7z')
        with py7zr.SevenZipFile(transcribe_archive_path, 'w') as archive:
            archive.writeall(output_transcribe_dir, "")
        transcribes.append(transcribe_archive_path)
        yield transcribes, translates, '转录结束, 等待启动翻译'
        # wait 5s
        time.sleep(5)
        gc.collect()
        torch.cuda.empty_cache()
        # translate
        i = 0
        for progress in self._translate(ss):
            if progress.data is None:
                yield transcribes, translates, progress.desc
                continue
            sub = progress.data
            curr_transcribes = subs.write_all(
                sub, output_translate_dir, os.path.splitext(os.path.basename(files[i].name))[0], formats,
            )
            translates.extend(curr_transcribes)
            i += 1
            yield transcribes, translates, progress.desc
        # archive
        translate_archive_path = os.path.join(output_dir, '翻译打包.7z')
        with py7zr.SevenZipFile(translate_archive_path, 'w') as archive:
            archive.writeall(output_translate_dir, "")
        translates.append(translate_archive_path)
        all_archive_path = os.path.join(output_dir, '全部打包.7z')
        with py7zr.SevenZipFile(all_archive_path, 'w') as archive:
            archive.writeall(output_transcribe_dir, "转录")
            archive.writeall(output_translate_dir, "翻译")
        translates.append(all_archive_path)
        yield transcribes, translates, '结束'

    def launch(self):
        with self.app:
            with gr.Tabs():
                with gr.TabItem("转录(whisper)+翻译(sakura)"):
                    with gr.Row():
                        with gr.Column():
                            input_files = gr.Files(type="filepath", label="上传音频文件", file_types=['audio'],
                                                   interactive=True)
                        with gr.Column():
                            output_formats = gr.Dropdown(label="输出文件格式", choices=['lrc', 'vtt', 'txt'],
                                                         value=['lrc', 'vtt'], multiselect=True)
                    with gr.Row():
                        btn_run = gr.Button("点击运行", variant="primary")
                        btn_clear = gr.Button("清空")
                    with gr.Row():
                        progress = gr.Text(label="进度")
                    with gr.Row():
                        transcribe_files = gr.Files(label="转录结果", interactive=False)
                        translate_files = gr.Files(label="翻译结果", interactive=False)
                    btn_run.click(self.transcribe_then_translate,
                                  inputs=[input_files, output_formats],
                                  outputs=[transcribe_files, translate_files, progress],
                                  concurrency_id=self.concurrent_id)
                    btn_clear.click(lambda: ([], ['lrc'], [], [], ''), outputs=[input_files, output_formats, transcribe_files, translate_files, progress])
                with gr.TabItem("转录(whisper)"):
                    with gr.Row():
                        input_files = gr.Files(type="filepath", label="上传音频文件", file_types=['audio'],
                                               interactive=True)
                        output_formats = gr.Dropdown(label="输出文件格式",
                                                     choices=['lrc', 'vtt', 'txt'], value=['lrc', 'vtt'],
                                                     multiselect=True)
                    with gr.Row():
                        btn_run = gr.Button("点击运行", variant="primary")
                        btn_clear = gr.Button("清空")
                    with gr.Row():
                        progress = gr.Text(label="进度")
                    with gr.Row():
                        transcribe_files = gr.Files(label="转录结果", interactive=False)
                    btn_run.click(self.transcribe, inputs=[input_files, output_formats], outputs=[transcribe_files, progress],
                                  concurrency_id=self.concurrent_id)
                    btn_clear.click(lambda: ([], ['lrc'], [], ''), outputs=[input_files, output_formats, transcribe_files, progress])
                with gr.TabItem("翻译(sakura)"):
                    with gr.Row():
                        input_files = gr.Files(type="filepath", label="上传文本文件",
                                               file_types=['.lrc', '.srt', '.vtt', '.txt'], interactive=True)
                        output_formats = gr.Dropdown(label="输出文件格式",
                                                     choices=['lrc', 'vtt', 'txt'],
                                                     value=['lrc'],
                                                     multiselect=True)
                    with gr.Row():
                        btn_run = gr.Button("点击运行", variant="primary")
                        btn_clear = gr.Button("清空")
                    with gr.Row():
                        progress = gr.Text(label="进度")
                    with gr.Row():
                        translate_files = gr.Files(label="翻译结果", interactive=False)
                    btn_run.click(self.translate, inputs=[input_files, output_formats], outputs=[translate_files, progress],
                                  concurrency_id=self.concurrent_id)
                    btn_clear.click(lambda: ([], ['lrc'], [], ''), outputs=[input_files, output_formats, translate_files, progress])
        gr_args = {
            k: self.args.__dict__[k] for k in (
                "server_name",
                "server_port",
                "root_path",
            )
        }
        if self.args.username and self.args.password:
            gr_args["auth"] = (self.args.username, self.args.password)
        self.app.queue(default_concurrency_limit=1).launch(**gr_args)


if __name__ == '__main__':
    args = parse_arguments()
    app = App(args)
    app.launch()
