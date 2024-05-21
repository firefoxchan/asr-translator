import os
import re
import collections
from typing import TextIO

import pysubs2
from pathvalidate import sanitize_filename

import dicts


class SubEventWord:
    def __init__(self, start=0.0, end=0.0, word='', score=0.0):
        self.start = start
        self.end = end
        self.word = word
        self.score = score


class SubEvent:
    CLEAN_RE = re.compile(r"(.{2,}?)\1{2,}")
    CLEAN_JA_RE = re.compile(r"([んうあ])\1{3,}")
    CLEAN_ZH_RE = re.compile(r"([啊嗯唔咦哦])\1{2,}")

    def __init__(self, start=0.0, end=0.0, text='', words: [SubEventWord] = None):
        self.start = start
        self.end = end
        self.text = text.replace("\r\n", "\n").replace("\n", " ").strip()
        self.line_count = 1
        self.words = words

    def __repr__(self):
        return f"[{self.start}, {self.end}] {self.text}"

    def __str__(self):
        return f"[{self.start}, {self.end}] {self.text}"

    def clean(self):
        self.text = re.sub(self.CLEAN_RE, r"\1", self.text)
        return self

    def clean_ja(self):
        self.clean()
        self.text = re.sub(self.CLEAN_JA_RE, r"\1", self.text)
        for old, new in dicts.whisper_ja_replace:
            self.text = self.text.replace(old, new)
        for old, new in dicts.whisper_ja_regex:
            self.text = re.sub(old, new, self.text)
        return self

    def clean_zh(self, transcript):
        self.clean()
        self.text = re.sub(self.CLEAN_ZH_RE, r"\1", self.text)
        for src, old, new in dicts.translate_zh_replace:
            if src != "" and transcript.find(src) < 0:
                continue
            self.text = self.text.replace(old, new)
        for src, old, new in dicts.translate_zh_regex:
            if src is not None and re.match(src, transcript):
                continue
            self.text = re.sub(old, new, self.text)
        return self


class Sub(collections.UserList):
    def __init__(self, events: [SubEvent] = None):
        if events is None:
            events = []
        super().__init__(events)

    @staticmethod
    def load_file(file):
        ext = os.path.splitext(file)[1].lstrip(".").lower()
        match ext:
            case 'txt':
                return Sub.load_txt(file)
            case 'lrc':
                return Sub.load_lrc(file)
            case 'srt':
                return Sub.load_pysubs2(file, ext)
            case 'vtt':
                return Sub.load_pysubs2(file, ext)
            case _:
                raise ValueError(
                    f"Unsupported file ext: {ext}"
                )

    @staticmethod
    def load_pysubs2(file, format_: str = None):
        ssa = pysubs2.load(file, format_=format_)
        return Sub([SubEvent(start=event.start / 1000.0, end=event.end / 1000.0, text=event.text) for event in ssa])

    @staticmethod
    def load_lrc(file):
        sub = Sub()
        with open(file, mode='r', encoding='utf8') as f:
            lines = f.readlines()
            half_events = []
            for i, segment in enumerate(lines):
                start, text = segment.split("]", 1)
                text = text.strip()
                if text == "":
                    continue
                start = parse_lrc_timestamp(start.lstrip("["))
                half_events.append((start, text))
            half_events.sort(key=lambda x: x[0])
            for i, segment in enumerate(half_events):
                start, text = segment
                end = start + 10  # fallback
                if i+1 < len(half_events):
                    end = half_events[i+1][0]
                words = []
                # TODO: parse A2 extension
                sub.append(SubEvent(start=start, end=end, text=text, words=words))
        return sub

    @staticmethod
    def load_txt(file):
        with open(file, mode='r', encoding='utf8') as f:
            lines = f.readlines()
            return Sub([SubEvent(text=line) for line in lines if len(line.strip()) > 0])

    @staticmethod
    def from_transformer_whisper(whisper_result):
        sub = Sub()
        for idx, chunk in enumerate(whisper_result["chunks"]):
            begin, end = chunk["timestamp"]
            if begin is None:
                if idx == 0:
                    begin = 0
                else:
                    begin = whisper_result["chunks"][idx - 1]["timestamp"][1]
            if end is None:
                if idx != len(whisper_result["chunks"]) - 1:
                    end = whisper_result["chunks"][idx + 1]["timestamp"][0]
                else:
                    end = begin + 10.0
            sub.append(SubEvent(start=begin, end=end, text=chunk["text"]))
        return merge_sub(sub)

    @staticmethod
    def from_fast_whisper(whisper_result):
        sub = Sub([])
        for idx, seg in enumerate(whisper_result["segments"]):
            begin, end = seg["start"], seg["end"]
            if begin is None:
                if idx == 0:
                    begin = 0
                else:
                    begin = whisper_result["segments"][idx - 1]["end"]
            if end is None:
                if idx != len(whisper_result["segments"]) - 1:
                    end = whisper_result["segments"][idx + 1]["begin"]
                else:
                    end = begin + 10.0
            words = []
            if seg["words"]:
                words = list(SubEventWord(**word) for word in seg["words"])
            sub.append(SubEvent(start=begin, end=end, text=seg["text"], words=words))
        return merge_sub(sub)


def merge_sub(sub: Sub) -> Sub:
    # try merge
    merged = Sub([])
    i = 0
    while i < len(sub):
        if sub[i].text.strip() == '':
            merged.append(sub[i])
            i += 1
            continue
        start, end, text = sub[i].start, sub[i].end, sub[i].text
        j = i+1
        while j < len(sub):
            if sub[j].text.startswith(text):
                end, text = sub[j].end, sub[j].text
                j += 1
                continue
            break
        merged.append(SubEvent(start=start, end=end, text=text))
        i = j
    return merged


def write_all(sub: Sub, base_dir, filename, formats) -> list[str]:
    writers = {
        'lrc': write_lrc,
        'srt': write_srt,
        'vtt': write_vtt,
        'txt': write_txt,
    }
    files = []
    filename = sanitize_filename(filename)
    if not formats or len(formats) == 0:
        formats = ['lrc']
    for fmt in formats:
        writer = writers[fmt]
        filepath = os.path.join(base_dir, f'{filename}.{fmt}')
        if writer:
            with open(filepath, "w", encoding='utf-8') as f:
                writer(sub, f)
            files.append(filepath)
    return files


def write_vtt(sub: Sub, f: TextIO):
    lines = ["WebVTT\n\n"]
    for idx, event in enumerate(sub):
        lines.append(f"{idx + 1}\n")
        lines.append(f"{format_vtt_timestamp(event.start)} --> {format_vtt_timestamp(event.end)}\n")
        lines.append(f"{event.text}\n\n")
    f.writelines(lines)


def write_srt(sub: Sub, f: TextIO):
    lines = []
    for idx, event in enumerate(sub):
        lines.append(f"{idx + 1}\n")
        lines.append(f"{format_srt_timestamp(event.start)} --> {format_srt_timestamp(event.end)}\n")
        lines.append(f"{event.text}\n\n")
    f.writelines(lines)


def format_vtt_timestamp(seconds: float):
    return format_timestamp(seconds, '.')


def format_srt_timestamp(seconds: float):
    return format_timestamp(seconds, ',')


def format_timestamp(seconds: float, delim: str):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    hours = milliseconds // 3600_000
    milliseconds -= hours * 3600_000
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return (
        f"{hours}{minutes:02d}:{seconds:02d}{delim}{milliseconds:03d}"
    )


def format_lrc_timestamp(seconds: float):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)
    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000
    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000
    return (
        f"{minutes:02d}:{seconds:02d}.{(milliseconds // 10):02d}"
    )


def parse_lrc_timestamp(s: str):
    def parse_int(v: str, prefix: str):
        v = v.lstrip(prefix)
        if v == "":
            return 0
        return int(v)
    minutes_and_seconds, milliseconds = s.split(".", 1)
    # ms
    milliseconds = 10 * parse_int(milliseconds, "0")
    # seconds
    minutes_or_seconds = minutes_and_seconds.split(":")
    minutes, seconds = 0, 0
    match len(minutes_or_seconds):
        case 1:
            seconds = parse_int(minutes_or_seconds[0], "0")
        case _:
            minutes = parse_int(minutes_or_seconds[0], "0")
            seconds = parse_int(minutes_or_seconds[1], "0")
    return minutes*60 + seconds*1 + milliseconds*0.001


def write_lrc(sub: Sub, f: TextIO):
    lines = []
    for idx, event in enumerate(sub):
        start_s = format_lrc_timestamp(event.start)
        end_s = format_lrc_timestamp(event.end)
        lines.append(f"[{start_s}]{event.text}\n")
        if idx != len(sub) - 1:
            next_start = sub[idx + 1].start
            if next_start is not None:
                next_start_s = format_lrc_timestamp(next_start)
                if end_s == next_start_s:
                    continue
        lines.append(f"[{end_s}]\n")
    f.writelines(lines)


def write_txt(sub: Sub, f: TextIO):
    lines = []
    for event in sub:
        lines.append(f"{event.text}\n")
    f.writelines(lines)
