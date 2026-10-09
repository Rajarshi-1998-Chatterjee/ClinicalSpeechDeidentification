from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import wave
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("Word timestamps must satisfy 0 <= start <= end")


@dataclass(frozen=True)
class TextEntity:
    start: int
    end: int
    label: str
    text: str = ""


@dataclass(frozen=True)
class PHISpan:
    start: float
    end: float
    label: str
    text: str = ""

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("PHI timestamps must satisfy 0 <= start <= end")


@dataclass(frozen=True)
class DeidentificationResult:
    output_path: Path
    words: tuple[Word, ...]
    spans: tuple[PHISpan, ...]


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> Sequence[Word]: ...


class PHIDetector(Protocol):
    def detect(self, text: str) -> Sequence[TextEntity]: ...


class AudioStandardizer:
    def __init__(self, sample_rate: int = 16_000, ffmpeg: str = "ffmpeg") -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.sample_rate = sample_rate
        self.ffmpeg = ffmpeg

    def standardize(self, source: Path, destination: Path) -> None:
        command = [
            self.ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-c:a",
            "pcm_s16le",
            str(destination),
        ]
        subprocess.run(command, check=True, capture_output=True)


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    def transcribe(self, audio_path: Path) -> Sequence[Word]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Install faster-whisper to use FasterWhisperTranscriber"
            ) from exc

        if self._model is None:
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        segments, _ = self._model.transcribe(str(audio_path), word_timestamps=True)
        words = []
        for segment in segments:
            for item in segment.words or ():
                words.append(Word(item.word, float(item.start), float(item.end)))
        return words


class HuggingFacePHIDetector:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._classifier = None

    def detect(self, text: str) -> Sequence[TextEntity]:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers to use HuggingFacePHIDetector"
            ) from exc

        if self._classifier is None:
            self._classifier = pipeline(
                "token-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
            )
        return [
            TextEntity(
                int(entity["start"]),
                int(entity["end"]),
                str(entity.get("entity_group", entity.get("entity", "PHI"))),
                str(entity.get("word", "")),
            )
            for entity in self._classifier(text)
        ]


def resolve_timestamps(
    words: Sequence[Word], entities: Sequence[TextEntity]
) -> tuple[PHISpan, ...]:
    offsets: list[tuple[int, int, Word]] = []
    cursor = 0
    for index, word in enumerate(words):
        if index:
            cursor += 1
        start = cursor
        cursor += len(word.text)
        offsets.append((start, cursor, word))

    spans = []
    text_length = cursor
    for entity in entities:
        if entity.start < 0 or entity.end <= entity.start or entity.end > text_length:
            continue
        matched = [
            word
            for start, end, word in offsets
            if start < entity.end and entity.start < end
        ]
        if matched:
            spans.append(
                PHISpan(
                    matched[0].start,
                    matched[-1].end,
                    entity.label,
                    entity.text,
                )
            )
    return tuple(spans)


class RedactionEngine:
    def redact(
        self,
        source: Path,
        destination: Path,
        spans: Sequence[PHISpan],
        strategy: str = "mute",
        replacement_provider: Callable[
            [PHISpan, int, int], Sequence[int]
        ] | None = None,
    ) -> None:
        if strategy not in {"mute", "beep", "replace"}:
            raise ValueError("strategy must be 'mute', 'beep', or 'replace'")
        if strategy == "replace" and replacement_provider is None:
            raise ValueError("replace strategy requires a replacement provider")

        with wave.open(str(source), "rb") as audio:
            if audio.getsampwidth() != 2:
                raise ValueError("Redaction requires 16-bit PCM WAV audio")
            params = audio.getparams()
            samples = array("h", audio.readframes(audio.getnframes()))
        if samples.itemsize != 2:
            raise ValueError("Platform does not support 16-bit audio samples")
        if sys.byteorder != "little":
            samples.byteswap()

        channels = params.nchannels
        frame_count = len(samples) // channels
        for span in spans:
            first = max(0, int(span.start * params.framerate))
            last = min(frame_count, math.ceil(span.end * params.framerate))
            if last <= first:
                continue
            replacement = (
                replacement_provider(span, last - first, params.framerate)
                if strategy == "replace"
                else ()
            )
            for frame in range(first, last):
                if strategy == "mute":
                    value = 0
                elif strategy == "beep":
                    value = int(
                        8192
                        * math.sin(
                            2 * math.pi * 1000 * frame / params.framerate
                        )
                    )
                else:
                    value = (
                        replacement[frame - first]
                        if frame - first < len(replacement)
                        else 0
                    )
                    value = max(-32768, min(32767, int(value)))
                for channel in range(channels):
                    samples[frame * channels + channel] = value

        if sys.byteorder != "little":
            samples.byteswap()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(destination), "wb") as output:
            output.setparams(params)
            output.writeframes(samples.tobytes())


class ClinicalDeidentificationPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        detector: PHIDetector,
        standardizer: AudioStandardizer | None = None,
        redactor: RedactionEngine | None = None,
    ) -> None:
        self.transcriber = transcriber
        self.detector = detector
        self.standardizer = standardizer or AudioStandardizer()
        self.redactor = redactor or RedactionEngine()

    def run(
        self,
        input_path: str | Path,
        output_path: str | Path,
        strategy: str = "mute",
        replacement_provider: Callable[
            [PHISpan, int, int], Sequence[int]
        ] | None = None,
    ) -> DeidentificationResult:
        source = Path(input_path)
        destination = Path(output_path)
        with tempfile.TemporaryDirectory() as temporary_directory:
            standardized = Path(temporary_directory) / "standardized.wav"
            self.standardizer.standardize(source, standardized)
            words = tuple(self.transcriber.transcribe(standardized))
            transcript = " ".join(word.text for word in words)
            entities = self.detector.detect(transcript)
            spans = resolve_timestamps(words, entities)
            self.redactor.redact(
                standardized, destination, spans, strategy, replacement_provider
            )
        return DeidentificationResult(destination, words, spans)
