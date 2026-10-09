import shutil
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

from clinical_speech_deidentification import (
    ClinicalDeidentificationPipeline,
    PHISpan,
    RedactionEngine,
    TextEntity,
    Word,
    resolve_timestamps,
)


class CopyStandardizer:
    def standardize(self, source: Path, destination: Path) -> None:
        shutil.copyfile(source, destination)


class FixedTranscriber:
    def transcribe(self, audio_path: Path):
        return [Word("Alice", 0.25, 0.75), Word("called", 0.8, 1.1)]


class FixedDetector:
    def detect(self, text: str):
        self.text = text
        return [TextEntity(0, 5, "PERSON", "Alice")]


def write_test_audio(path: Path) -> None:
    samples = array("h", [12000] * 16000)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.tobytes())


class PipelineTests(unittest.TestCase):
    def test_resolves_character_span_to_word_times(self):
        words = [Word("Alice", 0.25, 0.75), Word("called", 0.8, 1.1)]
        spans = resolve_timestamps(words, [TextEntity(0, 5, "PERSON", "Alice")])
        self.assertEqual(spans, (PHISpan(0.25, 0.75, "PERSON", "Alice"),))

    def test_ignores_entity_offsets_outside_transcript(self):
        spans = resolve_timestamps(
            [Word("Alice", 0, 0.5)],
            [TextEntity(-1, 2, "PERSON"), TextEntity(2, 99, "PERSON")],
        )
        self.assertEqual(spans, ())

    def test_pipeline_detects_and_mutes_audio_span(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            destination = Path(directory) / "redacted.wav"
            write_test_audio(source)
            detector = FixedDetector()
            pipeline = ClinicalDeidentificationPipeline(
                FixedTranscriber(), detector, standardizer=CopyStandardizer()
            )

            result = pipeline.run(source, destination)

            self.assertEqual(detector.text, "Alice called")
            self.assertEqual(result.spans, (PHISpan(0.25, 0.75, "PERSON", "Alice"),))
            with wave.open(str(destination), "rb") as audio:
                redacted = array("h", audio.readframes(audio.getnframes()))
            self.assertTrue(all(sample == 12000 for sample in redacted[:4000]))
            self.assertTrue(all(sample == 0 for sample in redacted[4000:12000]))
            self.assertTrue(all(sample == 12000 for sample in redacted[12000:]))

    def test_replace_strategy_requires_replacement_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            destination = Path(directory) / "redacted.wav"
            write_test_audio(source)
            pipeline = ClinicalDeidentificationPipeline(
                FixedTranscriber(), FixedDetector(), standardizer=CopyStandardizer()
            )
            with self.assertRaisesRegex(ValueError, "requires a replacement provider"):
                pipeline.run(source, destination, strategy="replace")

    def test_beep_and_replacement_redact_requested_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            beep_output = Path(directory) / "beep.wav"
            replace_output = Path(directory) / "replace.wav"
            write_test_audio(source)
            span = PHISpan(0.25, 0.75, "PERSON")
            engine = RedactionEngine()

            engine.redact(source, beep_output, [span], strategy="beep")
            engine.redact(
                source,
                replace_output,
                [span],
                strategy="replace",
                replacement_provider=lambda detected, frames, rate: [2000],
            )

            with wave.open(str(beep_output), "rb") as audio:
                beep = array("h", audio.readframes(audio.getnframes()))
            with wave.open(str(replace_output), "rb") as audio:
                replaced = array("h", audio.readframes(audio.getnframes()))
            self.assertEqual(beep[4000], 0)
            self.assertEqual(beep[4001], 3134)
            self.assertEqual(replaced[3999], 12000)
            self.assertEqual(replaced[4000:4002], array("h", [2000, 0]))
            self.assertEqual(replaced[12000], 12000)


if __name__ == "__main__":
    unittest.main()
