# ClinicalSpeechDeidentification

A modular baseline for turning clinical audio into a de-identified WAV file:

1. `ffmpeg` converts supported input formats to mono, 16 kHz, 16-bit PCM.
2. A word-timestamped ASR backend produces the transcript.
3. A PHI detector returns character offsets in that transcript.
4. Those offsets are mapped to word timestamps and redacted in the audio.

## Requirements

- Python 3.10 or later
- `ffmpeg` available on `PATH` (or pass its location to `AudioStandardizer`)
- Optional ASR dependency: `pip install faster-whisper`
- Optional NER dependency: `pip install transformers` and the model's runtime
  requirements

The NER model must be a token-classification model that returns entity start/end
character offsets. Use a clinical PHI model fine-tuned with labels such as
`PERSON`, `ADDRESS`, `MRN`, and `PHONE`; the pipeline does not select or train a
model for you. Models should be hosted and run in an environment approved for
the audio's sensitivity.

## Usage

```python
from clinical_speech_deidentification import (
    ClinicalDeidentificationPipeline,
    FasterWhisperTranscriber,
    HuggingFacePHIDetector,
)

pipeline = ClinicalDeidentificationPipeline(
    transcriber=FasterWhisperTranscriber(model_size="small"),
    detector=HuggingFacePHIDetector("your-org/clinical-phi-ner"),
)
result = pipeline.run("recording.mp3", "deidentified.wav", strategy="beep")
print(f"Redacted {len(result.spans)} detected PHI spans")
```

Supported redaction strategies are `mute`, `beep` (a 1 kHz tone), and
`replace`. For `replace`, pass a callback that accepts the detected `PHISpan`,
sample rate, and interval length in frames, and returns 16-bit PCM replacement
samples. Replacement samples are truncated or zero-padded to the redacted
interval. Output is mono WAV audio.

ASR and PHI detection are interfaces, so other engines can be provided by
implementing `transcribe(audio_path)` and `detect(text)`. ASR should return
`Word` objects with start/end times in seconds. Detectors return `TextEntity`
objects with character offsets into the space-joined word transcript.

This is a processing component, not a compliance guarantee. ASR and NER can
miss PHI; validate detections and redacted output before use or release.

## Tests

Run the dependency-free unit tests with:

```sh
python -m unittest discover -s tests
```
