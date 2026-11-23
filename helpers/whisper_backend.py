"""Local, free transcription backend using faster-whisper.

Produces JSON shaped like ElevenLabs Scribe output (`words` entries with
`text` / `start` / `end` / `type` / `speaker_id`), so pack_transcripts.py,
render.py and timeline_view.py work unchanged.

Differences from Scribe, worth knowing when editing:
  - No speaker diarization: `speaker_id` is always None (phrases break on
    silence only).
  - No audio events such as (laughter) or (applause).
  - Whisper tends to drop fillers ("um", "uh"). We bias it toward verbatim
    output with a filler-heavy initial prompt, but it is not perfect.
    Check takes_packed.md for missed fillers before trusting a filler cut.
  - Word timestamps can drift more than Scribe's. Keep cut padding generous.

Model is picked by --model or the VIDEO_USE_WHISPER_MODEL env var
(tiny, base, small, medium, large-v3, or a .en variant). Default: small.
Weights download from Hugging Face on first use, then run offline.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

DEFAULT_MODEL = os.environ.get("VIDEO_USE_WHISPER_MODEL", "small")

# Whisper imitates the style of its prompt, so a prompt full of fillers makes
# it far more likely to keep them in the transcript.
VERBATIM_PROMPT = "Umm, uh, so, like, you know, I mean, uh... right. Hmm, yeah, um."

_models: dict[tuple[str, str, str], object] = {}
_lock = threading.Lock()


def _load_model(model_size: str, device: str, compute_type: str):
    key = (model_size, device, compute_type)
    with _lock:
        if key not in _models:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise SystemExit(
                    "faster-whisper is not installed. Run: pip install faster-whisper "
                    "(or `pip install -e .` in the video-use repo)."
                )
            _models[key] = WhisperModel(model_size, device=device, compute_type=compute_type)
        return _models[key]


def call_whisper(
    audio_path: Path,
    language: str | None = None,
    model: str | None = None,
    device: str = "auto",
    compute_type: str = "auto",
) -> dict:
    """Transcribe a 16 kHz mono wav. Returns a Scribe-shaped dict."""
    model_size = model or DEFAULT_MODEL
    m = _load_model(model_size, device, compute_type)

    segments, info = m.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=True,
        initial_prompt=VERBATIM_PROMPT,
        beam_size=5,
    )

    words: list[dict] = []
    texts: list[str] = []
    prev_end: float | None = None
    for seg in segments:  # lazy generator: decoding happens here
        for w in seg.words or []:
            text = (w.word or "").strip()
            if not text:
                continue
            if prev_end is not None and w.start > prev_end:
                words.append({
                    "text": " ", "start": round(prev_end, 3), "end": round(w.start, 3),
                    "type": "spacing", "speaker_id": None,
                })
            words.append({
                "text": text, "start": round(w.start, 3), "end": round(w.end, 3),
                "type": "word", "speaker_id": None,
            })
            texts.append(text)
            prev_end = w.end

    return {
        "language_code": info.language,
        "language_probability": info.language_probability,
        "text": " ".join(texts),
        "words": words,
        "backend": f"faster-whisper:{model_size}",
    }
