import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from scipy.io import wavfile


LOCALAI_HOME = Path(os.environ.get("LOCALAI_HOME", str(Path.home() / "LocalAI")))
_WHISPER_EN = LOCALAI_HOME / "models" / "whisper-base.en"
_WHISPER_BASE = LOCALAI_HOME / "models" / "whisper-base"
WHISPER_MODEL_DIR = _WHISPER_EN if _WHISPER_EN.exists() else _WHISPER_BASE


class LocalSTTEngine:
    """
    100% Offline Speech-to-Text engine powered by Faster-Whisper (CTranslate2).
    Runs on CPU with int8 quantization so it consumes 0 MB of the 4 GB GPU VRAM.
    """

    def __init__(self, model_dir: Path = WHISPER_MODEL_DIR):
        self.model_dir = model_dir
        self._model: Optional[WhisperModel] = None
        self._lock = threading.Lock()
        self.is_recording = False
        self._audio_frames: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self.sample_rate = 16000

    def ensure_loaded(self) -> WhisperModel:
        with self._lock:
            if self._model is None:
                if not self.model_dir.exists():
                    raise FileNotFoundError(
                        f"Offline Whisper model folder not found at {self.model_dir}"
                    )
                self._model = WhisperModel(
                    str(self.model_dir),
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                    local_files_only=True,
                )
            return self._model

    def start_recording(self) -> None:
        if self.is_recording:
            return
        self._audio_frames = []
        self.is_recording = True

        def _callback(indata, frames, time_info, status):
            if self.is_recording:
                self._audio_frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()

    def stop_and_transcribe(self) -> str:
        if not self.is_recording:
            return ""
        self.is_recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

        if not self._audio_frames:
            return ""

        audio_data = np.concatenate(self._audio_frames, axis=0).flatten()
        if audio_data.size < int(self.sample_rate * 0.25):
            return ""

        model = self.ensure_loaded()
        segments, _info = model.transcribe(
            audio_data,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=400),
        )
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(text_parts).strip()

    def transcribe_audio_file(self, audio_path: str) -> str:
        model = self.ensure_loaded()
        segments, _info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
        )
        text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(text_parts).strip()
