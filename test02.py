import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps",
    dtype=torch.bfloat16,
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="こんにちは、私になにか御用ですか？",
    language="Japanese",
    speaker="Ono_Anna",
)
sf.write("test02.wav", wavs[0], sr)

