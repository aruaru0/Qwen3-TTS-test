import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="mps",
    dtype=torch.bfloat16,
    # attn_implementation="spda",
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="Hello, how are you?",
    language="English",
    speaker="Ryan",
)
sf.write("test01.wav", wavs[0], sr)

