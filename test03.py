import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="mps",
    dtype=torch.bfloat16,
)

ref_audio = "https://amitaro.net/download/voice/111_ganbaru/anmarimuzukashiku_01.wav"
ref_text  = "あんまりむずかしく考えすぎるといろいろ大変だしね、しっかり反省したらぱぱっと行こう！"

wavs, sr = model.generate_voice_clone(
    text="大丈夫だよ、これから何度だって挑戦すればいいんだから",
    language="Japanese",
    ref_audio=ref_audio,
    ref_text=ref_text,
    instruct="As the news anchor says"
)
sf.write("test03.wav", wavs[0], sr)
