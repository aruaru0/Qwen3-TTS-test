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


voice_clone_prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
)

wavs, sr = model.generate_voice_clone(
    text="""
    本日の天気は、全国的に概ね晴れとなる見込みです。
    しかし、午後は大気の状態が不安定になるため、急な雨や雷にご注意ください。
    お出かけの際は、折りたたみ傘を持っていくと安心でしょう。
    
    以上、気象情報をお伝えしました。
    """,
    language="Japanese",
    voice_clone_prompt=voice_clone_prompt,
    instruct="女性。標準的な日本語のアクセントで、落ち着いて発音。ゆっくり話す。",
)
sf.write("test04.wav", wavs[0], sr)
