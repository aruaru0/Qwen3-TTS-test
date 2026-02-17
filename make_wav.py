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


idx = 0
while True :
    inst = input() 
    if inst == "end" : break
    s = input()

    print(f"[{idx:02d}]")
    print(f"instruct:{inst}")
    print(f"text:{s}")
    wavs, sr = model.generate_voice_clone(
        text=s,
        language="Japanese",
        voice_clone_prompt=voice_clone_prompt,
        instruct=inst,
    )
    filename = f"voice{idx:02d}.wav"
    sf.write(filename, wavs[0], sr)
    print(f"save:{filename}..done.")
    idx += 1


