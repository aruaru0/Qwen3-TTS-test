import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
import time
from functools import wraps

def timer(func):
    @wraps(func)  # 元の関数のメタデータを保持
    def wrapper(*args, **kwargs):
        start = time.perf_counter()  # 計測開始
        result = func(*args, **kwargs)
        end = time.perf_counter()    # 計測終了
        print(f"実行時間 ({func.__name__}): {end - start:.4f} 秒")
        return result
    return wrapper

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="mps",
    dtype=torch.bfloat16,
)

ref_audio = "https://amitaro.net/download/voice/111_ganbaru/anmarimuzukashiku_01.wav"
ref_text  = "あんまりむずかしく考えすぎるといろいろ大変だしね、しっかり反省したらぱぱっと行こう！"


@timer
def method1():
    wavs, sr = model.generate_voice_clone(
        text="大丈夫だよ、これから何度だって挑戦すればいいんだから",
        language="Japanese",
        ref_audio=ref_audio,
        ref_text=ref_text,
    )

@timer
def method2_1():
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
    )
    return voice_clone_prompt

@timer
def method2_2():
    global voice_clone_prompt
    wavs, sr = model.generate_voice_clone(
        text="大丈夫だよ、これから何度だって挑戦すればいいんだから",
        language="Japanese",
        voice_clone_prompt=voice_clone_prompt,
    )

for _ in range(10):
    method1()

voice_clone_prompt = method2_1()

for _ in range(10):
    method2_2()