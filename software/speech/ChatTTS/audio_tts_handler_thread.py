import torch
import ChatTTS
import sounddevice as sd
import cn2an
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# 初始化 ChatTTS
chat = ChatTTS.Chat()
chat.load(compile=False)  # 设置为 True 以获得更好的性能
rand_spk = torch.load('./voice_pth/girl_1.pth')  # 配置语音生成的参数
params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb=rand_spk,
    temperature=0.000003,
    top_P=0.7,
    top_K=20
)
params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]'
)

# 定义线程池
executor = ThreadPoolExecutor(max_workers=4)  # 增加并行线程数


def clean_text(text):
    """清理无效字符"""
    text = text.replace('！', '!').replace('，', ',').replace('。', '.').replace('：', ':')
    return text


def split_text(text):
    """按标点符号分割文本，每个句子最后加入 [uv_break]"""
    import re
    sentences = re.split(r'[\.\!\?,]', text)
    return [sentence.strip() + " [uv_break]" for sentence in sentences if sentence.strip()]


def generate_audio(sentence):
    """为每个句子生成音频"""
    # 转换阿拉伯数字为中文数字
    sentence = cn2an.transform(sentence, "an2cn")

    # 使用 ChatTTS 生成音频
    wavs = chat.infer([sentence], skip_refine_text=True, params_refine_text=params_refine_text,
                      params_infer_code=params_infer_code)
    return wavs[0] if wavs is not None and len(wavs) > 0 else None


def combine_audios(audio_list):
    """组合多个音频"""
    return np.concatenate([audio for audio in audio_list if audio is not None])


def text_to_speech(response_text):
    # 清理无效字符
    response_text = clean_text(response_text)

    # 分割文本为句子
    sentences = split_text(response_text)

    # 使用线程池并行生成音频
    futures = [executor.submit(generate_audio, sentence) for sentence in sentences]
    audio_list = [future.result() for future in futures]

    # 合并音频
    combined_audio = combine_audios(audio_list)

    # 播放合并的音频
    if combined_audio is not None:
        sd.play(combined_audio, samplerate=24000)
        sd.wait()
    else:
        print("未生成任何有效音频。")


def async_text_to_speech(response_text):
    """异步调用 TTS，避免阻塞主线程"""
    executor.submit(text_to_speech, response_text)

if __name__ == "__main__":
    text_to_speech("你好，我是小飒！今天是个美好的日子，元旦快乐。")
