import pyaudio
import numpy as np
import soundfile as sf
import time
from collections import deque

THRESHOLD = 500  # 音量阈值
MAX_SILENCE_DURATION = 3  # 最大静音时长（秒）
PRE_RECORD_DURATION = 1  # 预录时间（秒）


def record_audio(file_path="audio_input.wav"):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    CHUNK_DURATION = CHUNK / RATE  # 每个块的持续时间（秒）

    # 计算预录缓冲区的大小
    pre_record_chunks = int(PRE_RECORD_DURATION / CHUNK_DURATION)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("正在监听音量，请开始说话...")
    frames = []
    silence_start = None
    started = False
    pre_record_buffer = deque(maxlen=pre_record_chunks)  # 环形缓冲区

    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)

            # 将音频数据加入缓冲区
            pre_record_buffer.append(data)

            # 计算当前音量
            volume = np.abs(audio_data).mean()
            print(f"当前音量: {volume:.2f}", end="\r")

            if volume > THRESHOLD:
                if not started:
                    print("\n检测到音量超出阈值，开始录音...")
                    started = True
                    # 将预录缓冲区中的音频加入录音
                    frames.extend(pre_record_buffer)
                silence_start = None  # 重置静音计时
                frames.append(data)  # 保存当前音频数据
            else:
                if started:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > MAX_SILENCE_DURATION:
                        print("\n检测到长时间静音，录音结束。")
                        break  # 超过最大静音时长，结束录音

    except KeyboardInterrupt:
        print("\n录音被中断。")

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    if frames:
        # 将录制的音频数据保存为 WAV 文件
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        sf.write(file_path, audio_data, RATE)
        print(f"音频已保存到 {file_path}")
    else:
        print("未检测到音频，无文件保存。")


if __name__ == "__main__":
    record_audio()
