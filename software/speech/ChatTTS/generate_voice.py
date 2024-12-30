import ChatTTS
import torch
import torchaudio
import os

# 如果文件夹不存在，则创建存储音频和模型参数的文件夹
os.makedirs('voice_wavs', exist_ok=True)
os.makedirs('voice_pth', exist_ok=True)

# 加载ChatTTS模型
chat = ChatTTS.Chat()
chat.load(compile=False)  # 如果需要更好的性能，可以将compile设置为True

# 生成100个独特的女孩声音
for i in range(100):
    # 生成一个随机讲话者嵌入
    rand_spk = chat.sample_random_speaker()

    # 保存随机生成的讲话者嵌入到.pth文件
    pth_filename = f'./voice_pth/girl_{i + 1}.pth'
    torch.save(rand_spk, pth_filename)

    # 配置语音生成的参数
    params_infer_code = ChatTTS.Chat.InferCodeParams(
        spk_emb=rand_spk,  # 使用采样的讲话者嵌入
        temperature=0.000003,  # 自定义温度，控制生成的多样性和随机性
        top_P=0.7,  # 使用Top-P采样进行解码
        top_K=20  # 使用Top-K采样进行解码
    )

    # 要生成的文本
    text = f'你好，我是小飒'

    # 生成语音波形
    wavs = chat.infer([text], params_infer_code=params_infer_code)

    # 保存生成的音频文件
    # wav_filename = f'./voice_wavs/girl_{i + 1}.wav'
    # # try:
    # torchaudio.save(wav_filename, torch.from_numpy(wavs[0]).unsqueeze(0)[0], 24000, format="wav")    # except:
    #     torchaudio.save(wav_filename, torch.from_numpy(wavs[0]), 24000)

print("已成功生成并保存100个女孩的声音。")
