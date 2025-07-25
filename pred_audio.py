import os
import torch
import torchaudio
from diffwave.inference import predict as diffwave_predict

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 模型路径
current_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(current_dir, 'models', '20250725')

# 输入和输出目录
mixture_dir = os.path.join(current_dir, 'mixture')
output_dir = os.path.join(current_dir, 'output')

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取最新的模型权重
model_weights = [f for f in os.listdir(model_dir) if f.startswith('weights-') and f.endswith('.pt')]
model_weights.sort(key=lambda x: int(x.split('-')[1].split('.')[0]) if x.split('-')[1].split('.')[0].isdigit() else 0)
latest_model = os.path.join(model_dir, model_weights[-1]) if model_weights else os.path.join(model_dir, 'weights.pt')

print(f'Using model: {latest_model}')

# 获取所有混合音频文件
mixture_files = [f for f in os.listdir(mixture_dir) if f.endswith('.wav')]

print(f'Processing {len(mixture_files)} files...')

# 处理每个文件
for i, filename in enumerate(mixture_files):
    print(f'Processing {i+1}/{len(mixture_files)}: {filename}')
    
    # 输入文件路径
    input_path = os.path.join(mixture_dir, filename)
    
    # 加载音频
    waveform, sample_rate = torchaudio.load(input_path)
    
    # 如果是立体声，转换为单声道
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # 转换为正确的格式
    waveform = waveform.to(device)
    
    # 进行推理
    # 注意：这里假设输入是波形，但根据DiffWave的原理，通常输入是梅尔频谱图
    # 如果需要从波形生成梅尔频谱图，需要额外的处理步骤
    # 这里简化处理，直接使用波形作为输入
    
    # 如果需要转换为梅尔频谱图，可以使用以下代码：
    # mel_transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=sample_rate,
    #     n_fft=1024,
    #     hop_length=256,
    #     n_mels=80
    # ).to(device)
    # spectrogram = mel_transform(waveform)
    
    # 但在这里，我们假设您已经有了适当的输入格式
    # 根据DiffWave的API，第一个参数应该是梅尔频谱图
    # spectrogram = waveform  # 这可能需要根据实际情况调整
    
    # 由于DiffWave需要梅尔频谱图作为输入，我们需要生成它
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=256,
        hop_length=64,
        n_mels=80
    ).to(device)
    
    # 添加一个小的epsilon以避免log(0)
    epsilon = 1e-5
    spectrogram = torch.log(mel_transform(waveform) + epsilon)
    
    # 进行预测
    # 使用fast_sampling以提高速度
    audio, sr = diffwave_predict(spectrogram, latest_model, fast_sampling=True)
    
    # 输出文件路径
    output_filename = f"enhanced_{filename}"
    output_path = os.path.join(output_dir, output_filename)
    
    # 保存音频
    torchaudio.save(output_path, audio.cpu(), sr)
    print(f'Saved enhanced audio to {output_path}')


print('All files processed.')