import shutil
import textgrid
import soundfile as sf
import os
import numpy as np
from scipy.signal import find_peaks
from pydub import AudioSegment
import random
from datetime import datetime
# 读取音频文件和TextGrid文件
# 获取当前文件所在目录的父目录
# 在Jupyter notebook中获取当前工作目录
parent_dir = os.getcwd()
# 获取所有音频文件
audio_files = [f for f in os.listdir(parent_dir) if f.endswith(('.wav', '.mp3'))]

# 根据音频文件名找到对应的TextGrid文件
tg_files = []
for audio_file in audio_files:
    # 移除音频文件扩展名并添加.TextGrid扩展名
    base_name = os.path.splitext(audio_file)[0]
    tg_file = base_name + '.TextGrid'
    if os.path.exists(os.path.join(parent_dir, tg_file)):
        tg_files.append(tg_file)


    print(audio_file)
    new_tg_file = audio_file.replace('.wav', '_.TextGrid')
    # 读取音频文件
    audio = AudioSegment.from_wav(audio_file).split_to_mono()[0]
    # 读取TextGrid文件
    try:
        tg = textgrid.TextGrid()
        tg.read(new_tg_file)
    except FileNotFoundError:
        print(f"TextGrid文件 {new_tg_file} 未找到，跳过。")
        continue

    # 遍历所有层级（tiers）
    for tier in tg:
        # 遍历层级中的所有Interval
        for interval in tier.intervals:
            if interval.mark == '-':
                print("detect")

                # 计算时间点（转换为毫秒）
                start_time = int(interval.minTime * 1000)
                end_time = int(interval.maxTime * 1000)
                # 截取音频片段
                sub_audio = audio[start_time:end_time]
                # 生成导出文件名
                export_filename = audio_file.replace('.wav', f'_{start_time}_{end_time}.wav')
                # 导出音频片段
                sub_audio.export(os.path.join(os.getcwd(), "pure noise", export_filename), format='wav')
            
            elif interval.mark == "-/":
                start_time = int(interval.minTime * 1000)
                end_time = int(interval.maxTime * 1000)
                # 截取音频片段
                sub_audio = audio[start_time:end_time]
                # 生成导出文件名
                export_filename = audio_file.replace('.wav', f'_{start_time}_{end_time}.wav')
                # 导出音频片段
                sub_audio.export(os.path.join(os.getcwd(), "mixture", export_filename), format='wav')

            else:
                print("skip")
            


def read_files_in_folder(folder_path):
    """
    读取指定文件夹中的所有文件路径并返回列表
    
    参数:
    folder_path (str): 文件夹路径
    
    返回:
    list: 文件路径列表
    """
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在")
        return []
    
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    return file_paths

# 分别读取 pure noise 和 pure speech 文件夹
pure_noise_folder = os.path.join(os.getcwd(), 'pure noise')
pure_speech_folder = os.path.join(os.getcwd(), 'pure speech')

pure_noise_files = read_files_in_folder(pure_noise_folder)
pure_speech_files = read_files_in_folder(pure_speech_folder)

print(pure_noise_files)
print(pure_speech_files)




def traverse_files(noise_files, speech_files, n, m, dataset_path):
    """
    遍历每一个noise file，然后随机抽取n个speech file进行遍历
    
    :param noise_files: 噪声文件列表
    :param speech_files: 语音文件列表
    :param n: 每个噪声文件对应的随机抽取的语音文件数量
    :param m: 每个语音文件对应的随机抽取的片段数量

    """
    audio_number = 0

    # 若dataset_folder存在，则删除它
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)

    # 创建数据集文件夹
    os.makedirs(dataset_folder, exist_ok=True)
    for folder_name in ["noisy", "clean", "speech"]:
        os.makedirs(os.path.join(dataset_path, folder_name), exist_ok=True)


    for noise_file in noise_files:
        # 随机抽取n个speech file
        random_speech_files = random.sample(speech_files, min(n, len(speech_files)))
        for speech_file in random_speech_files:
            print(speech_file)
            # 此处可添加具体功能代码

            # 读取noise文件为array
            noise_array, noise_sr = sf.read(noise_file)
            # 读取speech文件为array
            speech_array, speech_sr = sf.read(speech_file)
            # 如果两者采样率不同直接报错
            if noise_sr != speech_sr:
                raise ValueError("噪声文件和语音文件的采样率不同，请确保采样率一致。")
            sr = noise_sr

            # 仅取第一个channel
            if len(noise_array.shape) > 1:
                noise_array = noise_array[:, 0]
            if len(speech_array.shape) > 1:
                speech_array = speech_array[:, 0]

            # 获取noise数组和speech数组长度的最小值
            min_len = min(len(noise_array), len(speech_array))

            # 打印noise数组、speech数组长度以及它们的最小值
            # print("noise数组长度:", len(noise_array))
            # print("speech数组长度:", len(speech_array))
            # print("最小长度:", min_len)

            # 取m个大于0小于等于min_len的随机数
            random_numbers = [random.randint(1, min_len) for _ in range(m)]
            random_numbers += [min_len]  * len(random_numbers)

            for random_number in random_numbers:
                
                # 截取speech数组头部指定长度的片段
                speech_head = speech_array[:random_number]
                diff_len = len(noise_array) - len(speech_head)

                if random_number == len(speech_array):
                    # print("random_number和speech_array长度相等")
                    # 若random_number和speech_array长度相等，左右随机补0到noise长度
                    left_zeros = random.randint(0, diff_len)
                    right_zeros = diff_len - left_zeros
                    speech_head = np.concatenate([np.zeros(left_zeros), speech_head, np.zeros(right_zeros)])
                else:
                    # print("random_number和speech_array长度不相等")
                    # 否则在左侧补0到random_number长度
                    speech_head = np.concatenate([np.zeros(diff_len), speech_head])
                
                speech_head *= 0.55

                # 找出speech_head最小的非零振幅
                min_non_zero_amp = np.min(np.abs(speech_head[speech_head != 0]))

                # 生成与speech_head长度相等的白噪音音频信号，采样率为8000Hz
                white_noise = np.random.normal(0, min_non_zero_amp, len(speech_head))

                # 给speech_head加上白噪音
                speech_head += white_noise

                
                mixutre_head = speech_head + noise_array
                
                audio_number += 1

                sf.write(os.path.join(dataset_path, 'noisy', f'{audio_number}.wav'), mixutre_head, sr)  # 假设采样率为sr
                sf.write(os.path.join(dataset_path, 'clean', f'clean_fileid_{audio_number}.wav'), speech_head, sr)
                sf.write(os.path.join(dataset_path, 'speech', f'speech_fileid_{audio_number}.wav'), noise_array, sr)
                
                


                # 截取speech数组尾部指定长度的片段
                speech_tail = speech_array[-random_number:]
                diff_len = len(noise_array) - len(speech_tail)

                if random_number == len(speech_array):
                    # print("random_number和speech_array长度相等")
                    # 若random_number和speech_array长度相等，左右随机补0到noise长度
                    left_zeros = random.randint(0, diff_len)
                    right_zeros = diff_len - left_zeros
                    speech_tail = np.concatenate([np.zeros(left_zeros), speech_tail, np.zeros(right_zeros)])
                else:
                    # print("random_number和speech_array长度不相等")
                    # 否则在右侧补0到random_number长度
                    speech_tail = np.concatenate([speech_tail, np.zeros(diff_len)])
                

                speech_tail *= 0.55


                # 找出speech_tail最小的非负振幅
                min_non_zero_amp = np.min(np.abs(speech_tail[speech_tail != 0]))

                # 生成与speech_tail长度相等的白噪音，采样率为8000Hz
                white_noise = np.random.normal(0, min_non_zero_amp, len(speech_tail))

                # 给speech_tail加上白噪音
                speech_tail += white_noise


                mixture_tail = speech_tail + noise_array

                # 在dataset_path下创建按数字顺序命名的文件夹
                audio_number += 1

                sf.write(os.path.join(dataset_path, 'noisy', f'{audio_number}.wav'), mixture_tail, sr)  # 假设采样率为sr
                sf.write(os.path.join(dataset_path, 'clean', f'clean_fileid_{audio_number}.wav'), speech_tail, sr)
                sf.write(os.path.join(dataset_path, 'speech', f'speech_fileid_{audio_number}.wav'), noise_array, sr)
            




# 获取当前时间戳并格式化为指定格式
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 定义数据集文件夹路径
# dataset_folder = os.path.join('datasets', timestamp)
dataset_folder = os.path.join(os.getcwd(), 'datasets', "demri")#timestamp)


                


traverse_files(pure_noise_files, pure_speech_files, 2, 2, dataset_folder)


exit()
# 定义训练集、验证集和测试集的比例
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# 获取dataset_folder下的所有文件夹
folders = [f for f in os.listdir(dataset_folder) 
            if os.path.isdir(os.path.join(dataset_folder, f))]

# 打乱文件夹顺序
random.shuffle(folders)

# 定义valid、test和train文件夹路径
valid_folder = os.path.join(dataset_folder, 'valid')
test_folder = os.path.join(dataset_folder, 'test')
train_folder = os.path.join(dataset_folder, 'train')

# 创建目标文件夹
os.makedirs(valid_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)

# 计算验证集和测试集的分割索引
valid_split_index = int(len(folders) * valid_ratio)
test_split_index = int(len(folders) * (valid_ratio + test_ratio))

# 分割文件夹
valid_folders = folders[:valid_split_index]
test_folders = folders[valid_split_index:test_split_index]
train_folders = folders[test_split_index:]

# 移动验证集文件夹
for folder in valid_folders:
    src = os.path.join(dataset_folder, folder)
    dst = os.path.join(valid_folder, folder)
    shutil.move(src, dst)

# 移动测试集文件夹
for folder in test_folders:
    src = os.path.join(dataset_folder, folder)
    dst = os.path.join(test_folder, folder)
    shutil.move(src, dst)

# 移动训练集文件夹
for folder in train_folders:
    src = os.path.join(dataset_folder, folder)
    dst = os.path.join(train_folder, folder)
    shutil.move(src, dst)




