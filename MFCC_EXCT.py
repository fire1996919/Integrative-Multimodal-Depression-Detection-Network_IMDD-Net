import librosa
import pandas as pd
import os
import numpy as np

# 定义输入和输出文件夹路径
input_folder = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'
output_folder = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有 WAV 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.wav'):
        # 构建完整的输入和输出文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.wav', '.csv'))

        # 加载音频文件
        y, sr = librosa.load(input_path, sr=None)  # 使用音频的原始采样率

        # 计算 MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        # 计算一阶差分 (Delta MFCC)
        mfcc_delta = librosa.feature.delta(mfccs)

        # 计算二阶差分 (Delta-Delta MFCC)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

        # 将 MFCC 和差分特征拼接在一起
        features = np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=0)

        # 转换为 pandas DataFrame 并保存到 CSV 文件
        features_df = pd.DataFrame(features.T)  # 转置以便每行是一个时间帧的特征
        features_df.to_csv(output_path, index=False)

print("MFCC extraction and saving completed.")
