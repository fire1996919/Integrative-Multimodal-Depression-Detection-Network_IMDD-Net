import opensmile
import os

# 指定存放wav文件的文件夹路径
wav_folder_path = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'
# 指定输出csv文件的文件夹路径
csv_folder_path = 'D:/Depression/DATA/AVEC/AVEC2014/second_bang/text3'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(csv_folder_path):
    os.makedirs(csv_folder_path)

# 创建一个Smile对象，用于提取eGeMAPS特征
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# 遍历wav文件夹中的所有wav文件
for file_name in os.listdir(wav_folder_path):
    if file_name.endswith('.wav'):
        # 构建wav文件的完整路径
        wav_path = os.path.join(wav_folder_path, file_name)

        # 提取eGeMAPS特征
        features = smile.process_file(wav_path)

        # 构建输出csv文件的路径，保存到新的文件夹中
        csv_path = os.path.join(csv_folder_path, file_name.replace('.wav', '.csv'))

        # 将特征保存到csv文件
        features.to_csv(csv_path)

print('所有wav文件的eGeMAPS特征已提取并保存到D:/Depression/DATA/AVEC/AVEC2014/second_bang/egemaps文件夹中。')
