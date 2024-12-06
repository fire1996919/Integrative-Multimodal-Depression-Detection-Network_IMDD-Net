from torchvision import transforms
import pandas as pd
import os
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import SubsetRandomSampler
from timesformer_pytorch import TimeSformer
import torch.optim as optim
from transformers import BertModel
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold


class IntegratedDataset(Dataset):
    def __init__(self, audio_feature_dir, text_dir, video_dir, label_file, egemaps_dir, tokenizer, max_text_length):
        """
        初始化综合数据集
        :param audio_feature_dir: 音频特征文件的目录
        :param text_dir: 文本文件的目录
        :param video_dir: 视频帧的目录
        :param label_file: 包含样本标签的CSV文件路径
        :param tokenizer: 用于文本数据的分词器
        :param max_text_length: 文本分词的最大长度
        """
        self.audio_feature_dir = audio_feature_dir
        self.text_dir = text_dir
        self.video_dir = video_dir
        self.label_df = pd.read_csv(label_file)
        self.egemaps_dir = egemaps_dir
        self.samples = self.label_df.iloc[:, 0].values
        self.labels = self.label_df.iloc[:, 1].values
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_name = self.samples[idx]
        label = self.labels[idx]

        # 加载并处理音频特征
        audio_feature_path = os.path.join(self.audio_feature_dir, f"{sample_name}.csv")
        audio_features = pd.read_csv(audio_feature_path, nrows=1000).values[1:, :60]
        audio_features = (audio_features - np.mean(audio_features, axis=0)) / np.std(audio_features, axis=0)  # 归一化

        # 加载并处理文本数据
        text_path = os.path.join(self.text_dir, f"{sample_name}.txt")
        with open(text_path, 'r', encoding='utf-8') as file:
            text = file.read().replace('\n', '')
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_text_length,
            padding='max_length',
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 加载并处理视频帧
        video_folder = os.path.join(self.video_dir, sample_name)
        frames = []
        all_frames = sorted(os.listdir(video_folder), key=lambda x: int(x.split('.')[0]))
        selected_frames = all_frames[::3]  # 使用步长为2来选择帧

        # 限制最终帧的数量
        selected_frames = selected_frames[:30]

        for frame_file in selected_frames:
            frame_path = os.path.join(video_folder, frame_file)
            frame = Image.open(frame_path)
            frame = self.transform(frame)
            frames.append(frame)
        video_tensor = torch.stack(frames)
        # 加载并处理egemaps特征
        egemaps_path = os.path.join(self.egemaps_dir, f"{sample_name}.csv")
        egemaps_features = pd.read_csv(egemaps_path, nrows=1).values

        return {
            'audio_features': torch.tensor(audio_features, dtype=torch.float),
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'video': video_tensor,
            'egemaps_features': torch.tensor(egemaps_features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.float)
        }


# 示例化IntegratedDataset
dataset = IntegratedDataset(
    audio_feature_dir="/home/lyf/data/second_bang/MFCC",
    text_dir="/home/lyf/data/second_bang/text",
    video_dir="/home/lyf/data/second_bang/video_processed",
    label_file="/home/lyf/data/second_bang/all.csv",
    egemaps_dir="/home/lyf/data/second_bang/egemaps",
    tokenizer=BertTokenizer.from_pretrained('/home/lyf/code/second_bang/bert-base-german-cased'),  # 确保你已经实例化了一个分词器
    max_text_length=350
)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=input_dim,
                                                                            nhead=num_heads,
                                                                            dim_feedforward=hidden_dim,
                                                                            dropout=0.3,
                                                                            batch_first=True),
                                                 num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 5)  # 修改此处以输出1*x的向量

    def forward(self, src):
        output = self.transformer(src)
        output = self.fc(output.mean(dim=1))  # 对所有时间帧取平均，然后进行预测
        return output


class BertDepressionModel(nn.Module):
    def __init__(self):
        super(BertDepressionModel, self).__init__()
        self.bert = self.bert = BertModel.from_pretrained('/home/lyf/code/second_bang/bert-base-german-cased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 5)  # 修改此处以输出1*x的向量

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)


class TimeSformerDepressionModel(nn.Module):
    def __init__(self):
        super(TimeSformerDepressionModel, self).__init__()
        self.timesformer = TimeSformer(
            dim=512,
            image_size=64,
            patch_size=8,
            num_frames=30,
            num_classes=5,  # 修改此处以输出1*10的向量
            depth=12,
            heads=8,
            dim_head=64,
            attn_dropout=0.3,
            ff_dropout=0.3
        )

    def forward(self, videos):
        return self.timesformer(videos)


class ResidualBlock(nn.Module):
    def __init__(self, input_features, output_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_features, output_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(output_features, output_features)  # 使用相同的输出特征维度

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if x.size() != out.size():
            identity = nn.Linear(x.size(1), out.size(1))(x)  # 调整身份尺寸以匹配输出
        out += identity
        out = self.relu(out)
        return out


class egemaps(nn.Module):
    def __init__(self, num_residual_blocks=5, input_features=88, hidden_features=88, output_features=5):
        super(egemaps, self).__init__()
        self.input_fc = nn.Linear(input_features, hidden_features)  # 可选的输入调整层
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_features, hidden_features) for _ in range(num_residual_blocks)]
        )
        self.output_fc = nn.Linear(hidden_features, output_features)  # 调整最后输出尺寸的层

    def forward(self, x):
        x = self.input_fc(x)
        x = self.blocks(x)
        x = self.output_fc(x)
        return x


def kronecker_product(t1, t2):
    """计算两个张量的克罗内克积"""
    return torch.einsum('bi,bj->bij', t1, t2).reshape(t1.size(0), -1)


class FusionModel(nn.Module):
    def __init__(self, transformer_model, bert_model, timesformer_model, egemaps_model, num_features,
                 num_residual_blocks=5):
        super(FusionModel, self).__init__()
        self.transformer_model = transformer_model
        self.bert_model = bert_model
        self.timesformer_model = timesformer_model
        self.egemaps_model = egemaps_model
        # 使用相同的 num_features 作为输入和输出特征数
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features, num_features) for _ in range(num_residual_blocks)]
        )
        self.fc_final = nn.Linear(num_features, 1)  # 最终输出一个回归值

    def forward(self, transformer_input, bert_input_ids, bert_attention_mask, timesformer_input, egemaps_features):
        transformer_output = self.transformer_model(transformer_input)
        bert_output = self.bert_model(bert_input_ids, bert_attention_mask)
        timesformer_output = self.timesformer_model(timesformer_input)
        egemaps_output = self.egemaps_model(egemaps_features).squeeze(1)

        # 计算所有模态的克罗内克积
        fused_output = kronecker_product(transformer_output, bert_output)
        fused_output = kronecker_product(fused_output, timesformer_output)
        fused_output = kronecker_product(fused_output, egemaps_output)

        # 通过残差网络处理
        fused_output = self.residual_blocks(fused_output)
        score = self.fc_final(fused_output)
        return score


# 数据集分割为训练集和验证集
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=None)
train_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(train_indices))
val_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(val_indices))

# 初始化模型
transformer_model = TransformerModel(input_dim=60, num_heads=6, hidden_dim=64, num_layers=3)
bert_model = BertDepressionModel()  # 确保这里使用的是修改后的 BertDepressionModel
timesformer_model = TimeSformerDepressionModel()
egemaps_model = egemaps()

num_features = 625
fusion_model = FusionModel(transformer_model, bert_model, timesformer_model, egemaps_model, num_features)

# 将模型移动到设备上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fusion_model = fusion_model.to(device)

# 定义损失函数和优化器
criterion = nn.SmoothL1Loss(beta=4)
optimizer = optim.AdamW(fusion_model.parameters(), lr=0.001, weight_decay=0.01)


# 定义五折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 准备保存结果的列表
results = []

# 遍历每一折
for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
    train_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=8, sampler=SubsetRandomSampler(val_indices))

    # 初始化模型
    transformer_model = TransformerModel(input_dim=60, num_heads=6, hidden_dim=64, num_layers=3)
    bert_model = BertDepressionModel()
    timesformer_model = TimeSformerDepressionModel()
    egemaps_model = egemaps()

    num_features = 625
    fusion_model = FusionModel(transformer_model, bert_model, timesformer_model, egemaps_model, num_features)
    fusion_model = fusion_model.to(device)

    # 定义损失函数和优化器
    criterion = nn.SmoothL1Loss(beta=4)
    optimizer = optim.AdamW(fusion_model.parameters(), lr=0.001, weight_decay=0.01)

    # 训练和验证
    best_val_loss = float('inf')
    for epoch in range(10):  # 每一折训练100个epoch
        # 训练阶段
        fusion_model.train()
        train_loss = 0.0
        for batch in train_loader:
            audio_features = batch['audio_features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            video = batch['video'].to(device)
            egemaps_feature = batch['egemaps_features'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = fusion_model(audio_features, input_ids, attention_mask, video, egemaps_feature)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证阶段
        fusion_model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for batch in val_loader:
                audio_features = batch['audio_features'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                video = batch['video'].to(device)
                egemaps_feature = batch['egemaps_features'].to(device)
                labels = batch['label'].to(device)

                outputs = fusion_model(audio_features, input_ids, attention_mask, video, egemaps_feature)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()

                all_labels.extend(labels.detach().cpu().numpy())
                all_outputs.extend(outputs.detach().cpu().numpy())

        # 计算 RMSE 和 MAE
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)
        rmse = np.sqrt(mean_squared_error(all_labels, all_outputs))
        mae = mean_absolute_error(all_labels, all_outputs)

        # 打印训练和验证损失以及 RMSE 和 MAE
        print(f"Fold {fold + 1}, Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(fusion_model.state_dict(), f'/home/lyf/code/second_bang/best_fusion_model_fold{fold + 1}.pth')

    # 保存当前折的真实值和预测值
    results.append(pd.DataFrame({'True Values': all_labels, 'Predicted Values': all_outputs}))

# 合并所有折的结果
all_results = pd.concat(results, ignore_index=True)
all_results.to_csv('/home/lyf/code/second_bang/cross_validation_results.csv', index=False)

print("Cross-validation completed and results saved.")
