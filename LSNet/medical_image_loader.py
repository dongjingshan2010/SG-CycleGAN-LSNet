import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# class PairedMedicalImageDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         """
#         初始化医学图像数据集
#
#         参数:
#         root_dir (string): 数据集根目录
#         transform (callable, optional): 可选的数据转换函数
#         """
#         self.root_dir = root_dir
#         self.transform = transform
#
#         # 构建图像对列表
#         self.image_pairs = self._find_image_pairs()
#
#     def _find_image_pairs(self):
#         """查找并匹配超声和磁共振图像对"""
#         pairs = []
#
#         # 获取超声和磁共振图像路径
#         us_normal_dir = os.path.join(self.root_dir, "Ultrasonic", "Normal")
#         us_disease_dir = os.path.join(self.root_dir, "Ultrasonic", "Abnormal")
#         mr_normal_dir = os.path.join(self.root_dir, "Magnetic", "Normal")
#         mr_disease_dir = os.path.join(self.root_dir, "Magnetic", "Abnormal")
#
#         # 处理正常图像对
#         us_normal_images = sorted(os.listdir(us_normal_dir))
#         mr_normal_images = sorted(os.listdir(mr_normal_dir))
#
#         for us_img, mr_img in zip(us_normal_images, mr_normal_images):
#             # 确保文件名匹配（假设文件名相同，扩展名可能不同）
#             us_base, _ = os.path.splitext(us_img)
#             mr_base, _ = os.path.splitext(mr_img)
#             if us_base == mr_base:
#                 pairs.append((
#                     os.path.join(us_normal_dir, us_img),
#                     os.path.join(mr_normal_dir, mr_img),
#                     0  # 类别标签：0表示正常
#                 ))
#
#         # 处理疾病图像对
#         us_disease_images = sorted(os.listdir(us_disease_dir))
#         mr_disease_images = sorted(os.listdir(mr_disease_dir))
#
#         for us_img, mr_img in zip(us_disease_images, mr_disease_images):
#             # 确保文件名匹配
#             us_base, _ = os.path.splitext(us_img)
#             mr_base, _ = os.path.splitext(mr_img)
#             if us_base == mr_base:
#                 pairs.append((
#                     os.path.join(us_disease_dir, us_img),
#                     os.path.join(mr_disease_dir, mr_img),
#                     1  # 类别标签：1表示疾病
#                 ))
#
#         return pairs
#
#     def __len__(self):
#         return len(self.image_pairs)
#
#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#
#         us_path, mr_path, label = self.image_pairs[idx]
#
#         # 加载图像
#         us_image = Image.open(us_path).convert('RGB')
#         mr_image = Image.open(mr_path).convert('RGB')
#
#         # 应用数据转换
#         if self.transform:
#             us_image = self.transform(us_image)
#             mr_image = self.transform(mr_image)
#
#         return {
#             'ultrasound': us_image,
#             'mri': mr_image,
#             'label': label
#         }


class PairedMedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化医学图像数据集

        参数:
        root_dir (string): 数据集根目录
        transform (callable, optional): 可选的数据转换函数
        """
        self.root_dir = root_dir
        self.transform = transform

        # 构建图像对列表
        self.image_pairs = self._find_image_pairs()

    def _find_image_pairs(self):
        """查找并匹配超声和磁共振图像对"""
        pairs = []

        # 获取超声和磁共振图像路径 - 新增浅浸润和深浸润类别
        us_normal_dir = os.path.join(self.root_dir, "Uto", "Normal")
        us_superficial_dir = os.path.join(self.root_dir, "Uto", "Superficial")
        us_deep_dir = os.path.join(self.root_dir, "Uto", "Deep")

        mr_normal_dir = os.path.join(self.root_dir, "Mri", "Normal")
        mr_superficial_dir = os.path.join(self.root_dir, "Mri", "Superficial")
        mr_deep_dir = os.path.join(self.root_dir, "Mri", "Deep")

        # 处理正常图像对
        us_normal_images = sorted(os.listdir(us_normal_dir))
        mr_normal_images = sorted(os.listdir(mr_normal_dir))

        for us_img, mr_img in zip(us_normal_images, mr_normal_images):
            # 确保文件名匹配（假设文件名相同，扩展名可能不同）
            us_base, _ = os.path.splitext(us_img)
            mr_base, _ = os.path.splitext(mr_img)
            if us_base == mr_base:
                pairs.append((
                    os.path.join(us_normal_dir, us_img),
                    os.path.join(mr_normal_dir, mr_img),
                    0  # 类别标签：0表示正常
                ))

        # 处理浅浸润图像对
        us_superficial_images = sorted(os.listdir(us_superficial_dir))
        mr_superficial_images = sorted(os.listdir(mr_superficial_dir))

        for us_img, mr_img in zip(us_superficial_images, mr_superficial_images):
            us_base, _ = os.path.splitext(us_img)
            mr_base, _ = os.path.splitext(mr_img)
            if us_base == mr_base:
                pairs.append((
                    os.path.join(us_superficial_dir, us_img),
                    os.path.join(mr_superficial_dir, mr_img),
                    1  # 类别标签：1表示浅浸润
                ))

        # 处理深浸润图像对
        us_deep_images = sorted(os.listdir(us_deep_dir))
        mr_deep_images = sorted(os.listdir(mr_deep_dir))

        for us_img, mr_img in zip(us_deep_images, mr_deep_images):
            us_base, _ = os.path.splitext(us_img)
            mr_base, _ = os.path.splitext(mr_img)
            if us_base == mr_base:
                pairs.append((
                    os.path.join(us_deep_dir, us_img),
                    os.path.join(mr_deep_dir, mr_img),
                    2  # 类别标签：2表示深浸润
                ))

        return pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        us_path, mr_path, label = self.image_pairs[idx]

        # 加载图像
        us_image = Image.open(us_path).convert('RGB')
        mr_image = Image.open(mr_path).convert('RGB')

        # 应用数据转换
        if self.transform:
            us_image = self.transform(us_image)
            mr_image = self.transform(mr_image)

        return {
            'ultrasound': us_image,
            'mri': mr_image,
            'label': label
        }








# # 模型训练使用示例
# def train_with_paired_images(model, dataloader, criterion, optimizer, device):
#     """使用配对图像训练模型的示例"""
#     model.train()
#     for batch in dataloader:
#         # 获取超声和磁共振图像批次
#         us_images = batch['ultrasound'].to(device)
#         mr_images = batch['mri'].to(device)
#         labels = batch['label'].to(device)
#
#         # 前向传播
#         us_outputs = model(us_images)
#         mr_outputs = model(mr_images)
#
#         # 计算损失
#         us_loss = criterion(us_outputs, labels)
#         mr_loss = criterion(mr_outputs, labels)
#         loss = us_loss + mr_loss
#
#         # 反向传播和优化
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
