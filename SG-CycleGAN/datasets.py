import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# class ImageDataset(Dataset):
#     def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
#         self.transform = transforms.Compose(transforms_)
#         self.unaligned = unaligned
#
#         self.files_A = sorted(glob.glob(os.path.join(root, '%s/A' % mode) + '/*.*'))
#         self.files_B = sorted(glob.glob(os.path.join(root, '%s/B' % mode) + '/*.*'))
#
#     def __getitem__(self, index):
#         item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
#
#         if self.unaligned:
#             item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert('RGB'))
#         else:
#             item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert('RGB'))
#
#         # 只返回文件名，不包含路径
#         filename_A = os.path.basename(self.files_A[index])
#         filename_B = os.path.basename(self.files_B[index])
#         # 返回图像数据和文件名
#         return {
#             'A': item_A,
#             'B': item_B,
#             'A_name': filename_A,
#             'B_name': filename_B
#         }
#
#     def __len__(self):
#         return max(len(self.files_A), len(self.files_B))

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/Uto' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/MRI' % mode) + '/*.*'))

    def __getitem__(self, index):
        # 确保 index 在 files_A 的范围内
        index_A = index % len(self.files_A)
        item_A = self.transform(Image.open(self.files_A[index_A]))

        # 确保 index 在 files_B 的范围内
        if self.unaligned:
            index_B = random.randint(0, len(self.files_B) - 1)
        else:
            index_B = index % len(self.files_B)
        item_B = self.transform(Image.open(self.files_B[index_B]).convert('RGB'))

        # 只返回文件名，不包含路径
        filename_A = os.path.basename(self.files_A[index_A])
        filename_B = os.path.basename(self.files_B[index_B])
        # 返回图像数据和文件名
        return {
            'A': item_A,
            'B': item_B,
            'A_name': filename_A,
            'B_name': filename_B
        }

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))