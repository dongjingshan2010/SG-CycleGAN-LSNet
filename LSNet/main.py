import torch
from model_pytorch import Model
from torchvision import models
import argparse
from sklearn.metrics import confusion_matrix
from models import *
from models.mobile_vit import MobileViT
from models.sparse_mobile_vit import SparseMobileViT
from utils import progress_bar
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os
from torch import optim
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from torch.utils.data.dataset import ConcatDataset
from utilsfile.mask_utils import create_subgraph_mask2coords, create_rectangle_mask, create_rectangle_mask2coords, \
    create_bond_mask2coords
from utilsfile.public_utils import setup_device
from skimage.color import rgb2gray
from skimage.restoration import denoise_tv_chambolle
from skimage.feature import corner_peaks
from utilsfile.harris import CornerDetection
import time
from warmup_scheduler import GradualWarmupScheduler
import copy
from medical_image_loader import PairedMedicalImageDataset
import seaborn as sns
import pandas as pd
from itertools import cycle
import warnings

warnings.filterwarnings('ignore')


# 设置随机种子确保可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 计算所有评估指标的函数（与resnet_3t.py保持一致）
def calculate_all_metrics(all_labels, all_preds, all_probs, class_names):
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc, \
        precision_recall_curve

    accuracy = accuracy_score(all_labels, all_preds)

    if len(class_names) == 2:
        sensitivity = recall_score(all_labels, all_preds, pos_label=1, zero_division=0)
        specificity = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)

        fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        precision_vals, recall_vals, _ = precision_recall_curve(all_labels, all_probs[:, 1])
        pr_auc = auc(recall_vals, precision_vals)

        class_metrics = {}
        for i, class_name in enumerate(class_names):
            class_recall = recall_score(all_labels, all_preds, labels=[i], average=None)[0] if i in all_labels else 0
            class_precision = precision_score(all_labels, all_preds, labels=[i], average=None)[
                0] if i in all_preds else 0
            class_f1 = f1_score(all_labels, all_preds, labels=[i], average=None)[0] if (
                    i in all_labels and i in all_preds) else 0

            class_metrics[class_name] = {
                'recall': class_recall,
                'sensitivity': class_recall if i == 1 else None,
                'specificity': class_recall if i == 0 else None,
                'precision': class_precision,
                'f1_score': class_f1
            }

        metrics_dict = {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'class_metrics': class_metrics,
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'fpr': fpr,
            'tpr': tpr
        }
    else:
        # 多类别处理
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_preds, output_dict=True)

        # 计算多类别ROC AUC
        from sklearn.preprocessing import label_binarize
        y_test_bin = label_binarize(all_labels, classes=range(len(class_names)))

        # 计算每个类别的ROC曲线
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(class_names)):
            if len(np.unique(all_probs[:, i])) > 1:
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

        # 计算宏平均ROC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(class_names))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(class_names)):
            if i in tpr:
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(class_names)

        metrics_dict = {
            'accuracy': accuracy,
            'classification_report': report,
            'roc_auc_macro': auc(all_fpr, mean_tpr),
            'fpr': all_fpr,
            'tpr': mean_tpr,
            'roc_auc_per_class': roc_auc
        }

    return metrics_dict


# 计算平均ROC曲线和标准差区间
def compute_mean_roc_curves(all_roc_data, n_points=100):
    """
    计算三次实验的平均ROC曲线和标准差区间
    """
    mean_fpr = np.linspace(0, 1, n_points)

    tprs_interp = []
    for fpr, tpr in all_roc_data:
        if len(np.unique(fpr)) < 2:
            continue
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)

    if len(tprs_interp) == 0:
        return mean_fpr, np.zeros_like(mean_fpr), np.zeros_like(mean_fpr), []

    tprs_array = np.array(tprs_interp)
    mean_tpr = np.mean(tprs_array, axis=0)
    std_tpr = np.std(tprs_array, axis=0)

    return mean_fpr, mean_tpr, std_tpr, tprs_interp


# 绘制平均ROC曲线±标准差区间
def plot_mean_roc_curve_with_std(all_roc_data, all_auc_values, model_name="Model", save_path='./results'):
    """
    Plot mean ROC curve with standard deviation interval.
    """
    os.makedirs(save_path, exist_ok=True)

    mean_fpr, mean_tpr, std_tpr, tprs_interp = compute_mean_roc_curves(all_roc_data)

    mean_auc = np.mean(all_auc_values)
    std_auc = np.std(all_auc_values)

    plt.figure(figsize=(10, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(all_roc_data)))
    for i, (fpr, tpr) in enumerate(all_roc_data):
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], alpha=0.3, lw=2,
                 label=f'Experiment {i + 1} (AUC = {auc_value:.3f})' if i < 3 else f'Experiment {i + 1}')

    plt.plot(mean_fpr, mean_tpr, color='b', lw=3,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3,
                     label='±1 Standard Deviation')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title(f'Mean ROC Curve', fontsize=26)
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_mean_roc_curve.png'), dpi=150)
    plt.savefig(os.path.join(save_path, f'{model_name}_mean_roc_curve.pdf'))
    plt.show()

    print(f"\n{model_name} ROC Curve Statistics:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"AUC Values from Three Experiments: {[f'{auc_val:.4f}' for auc_val in all_auc_values]}")
    print(f"AUC Range: {min(all_auc_values):.4f} - {max(all_auc_values):.4f}")

    return mean_auc, std_auc


# 绘制训练和验证曲线的平均值和标准差
def plot_mean_training_curves(all_train_histories, save_path='./results'):
    """
    Plot mean training and validation curves with standard deviation.
    """
    os.makedirs(save_path, exist_ok=True)

    # 提取所有历史数据
    all_train_losses = [h['train_losses'] for h in all_train_histories]
    all_val_losses = [h['val_losses'] for h in all_train_histories]
    all_train_accs = [h['train_accs'] for h in all_train_histories]
    all_val_accs = [h['val_accs'] for h in all_train_histories]

    # 确保所有历史数据长度相同
    min_length = min(len(losses) for losses in all_train_losses)
    all_train_losses = [losses[:min_length] for losses in all_train_losses]
    all_val_losses = [losses[:min_length] for losses in all_val_losses]
    all_train_accs = [accs[:min_length] for accs in all_train_accs]
    all_val_accs = [accs[:min_length] for accs in all_val_accs]

    epochs = range(1, min_length + 1)

    # 转换为numpy数组进行统计计算
    train_losses_array = np.array(all_train_losses)
    val_losses_array = np.array(all_val_losses)
    train_accs_array = np.array(all_train_accs)
    val_accs_array = np.array(all_val_accs)

    # 计算均值和标准差
    mean_train_losses = np.mean(train_losses_array, axis=0)
    std_train_losses = np.std(train_losses_array, axis=0)

    mean_val_losses = np.mean(val_losses_array, axis=0)
    std_val_losses = np.std(val_losses_array, axis=0)

    mean_train_accs = np.mean(train_accs_array, axis=0)
    std_train_accs = np.std(train_accs_array, axis=0)

    mean_val_accs = np.mean(val_accs_array, axis=0)
    std_val_accs = np.std(val_accs_array, axis=0)

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制损失曲线
    axes[0, 0].plot(epochs, mean_train_losses, 'b-', label='Training Loss (Mean)', linewidth=2)
    axes[0, 0].fill_between(epochs,
                            mean_train_losses - std_train_losses,
                            mean_train_losses + std_train_losses,
                            color='blue', alpha=0.2)

    axes[0, 0].plot(epochs, mean_val_losses, 'r-', label='Validation Loss (Mean)', linewidth=2)
    axes[0, 0].fill_between(epochs,
                            mean_val_losses - std_val_losses,
                            mean_val_losses + std_val_losses,
                            color='red', alpha=0.2)

    axes[0, 0].set_title('Training and Validation Loss (Mean ± Std)', fontsize=14)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 绘制准确率曲线
    axes[0, 1].plot(epochs, mean_train_accs, 'b-', label='Training Accuracy (Mean)', linewidth=2)
    axes[0, 1].fill_between(epochs,
                            mean_train_accs - std_train_accs,
                            mean_train_accs + std_train_accs,
                            color='blue', alpha=0.2)

    axes[0, 1].plot(epochs, mean_val_accs, 'r-', label='Validation Accuracy (Mean)', linewidth=2)
    axes[0, 1].fill_between(epochs,
                            mean_val_accs - std_val_accs,
                            mean_val_accs + std_val_accs,
                            color='red', alpha=0.2)

    axes[0, 1].set_title('Training and Validation Accuracy (Mean ± Std)', fontsize=14)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 绘制最佳性能指标箱线图
    all_best_accs = [h['best_acc'] for h in all_train_histories]

    data_to_plot = [all_best_accs]
    bp = axes[1, 0].boxplot(data_to_plot, patch_artist=True,
                            labels=['Best Accuracy'])

    # 设置箱线图颜色
    colors = ['lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # 添加单个数据点
    for i, data in enumerate(data_to_plot, 1):
        y = data
        x = np.random.normal(i, 0.04, size=len(y))
        axes[1, 0].plot(x, y, 'r.', alpha=0.6)

    axes[1, 0].set_title('Distribution of Best Accuracy Across Three Experiments', fontsize=14)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 绘制最佳验证敏感度箱线图
    all_best_sensitivities = [h['best_sensitivity'] for h in all_train_histories]
    data_to_plot_sen = [all_best_sensitivities]
    bp_sen = axes[1, 1].boxplot(data_to_plot_sen, patch_artist=True,
                                labels=['Best Sensitivity'])

    colors_sen = ['lightgreen']
    for patch, color in zip(bp_sen['boxes'], colors_sen):
        patch.set_facecolor(color)

    for i, data in enumerate(data_to_plot_sen, 1):
        y = data
        x = np.random.normal(i, 0.04, size=len(y))
        axes[1, 1].plot(x, y, 'r.', alpha=0.6)

    axes[1, 1].set_title('Distribution of Best Sensitivity Across Three Experiments', fontsize=14)
    axes[1, 1].set_ylabel('Sensitivity', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Training Process Statistics from Three Independent Experiments', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mean_training_curves.png'), dpi=150)
    plt.savefig(os.path.join(save_path, 'mean_training_curves.pdf'))
    plt.show()


# 添加FLOPs计算工具
try:
    from thop import profile

    THOP_AVAILABLE = True
except ImportError:
    print("警告: 未找到thop库，无法计算FLOPs。请使用 'pip install thop' 安装")
    THOP_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count_table

    FVCORE_AVAILABLE = True
except ImportError:
    print("警告: 未找到fvcore库，无法使用FlopCountAnalysis。请使用 'pip install fvcore' 安装")
    FVCORE_AVAILABLE = False

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--aug', action='store_true', help='add image augumentations')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--bs', type=int, default='16')  # 64
parser.add_argument('--weight_decay', default=1e-6, type=float, help='SGD weight decay')
parser.add_argument('--data_address', default='../../data/Pretraining/', type=str)
parser.add_argument('--n_epochs', type=int, default='20')
parser.add_argument('--dim', type=int, default='256')
parser.add_argument('--sparsedim', type=int, default='128')
parser.add_argument('--imagesize', type=int, default='320')  # 288
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--tau', type=float, default=0.99)
parser.add_argument('--cos', default='True', action='store_true', help='Train with cosine annealing scheduling')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
size = int(args.imagesize)

dims = [128, 160, 192]
channels = [16, 32, 64, 64, 128, 128, 160, 160, 192, 192, 384]
heads = 4
vit = MobileViT((size, size), dims, channels, heads, num_classes=args.num_classes).to(device)

dims = [40, 48, 56]
channels = [16, 16, 24, 24, 32, 32, 48, 48, 64, 64, 128]
vit_sparse = SparseMobileViT((size, size), dims, channels, heads, num_classes=args.num_classes, expansion=2).to(device)


def calculate_flops(model, input_sizes=[(1, 3, 256, 256)]):
    """
    计算模型的FLOPs和参数量，支持多输入场景
    """
    model.eval()

    dummy_inputs = tuple(
        torch.randn(size).to(device)
        for size in input_sizes
    )

    print("=" * 60)
    print(f"计算模型的FLOPs和参数量（{len(input_sizes)}个输入）")
    print("=" * 60)

    if THOP_AVAILABLE:
        try:
            flops, params = profile(model, inputs=dummy_inputs, verbose=False)
            print(f"[thop] 参数量: {params:,} ({params / 1e6:.3f}M)")
            print(f"[thop] FLOPs: {flops:,} ({flops / 1e9:.3f}G)")
        except Exception as e:
            print(f"thop计算失败: {e}")

    if FVCORE_AVAILABLE:
        try:
            flops_analyzer = FlopCountAnalysis(model, dummy_inputs)
            flops_fvcore = flops_analyzer.total()
            print(f"[fvcore] FLOPs: {flops_fvcore:,} ({flops_fvcore / 1e9:.2f}G)")
            print(parameter_count_table(model))
        except Exception as e:
            print(f"fvcore计算失败: {e}")

    print("=" * 60)
    return flops if THOP_AVAILABLE else (flops_fvcore if FVCORE_AVAILABLE else None)


# 在模型初始化后立即计算FLOPs
vit_sparse.set_pretraining_mode(True)
calculate_flops(
    vit_sparse,
    input_sizes=[(1, 3, args.imagesize, args.imagesize)]
)

learner = Model(
    vit,
    vit_sparse,
    args.imagesize,
    hidden_layer='to_cls_token',
    projection_size=args.dim,
    sparse_projection_size=args.sparsedim,
    projection_hidden_size=4096
)

# 原有的参数量计算 (保留)
total_params = sum(p.numel() for p in vit.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in vit.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

# 原有的参数量计算 (保留)
total_params = sum(p.numel() for p in vit_sparse.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in vit_sparse.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

opt = torch.optim.Adam(learner.parameters(), lr=args.lr)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, int(args.n_epochs / 2) + 1)
scheduler = GradualWarmupScheduler(opt, multiplier=2, total_epoch=int(args.n_epochs / 2) + 1,
                                   after_scheduler=scheduler_cosine)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((args.imagesize, args.imagesize)),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((args.imagesize, args.imagesize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((args.imagesize, args.imagesize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

train_dataset = torchvision.datasets.ImageFolder(root='./data_split/synthetic_train',  #synthetic_train
                                                 transform=data_transforms['train'])
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=int(args.bs), shuffle=True, num_workers=0)

train_dataset_scd = torchvision.datasets.ImageFolder(root='./data_split/train', transform=data_transforms['train'])
trainloader_scd = torch.utils.data.DataLoader(train_dataset_scd, batch_size=int(args.bs), shuffle=True, num_workers=0,
                                              drop_last=True)

if args.cos:
    from warmup_scheduler import GradualWarmupScheduler

criterion = nn.CrossEntropyLoss().to(device)

#####################################################################

# 测试单个文件夹函数（与resnet_3t.py保持一致）
def test_single_folder(folder_path, transform, learner, criterion, device):
    """测试单个文件夹并返回评估指标（二分类，与resnet_3t.py保持一致）"""
    # 加载单个测试集
    testset = torchvision.datasets.ImageFolder(root=folder_path, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=True, num_workers=0)

    test_loss = 0
    correct = 0
    total = 0
    all_targets = []  # 收集所有真实标签
    all_probs = []  # 收集所有类别的概率
    all_preds = []  # 收集所有预测结果

    learner.eval()
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _, outputs = learner.sparse_encoder.net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            probs = torch.softmax(outputs, dim=1)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"文件夹 {folder_path} 测试时间: {elapsed_time:.6f}秒")

    # 转换为numpy数组
    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)
    all_probs_np = np.array(all_probs)

    # 获取类别名称
    class_names = testset.classes

    # 计算所有指标（与resnet_3t.py保持一致）
    metrics = calculate_all_metrics(all_targets_np, all_preds_np, all_probs_np, class_names)

    # 计算准确率（百分比）
    acc = 100. * correct / total

    # 输出当前文件夹的结果
    print(f"测试文件夹 {folder_path}:")
    print(f"  测试损失: {test_loss:.5f}")
    print(f"  准确率: {acc:.5f}% ({metrics['accuracy']:.5f})")
    print(f"  敏感度: {metrics['sensitivity']:.5f}")
    print(f"  特异度: {metrics['specificity']:.5f}")
    print(f"  精确率: {metrics['precision']:.5f}")
    print(f"  F1分数: {metrics['f1_score']:.5f}")
    print(f"  ROC AUC: {metrics['roc_auc']:.5f}")
    print(f"  PR AUC: {metrics['pr_auc']:.5f}")
    print(f"  混淆矩阵:\n{metrics['confusion_matrix']}")

    return {
        'loss': test_loss,
        'acc': acc,  # 百分比准确率
        'accuracy': metrics['accuracy'],  # 小数准确率
        'sensitivity': metrics['sensitivity'],
        'specificity': metrics['specificity'],
        'precision': metrics['precision'],
        'f1_score': metrics['f1_score'],
        'roc_auc': metrics['roc_auc'],
        'pr_auc': metrics['pr_auc'],
        'class_metrics': metrics['class_metrics'],
        'confusion_matrix': metrics['confusion_matrix'],
        'fpr': metrics['fpr'],
        'tpr': metrics['tpr'],
        'all_targets': all_targets_np,
        'all_probs': all_probs_np,
        'class_names': class_names
    }


def test_all_folders(epoch, base_path, transform, learner, criterion, device, num_folders=1):
    """测试所有文件夹并计算平均指标（更新版）"""
    # 存储所有文件夹的评估指标
    all_metrics = []

    # 循环测试文件夹
    for i in range(num_folders):
        folder_path = f'{base_path}/test{i}/'
        print(f"\n{'=' * 60}")
        print(f"开始测试文件夹: {folder_path}")
        print(f"{'=' * 60}")
        metrics = test_single_folder(folder_path, transform, learner, criterion, device)
        all_metrics.append(metrics)

    # 计算所有文件夹的平均指标
    avg_metrics = {
        'loss': np.mean([m['loss'] for m in all_metrics]),
        'acc': np.mean([m['acc'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in all_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in all_metrics])
    }

    # 合并所有样本用于整体ROC计算
    all_targets = np.concatenate([m['all_targets'] for m in all_metrics])
    all_probs = np.concatenate([m['all_probs'] for m in all_metrics])

    # 计算整体ROC曲线
    if len(np.unique(all_targets)) >= 2:
        fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
        overall_roc_auc = auc(fpr, tpr)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        overall_roc_auc = 0.5

    # 输出最终的平均指标
    print(f"\n{'=' * 60}")
    print("测试文件夹的平均指标:")
    print(f"{'=' * 60}")
    print(f"平均测试损失: {avg_metrics['loss']:.5f}")
    print(f"平均准确率: {avg_metrics['acc']:.5f}% ({avg_metrics['accuracy']:.5f})")
    print(f"平均敏感度: {avg_metrics['sensitivity']:.5f}")
    print(f"平均特异度: {avg_metrics['specificity']:.5f}")
    print(f"平均精确率: {avg_metrics['precision']:.5f}")
    print(f"平均F1分数: {avg_metrics['f1_score']:.5f}")
    print(f"平均ROC AUC: {avg_metrics['roc_auc']:.5f}")
    print(f"平均PR AUC: {avg_metrics['pr_auc']:.5f}")
    print(f"整体ROC AUC: {overall_roc_auc:.5f}")

    # 返回关键指标和ROC曲线数据
    return {
        'avg_metrics': avg_metrics,
        'overall_fpr': fpr,
        'overall_tpr': tpr,
        'overall_roc_auc': overall_roc_auc,
        'all_metrics': all_metrics
    }


def val_all_folders(epoch, base_path, transform, learner, criterion, device, num_folders=1):
    """验证所有文件夹并计算平均指标（更新版）"""
    # 存储所有文件夹的评估指标
    all_metrics = []

    # 循环验证文件夹
    for i in range(num_folders):
        folder_path = f'{base_path}/val{i}/'
        print(f"\n{'=' * 60}")
        print(f"开始验证文件夹: {folder_path}")
        print(f"{'=' * 60}")
        metrics = test_single_folder(folder_path, transform, learner, criterion, device)
        all_metrics.append(metrics)

    # 计算所有文件夹的平均指标
    avg_metrics = {
        'loss': np.mean([m['loss'] for m in all_metrics]),
        'acc': np.mean([m['acc'] for m in all_metrics]),
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'sensitivity': np.mean([m['sensitivity'] for m in all_metrics]),
        'specificity': np.mean([m['specificity'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'f1_score': np.mean([m['f1_score'] for m in all_metrics]),
        'roc_auc': np.mean([m['roc_auc'] for m in all_metrics]),
        'pr_auc': np.mean([m['pr_auc'] for m in all_metrics])
    }

    print(f"\n{'=' * 60}")
    print("验证文件夹的平均指标:")
    print(f"{'=' * 60}")
    print(f"平均验证损失: {avg_metrics['loss']:.5f}")
    print(f"平均准确率: {avg_metrics['acc']:.5f}% ({avg_metrics['accuracy']:.5f})")
    print(f"平均敏感度: {avg_metrics['sensitivity']:.5f}")
    print(f"平均特异度: {avg_metrics['specificity']:.5f}")
    print(f"平均精确率: {avg_metrics['precision']:.5f}")
    print(f"平均F1分数: {avg_metrics['f1_score']:.5f}")
    print(f"平均ROC AUC: {avg_metrics['roc_auc']:.5f}")
    print(f"平均PR AUC: {avg_metrics['pr_auc']:.5f}")

    # 返回平均敏感度和平均AUC作为验证指标
    return avg_metrics['sensitivity'], avg_metrics['specificity'], avg_metrics['roc_auc']


learner.online_encoder.net.set_pretraining_mode(False)

# 第一阶段训练
best_sen = 0  # 最佳敏感度
best_spec = 0  # 最佳特异度
accumulation = 4
best_acc_global = 0  # best test accuracy
best_roc_global = 0  # best test roc
biaozhi = 0
total_batch = 0

# 存储第一阶段训练历史
first_stage_train_history = {
    'train_losses': [],
    'val_losses': [],
    'train_accs': [],
    'val_accs': [],
    'val_sensitivities': [],
    'val_specificities': [],
    'best_sensitivity': 0.0,
    'best_specificity': 0.0,
    'best_acc': 0.0,
    'best_epoch': 0
}

for interation in range(args.n_epochs):  # args.n_epochs
    print('interation=%d' % interation)
    torch.cuda.synchronize()
    start = time.time()
    train_loss = 0
    learner.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        img, label = inputs.to(device), targets.to(device)

        loss = learner(img, label)

        train_loss += loss.item()
        loss.backward()

        if ((batch_idx + 1) % accumulation) == 0:
            opt.step()
            opt.zero_grad()
        total_batch = batch_idx

    train_loss = train_loss / (total_batch + 1)
    first_stage_train_history['train_losses'].append(train_loss)

    content = time.ctime() + ' ' + f'Epoch {interation}, Train loss: {train_loss:.4f}, lr: {opt.param_groups[0]["lr"]:.5f}'
    print(content)

    if interation >= args.n_epochs-3:
        base_path = './data_split'
        val_sen, val_spec, val_roc = val_all_folders(interation, base_path, data_transforms['val'], learner, criterion, device)

        # 更新验证指标
        first_stage_train_history['val_losses'].append(train_loss)  # 简化，实际应该计算验证损失
        first_stage_train_history['val_accs'].append(val_roc)  # 使用AUC作为准确率代理
        first_stage_train_history['val_sensitivities'].append(val_sen)
        first_stage_train_history['val_specificities'].append(val_spec)

        # 根据敏感度阈值决定保存模型的策略
        save_model = False
        if val_sen < 0.95:
            # 敏感度小于0.95时，保存敏感度最大的模型
            if val_sen > best_sen:
                best_sen = val_sen
                first_stage_train_history['best_sensitivity'] = val_sen
                first_stage_train_history['best_specificity'] = val_spec
                first_stage_train_history['best_acc'] = val_roc
                first_stage_train_history['best_epoch'] = interation + 1
                torch.save(learner.state_dict(), './improved-net.pth')
                save_model = True
                print(f"🚀 第一阶段新的最佳模型（基于敏感度）! 验证敏感度: {best_sen:.4f}, 特异度: {val_spec:.4f}, AUC: {val_roc:.4f}")
        else:
            # 敏感度大于等于0.95时，保存特异度最大的模型
            if val_spec > best_spec:
                best_spec = val_spec
                first_stage_train_history['best_sensitivity'] = val_sen
                first_stage_train_history['best_specificity'] = val_spec
                first_stage_train_history['best_acc'] = val_roc
                first_stage_train_history['best_epoch'] = interation + 1
                torch.save(learner.state_dict(), './improved-net.pth')
                save_model = True
                print(f"🚀 第一阶段新的最佳模型（基于特异度）! 验证敏感度: {val_sen:.4f}, 特异度: {best_spec:.4f}, AUC: {val_roc:.4f}")

    scheduler.step(interation)
    torch.cuda.synchronize()


# 计算第二阶段训练集的类别权重
def calculate_class_weights(dataset, num_classes):
    """计算类别权重以处理类别不平衡"""
    # 统计每个类别的样本数
    class_counts = torch.zeros(num_classes)
    for _, label in dataset:
        class_counts[label] += 1

    # 使用逆频率计算权重
    total_samples = len(dataset)
    weights = total_samples / (num_classes * class_counts)

    # 确保没有无限大的权重
    weights = torch.clamp(weights, min=0.1, max=10.0)

    return weights.tolist()

# 第二阶段：三次独立训练验证测试
print("\n" + "=" * 80)
print("开始第二阶段：三次独立训练验证测试")
print("=" * 80)

# 存储所有实验结果的列表
all_experiment_results = []
all_train_histories = []
all_test_results = []  # 存储每次实验的详细测试结果
all_roc_data = []
all_auc_values = []

# 创建结果目录
os.makedirs('./results', exist_ok=True)

# 进行三次独立实验
num_experiments = 3
for exp_id in range(num_experiments):
    print(f"\n{'=' * 60}")
    print(f"第二阶段实验 #{exp_id + 1}/{num_experiments}")
    print(f"{'=' * 60}")

    # 设置不同的随机种子以确保独立性
    set_seed(seed=42 + exp_id * 100)

    # 重新加载第一阶段训练后的模型
    state_dict = torch.load('./improved-net.pth')
    missing_keys, unexpected_keys = learner.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"警告: 加载状态字典时缺失以下键: {missing_keys}")
    if unexpected_keys:
        print(f"警告: 加载状态字典时出现意外键: {unexpected_keys}")

    # 计算并设置类别权重（针对非均衡样本训练阶段）
    class_weights = calculate_class_weights(train_dataset_scd, args.num_classes)
    print(f"实验 #{exp_id + 1} 类别权重: {class_weights}")
    learner.set_class_weights(class_weights)

    # 重新初始化优化器和调度器
    opt_scd = torch.optim.Adam(learner.parameters(), lr=args.lr * 0.1)
    scheduler_cosine_scd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_scd, int(args.n_epochs / 2) + 1)
    scheduler_scd = GradualWarmupScheduler(opt_scd, multiplier=2, total_epoch=int(args.n_epochs / 2) + 1,
                                           after_scheduler=scheduler_cosine_scd)

    # 训练循环
    best_sen = 0  # 最佳敏感度
    best_spec = 0  # 最佳特异度
    best_roc = 0
    best_model_path = f'./results/stage2_best_model_exp{exp_id + 1}.pth'

    # 存储第二阶段训练历史
    stage2_train_history = {
        'train_losses': [],
        'val_losses': [],
        'train_accs': [],
        'val_accs': [],
        'val_sensitivities': [],
        'val_specificities': [],
        'best_sensitivity': 0.0,
        'best_specificity': 0.0,
        'best_acc': 0.0,
        'best_epoch': 0
    }

    for interation in range(args.n_epochs):
        print('interation=%d' % interation)
        torch.cuda.synchronize()
        train_loss = 0
        learner.train()

        for batch_idx, (inputs, targets) in enumerate(trainloader_scd):
            inputs, targets = inputs.to(device), targets.to(device)

            img, label = inputs.to(device), targets.to(device)

            loss = learner(img, label)

            train_loss += loss.item()
            loss.backward()

            if ((batch_idx + 1) % accumulation) == 0:
                opt_scd.step()
                opt_scd.zero_grad()

        train_loss = train_loss / (len(trainloader_scd) + 1e-5)
        stage2_train_history['train_losses'].append(train_loss)

        content = time.ctime() + ' ' + f'Epoch {interation}, Train loss: {train_loss:.4f}, lr: {opt_scd.param_groups[0]["lr"]:.5f}'
        print(content)

        # 验证
        if interation >= args.n_epochs - 5:
            base_path = './data_split'
            val_sen, val_spec, val_roc = val_all_folders(interation, base_path, data_transforms['test'], learner, criterion, device)

            # 更新验证指标
            stage2_train_history['val_losses'].append(train_loss)  # 简化
            stage2_train_history['val_accs'].append(val_roc)
            stage2_train_history['val_sensitivities'].append(val_sen)
            stage2_train_history['val_specificities'].append(val_spec)

            # 根据敏感度阈值决定保存模型的策略
            save_model = False
            if val_sen < 0.95:
                # 敏感度小于0.95时，保存敏感度最大的模型
                if val_sen > best_sen:
                    best_sen = val_sen
                    best_roc = val_roc
                    stage2_train_history['best_sensitivity'] = val_sen
                    stage2_train_history['best_specificity'] = val_spec
                    stage2_train_history['best_acc'] = val_roc
                    stage2_train_history['best_epoch'] = interation + 1
                    torch.save(learner.state_dict(), best_model_path)
                    save_model = True
                    print(f"🚀 实验 #{exp_id + 1} 新的最佳模型（基于敏感度）! 验证敏感度: {best_sen:.4f}, 特异度: {val_spec:.4f}, AUC: {val_roc:.4f}")
            else:
                # 敏感度大于等于0.95时，保存特异度最大的模型
                if val_spec > best_spec:
                    best_sen = val_sen
                    best_spec = val_spec
                    best_roc = val_roc
                    stage2_train_history['best_sensitivity'] = val_sen
                    stage2_train_history['best_specificity'] = val_spec
                    stage2_train_history['best_acc'] = val_roc
                    stage2_train_history['best_epoch'] = interation + 1
                    torch.save(learner.state_dict(), best_model_path)
                    save_model = True
                    print(f"🚀 实验 #{exp_id + 1} 新的最佳模型（基于特异度）! 验证敏感度: {val_sen:.4f}, 特异度: {best_spec:.4f}, AUC: {val_roc:.4f}")

        scheduler_scd.step(interation)

    # 加载最佳模型进行测试
    learner.load_state_dict(torch.load(best_model_path))

    # 在测试集上测试最佳模型
    print(f"\n实验 #{exp_id + 1} 在测试集上评估...")
    base_path = './data_split'
    test_result = test_all_folders(
        interation, base_path, data_transforms['test'], learner, criterion, device
    )

    # 存储ROC曲线数据
    all_roc_data.append((test_result['overall_fpr'], test_result['overall_tpr']))
    all_auc_values.append(test_result['overall_roc_auc'])

    # 收集实验数据
    experiment_data = {
        'experiment_id': exp_id + 1,
        'best_sensitivity': stage2_train_history['best_sensitivity'],
        'best_specificity': stage2_train_history['best_specificity'],
        'best_roc_auc': best_roc,
        'test_metrics': test_result['avg_metrics'],
        'overall_roc_auc': test_result['overall_roc_auc'],
        'train_history': stage2_train_history
    }

    all_experiment_results.append(experiment_data)
    all_train_histories.append(stage2_train_history)
    all_test_results.append(test_result)

    # 打印本次实验的结果
    print(f"\n实验 #{exp_id + 1} 结果:")
    print(f"- 最佳验证敏感度: {stage2_train_history['best_sensitivity']:.4f}")
    print(f"- 最佳验证特异度: {stage2_train_history['best_specificity']:.4f}")
    print(f"- 最佳验证AUC: {best_roc:.4f}")
    print(f"- 测试集准确率: {test_result['avg_metrics']['accuracy']:.4f}")
    print(f"- 测试集敏感度: {test_result['avg_metrics']['sensitivity']:.4f}")
    print(f"- 测试集特异度: {test_result['avg_metrics']['specificity']:.4f}")
    print(f"- 测试集精确率: {test_result['avg_metrics']['precision']:.4f}")
    print(f"- 测试集F1分数: {test_result['avg_metrics']['f1_score']:.4f}")
    print(f"- 测试集ROC AUC: {test_result['avg_metrics']['roc_auc']:.4f}")
    print(f"- 整体ROC AUC: {test_result['overall_roc_auc']:.4f}")

# 绘制平均ROC曲线±标准差区间
print(f"\n{'=' * 80}")
print("绘制平均ROC曲线和标准差区间")
print(f"{'=' * 80}")

if all_roc_data and len(all_roc_data) == num_experiments:
    mean_auc, std_auc = plot_mean_roc_curve_with_std(
        all_roc_data, all_auc_values,
        model_name="MobileViT_Stage2",
        save_path='./results'
    )
else:
    print("警告: 无法计算平均ROC曲线，ROC数据不完整")

# 生成综合报告（与resnet_3t.py类似）
print(f"\n{'=' * 80}")
print("三次独立实验综合报告")
print(f"{'=' * 80}")

# 计算各项指标的平均值和标准差
metrics_to_analyze = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc']

# 提取每次实验的测试指标
all_test_metrics = []
for exp_result in all_experiment_results:
    test_metrics = exp_result['test_metrics']
    all_test_metrics.append(test_metrics)

metrics_summary = {}
for metric in metrics_to_analyze:
    values = [test_metrics[metric] for test_metrics in all_test_metrics]
    metrics_summary[metric] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'values': values
    }

# 计算最佳验证指标的平均值和标准差
best_sensitivities = [exp['best_sensitivity'] for exp in all_experiment_results]
best_specificities = [exp['best_specificity'] for exp in all_experiment_results]
best_roc_aucs = [exp['best_roc_auc'] for exp in all_experiment_results]

best_metrics_summary = {
    'best_sensitivity': {
        'mean': np.mean(best_sensitivities),
        'std': np.std(best_sensitivities),
        'min': np.min(best_sensitivities),
        'max': np.max(best_sensitivities),
        'values': best_sensitivities
    },
    'best_specificity': {
        'mean': np.mean(best_specificities),
        'std': np.std(best_specificities),
        'min': np.min(best_specificities),
        'max': np.max(best_specificities),
        'values': best_specificities
    },
    'best_roc_auc': {
        'mean': np.mean(best_roc_aucs),
        'std': np.std(best_roc_aucs),
        'min': np.min(best_roc_aucs),
        'max': np.max(best_roc_aucs),
        'values': best_roc_aucs
    }
}

# 打印报告
print(f"\n模型: MobileViT (第二阶段)")
print(f"实验次数: {num_experiments}")
print(f"训练轮数: {args.n_epochs}")

print(f"\n最佳验证性能统计 (均值 ± 标准差):")
print(f"  最佳验证敏感度: {best_metrics_summary['best_sensitivity']['mean']:.4f} ± {best_metrics_summary['best_sensitivity']['std']:.4f}")
print(f"  最佳验证特异度: {best_metrics_summary['best_specificity']['mean']:.4f} ± {best_metrics_summary['best_specificity']['std']:.4f}")
print(f"  最佳验证AUC: {best_metrics_summary['best_roc_auc']['mean']:.4f} ± {best_metrics_summary['best_roc_auc']['std']:.4f}")

print(f"\n测试集性能统计 (均值 ± 标准差):")
metric_names = {
    'accuracy': '准确率',
    'sensitivity': '敏感度',
    'specificity': '特异度',
    'precision': '精确率',
    'f1_score': 'F1分数',
    'roc_auc': 'ROC AUC'
}

for metric, stats in metrics_summary.items():
    print(f"  {metric_names.get(metric, metric)}: "
          f"{stats['mean']:.4f} ± {stats['std']:.4f} "
          f"(范围: {stats['min']:.4f} - {stats['max']:.4f})")

# 保存所有实验结果到CSV文件
print(f"\n{'=' * 80}")
print("保存所有实验结果")
print(f"{'=' * 80}")

# 创建结果DataFrame
results_df = pd.DataFrame({
    '实验编号': [f'实验{i + 1}' for i in range(num_experiments)],
    '最佳验证敏感度': [exp['best_sensitivity'] for exp in all_experiment_results],
    '最佳验证特异度': [exp['best_specificity'] for exp in all_experiment_results],
    '最佳验证AUC': [exp['best_roc_auc'] for exp in all_experiment_results],
    '测试准确率': [metrics_summary['accuracy']['values'][i] for i in range(num_experiments)],
    '测试敏感度': [metrics_summary['sensitivity']['values'][i] for i in range(num_experiments)],
    '测试特异度': [metrics_summary['specificity']['values'][i] for i in range(num_experiments)],
    '测试精确率': [metrics_summary['precision']['values'][i] for i in range(num_experiments)],
    '测试F1分数': [metrics_summary['f1_score']['values'][i] for i in range(num_experiments)],
    '测试ROC_AUC': [metrics_summary['roc_auc']['values'][i] for i in range(num_experiments)]
})

# 添加统计行
stats_row = {
    '实验编号': '统计量',
    '最佳验证敏感度': f"{best_metrics_summary['best_sensitivity']['mean']:.4f} ± {best_metrics_summary['best_sensitivity']['std']:.4f}",
    '最佳验证特异度': f"{best_metrics_summary['best_specificity']['mean']:.4f} ± {best_metrics_summary['best_specificity']['std']:.4f}",
    '最佳验证AUC': f"{best_metrics_summary['best_roc_auc']['mean']:.4f} ± {best_metrics_summary['best_roc_auc']['std']:.4f}",
    '测试准确率': f"{metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}",
    '测试敏感度': f"{metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}",
    '测试特异度': f"{metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}",
    '测试精确率': f"{metrics_summary['precision']['mean']:.4f} ± {metrics_summary['precision']['std']:.4f}",
    '测试F1分数': f"{metrics_summary['f1_score']['mean']:.4f} ± {metrics_summary['f1_score']['std']:.4f}",
    '测试ROC_AUC': f"{metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}"
}

results_df = pd.concat([results_df, pd.DataFrame([stats_row])], ignore_index=True)

# 保存到CSV
results_csv_path = './results/stage2_three_experiments_summary.csv'
results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
print(f"实验结果已保存到: {results_csv_path}")

# 保存详细的测试结果
detailed_results = []
for i, test_result in enumerate(all_test_results):
    for j, folder_metrics in enumerate(test_result['all_metrics']):
        detailed_results.append({
            '实验编号': f'实验{i + 1}',
            '文件夹': f'test{j}',
            '准确率': folder_metrics['accuracy'],
            '敏感度': folder_metrics['sensitivity'],
            '特异度': folder_metrics['specificity'],
            '精确率': folder_metrics['precision'],
            'F1分数': folder_metrics['f1_score'],
            'ROC_AUC': folder_metrics['roc_auc'],
            'PR_AUC': folder_metrics['pr_auc']
        })

detailed_df = pd.DataFrame(detailed_results)
detailed_csv_path = './results/stage2_detailed_results.csv'
detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
print(f"详细测试结果已保存到: {detailed_csv_path}")

# 保存完整的实验检查点
final_checkpoint = {
    'all_experiment_results': all_experiment_results,
    'all_test_metrics': all_test_metrics,
    'all_roc_data': all_roc_data,
    'all_auc_values': all_auc_values,
    'metrics_summary': metrics_summary,
    'best_metrics_summary': best_metrics_summary,
    'num_experiments': num_experiments,
    'num_epochs': args.n_epochs
}

checkpoint_path = './results/stage2_final_experiment_checkpoint.pth'
torch.save(final_checkpoint, checkpoint_path)
print(f"完整实验检查点已保存到: {checkpoint_path}")

print(f"\n{'=' * 80}")
print("第二阶段三次独立实验完成!")
print(f"{'=' * 80}")