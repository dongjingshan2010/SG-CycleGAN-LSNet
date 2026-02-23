import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights, ResNet34_Weights, EfficientNet_B0_Weights
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, \
    recall_score, accuracy_score
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 数据路径
data_dir = './data/data2'
train_dir = os.path.join(data_dir, 'fake_UNIT')
val_dir = os.path.join(data_dir, 'val0')
test_dir = os.path.join(data_dir, 'test0')

# 检查数据路径是否存在
for path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(path):
        print(f"警告: 路径 {path} 不存在!")
        os.makedirs(path, exist_ok=True)
        print(f"已创建路径: {path}")

# 数据预处理和增强
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


# 加载数据集
def load_datasets():
    image_datasets = {}

    try:
        image_datasets['train'] = datasets.ImageFolder(train_dir, data_transforms['train'])
        image_datasets['val'] = datasets.ImageFolder(val_dir, data_transforms['val'])
        image_datasets['test'] = datasets.ImageFolder(test_dir, data_transforms['test'])
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        print("请确保数据目录结构正确，每个子文件夹内都有图片文件")
        exit(1)

    # 获取类别名称
    class_names = image_datasets['train'].classes
    print(f"发现 {len(class_names)} 个类别: {class_names}")

    # 创建数据加载器
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    print(f"数据集大小: 训练集={dataset_sizes['train']}, 验证集={dataset_sizes['val']}, 测试集={dataset_sizes['test']}")

    return dataloaders, class_names, dataset_sizes, image_datasets


# MobileNetV2分类器（非常轻量）
class MobileNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetV2Classifier, self).__init__()

        # 加载预训练的MobileNetV2模型
        self.model = models.MobileNetV2(width_mult=0.45)
        # 替换最后的分类层
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# 模型工厂函数
def create_model(model_type='mobilenet_v2', num_classes=2, pretrained=True):
    """
    创建指定类型的模型
    """
    if model_type == 'mobilenet_v2':
        model = MobileNetV2Classifier(num_classes=num_classes, pretrained=pretrained)
        print(f"创建MobileNetV2模型，参数量约3.4M")
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

    # 计算并打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    return model


# 加权交叉熵损失计算函数
def get_weighted_cross_entropy(class_counts, device='cuda'):
    total_samples = sum(class_counts)
    num_classes = len(class_counts)

    class_weights = [total_samples / (num_classes * count) for count in class_counts]
    weights = torch.FloatTensor(class_weights).to(device)
    print(f"类别权重: {weights.cpu().numpy()}")

    return nn.CrossEntropyLoss(weight=weights)


# 计算所有评估指标的函数
def calculate_all_metrics(all_labels, all_preds, all_probs, class_names):
    accuracy = accuracy_score(all_labels, all_preds)

    if len(class_names) == 2:
        sensitivity = recall_score(all_labels, all_preds, pos_label=1)
        specificity = recall_score(all_labels, all_preds, pos_label=0)
        precision = precision_score(all_labels, all_preds, pos_label=1, zero_division=0)
        f1 = f1_score(all_labels, all_preds, pos_label=1)

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
        # 多类别处理（省略，保持原样）
        pass

    return metrics_dict


# 训练函数（以敏感度作为最佳模型判定标准）
def train_model_with_sensitivity_criterion(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
                                           class_names, model_type, num_epochs=7, experiment_id=0):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_sensitivities = []

    best_model_wts = None
    best_sensitivity = 0.0
    best_acc = 0.0
    best_epoch = 0
    best_val_metrics = None

    print(f'\n实验 #{experiment_id + 1} 开始训练...')

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []
            all_probs = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().detach().numpy())

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            all_labels = np.array(all_labels)
            all_preds = np.array(all_preds)
            all_probs = np.array(all_probs)

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

                val_metrics = calculate_all_metrics(all_labels, all_preds, all_probs, class_names)

                if 'sensitivity' in val_metrics:
                    current_sensitivity = val_metrics['sensitivity']
                else:
                    current_sensitivity = val_metrics['macro_recall']

                val_sensitivities.append(current_sensitivity)

                if current_sensitivity > best_sensitivity:
                    best_sensitivity = current_sensitivity
                    best_acc = epoch_acc.item()
                    best_epoch = epoch + 1
                    best_model_wts = model.state_dict().copy()
                    best_val_metrics = val_metrics
                    print(f"🚀 新的最佳模型! 敏感度: {best_sensitivity:.4f}, 准确率: {best_acc:.4f} (第{best_epoch}轮)")

                    best_model_weights_path = f'./results/{model_type}_best_weights.pth'
                    torch.save(model.state_dict(), best_model_weights_path)
                    print(f"最佳模型权重已保存到: {best_model_weights_path}")

        print(f'实验 #{experiment_id + 1} Epoch {epoch + 1}/{num_epochs} - '
              f'训练损失: {train_losses[-1]:.4f}, 验证损失: {val_losses[-1]:.4f}, '
              f'验证敏感度: {current_sensitivity:.4f}')

    print(f'实验 #{experiment_id + 1} 训练完成! '
          f'最佳验证敏感度: {best_sensitivity:.4f} (第{best_epoch}轮)')

    return {
        'model_state': best_model_wts,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'val_sensitivities': val_sensitivities,
        'best_sensitivity': best_sensitivity,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'best_val_metrics': best_val_metrics
    }


# 测试函数
def test_model_comprehensive(model, test_loader, class_names, phase='测试'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    test_metrics = calculate_all_metrics(all_labels, all_preds, all_probs, class_names)

    return test_metrics


# 计算平均ROC曲线和标准差区间
def compute_mean_roc_curves(all_roc_data, n_points=100):
    """
    计算三次实验的平均ROC曲线和标准差区间

    参数:
    - all_roc_data: 列表，包含每次实验的(fpr, tpr)元组
    - n_points: 插值点数

    返回:
    - mean_fpr: 平均FPR网格
    - mean_tpr: 平均TPR
    - std_tpr: TPR的标准差
    - tprs_interp: 插值后的所有TPR曲线
    """
    # 创建共同的FPR网格
    mean_fpr = np.linspace(0, 1, n_points)

    # 对所有TPR曲线进行插值
    tprs_interp = []
    for fpr, tpr in all_roc_data:
        # 避免重复的FPR值
        if len(np.unique(fpr)) < 2:
            continue
        # 插值到共同的FPR网格
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0  # 确保从0开始
        tprs_interp.append(tpr_interp)

    # 计算平均值和标准差
    tprs_array = np.array(tprs_interp)
    mean_tpr = np.mean(tprs_array, axis=0)
    std_tpr = np.std(tprs_array, axis=0)

    return mean_fpr, mean_tpr, std_tpr, tprs_interp


# 绘制平均ROC曲线±标准差区间
def plot_mean_roc_curve_with_std(all_roc_data, all_auc_values, model_name="Model", save_path='./results'):
    """
    Plot mean ROC curve with standard deviation interval.

    Parameters:
    - all_roc_data: List of tuples, each containing (fpr, tpr) for one experiment
    - all_auc_values: List of AUC values for each experiment
    - model_name: Name of the model
    - save_path: Path to save the figure
    """
    os.makedirs(save_path, exist_ok=True)

    # Compute mean ROC curve and standard deviation
    mean_fpr, mean_tpr, std_tpr, tprs_interp = compute_mean_roc_curves(all_roc_data)

    # Compute mean AUC and standard deviation
    mean_auc = np.mean(all_auc_values)
    std_auc = np.std(all_auc_values)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot all three ROC curves (semi-transparent)
    colors = plt.cm.Set1(np.linspace(0, 1, len(all_roc_data)))
    for i, (fpr, tpr) in enumerate(all_roc_data):
        auc_value = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], alpha=0.3, lw=2,
                 label=f'Experiment {i + 1} (AUC = {auc_value:.3f})' if i < 3 else f'Experiment {i + 1}')

    # Plot mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='b', lw=3,
             label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    # Plot standard deviation interval (shaded area)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='grey', alpha=0.3,
                     label='±1 Standard Deviation')

    # Plot random guess line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')

    # Configure plot properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.title(f'Mean ROC Curve', fontsize=26)
    plt.legend(loc="lower right", fontsize=20)
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{model_name}_mean_roc_curve.png'), dpi=150)
    plt.savefig(os.path.join(save_path, f'{model_name}_mean_roc_curve.pdf'))
    plt.show()

    # Print statistics
    print(f"\n{model_name} ROC Curve Statistics:")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"AUC Values from Three Experiments: {[f'{auc_val:.4f}' for auc_val in all_auc_values]}")
    print(f"AUC Range: {min(all_auc_values):.4f} - {max(all_auc_values):.4f}")

    return mean_auc, std_auc


# 绘制训练和验证曲线的平均值和标准差
def plot_mean_training_curves(all_train_histories, save_path='./results'):
    """
    Plot mean training and validation curves with standard deviation.

    Parameters:
    - all_train_histories: List, containing training histories from each experiment
    - save_path: Path to save the figure
    """
    os.makedirs(save_path, exist_ok=True)

    # Extract all history data
    all_train_losses = [h['train_losses'] for h in all_train_histories]
    all_val_losses = [h['val_losses'] for h in all_train_histories]
    all_train_accs = [h['train_accs'] for h in all_train_histories]
    all_val_accs = [h['val_accs'] for h in all_train_histories]
    all_val_sensitivities = [h['val_sensitivities'] for h in all_train_histories]

    # Ensure all histories have the same length
    min_length = min(len(losses) for losses in all_train_losses)
    all_train_losses = [losses[:min_length] for losses in all_train_losses]
    all_val_losses = [losses[:min_length] for losses in all_val_losses]
    all_train_accs = [accs[:min_length] for accs in all_train_accs]
    all_val_accs = [accs[:min_length] for accs in all_val_accs]
    all_val_sensitivities = [sens[:min_length] for sens in all_val_sensitivities]

    epochs = range(1, min_length + 1)

    # Convert to numpy arrays for statistical calculation
    train_losses_array = np.array(all_train_losses)
    val_losses_array = np.array(all_val_losses)
    train_accs_array = np.array(all_train_accs)
    val_accs_array = np.array(all_val_accs)
    val_sensitivities_array = np.array(all_val_sensitivities)

    # Calculate mean and standard deviation
    mean_train_losses = np.mean(train_losses_array, axis=0)
    std_train_losses = np.std(train_losses_array, axis=0)

    mean_val_losses = np.mean(val_losses_array, axis=0)
    std_val_losses = np.std(val_losses_array, axis=0)

    mean_train_accs = np.mean(train_accs_array, axis=0)
    std_train_accs = np.std(train_accs_array, axis=0)

    mean_val_accs = np.mean(val_accs_array, axis=0)
    std_val_accs = np.std(val_accs_array, axis=0)

    mean_val_sensitivities = np.mean(val_sensitivities_array, axis=0)
    std_val_sensitivities = np.std(val_sensitivities_array, axis=0)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot loss curves
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

    # Plot accuracy curves
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

    # Plot sensitivity curves
    axes[1, 0].plot(epochs, mean_val_sensitivities, 'g-', label='Validation Sensitivity (Mean)', linewidth=2)
    axes[1, 0].fill_between(epochs,
                            mean_val_sensitivities - std_val_sensitivities,
                            mean_val_sensitivities + std_val_sensitivities,
                            color='green', alpha=0.2)

    axes[1, 0].set_title('Validation Sensitivity (Mean ± Std)', fontsize=14)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Sensitivity', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot boxplot for best performance metrics
    all_best_sensitivities = [h['best_sensitivity'] for h in all_train_histories]
    all_best_accs = [h['best_acc'] for h in all_train_histories]

    data_to_plot = [all_best_sensitivities, all_best_accs]
    bp = axes[1, 1].boxplot(data_to_plot, patch_artist=True,
                            labels=['Best Sensitivity', 'Best Accuracy'])

    # Set boxplot colors
    colors = ['lightgreen', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add individual data points
    for i, data in enumerate(data_to_plot, 1):
        y = data
        x = np.random.normal(i, 0.04, size=len(y))
        axes[1, 1].plot(x, y, 'r.', alpha=0.6)

    axes[1, 1].set_title('Distribution of Best Performance Metrics Across Three Experiments', fontsize=14)
    axes[1, 1].set_ylabel('Value', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Training Process Statistics from Three Independent Experiments', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'mean_training_curves.png'), dpi=150)
    plt.savefig(os.path.join(save_path, 'mean_training_curves.pdf'))
    plt.show()


# 主函数 - 运行三次独立实验
def main():
    # 创建结果目录
    os.makedirs('./results', exist_ok=True)

    print("=" * 80)
    print("开始三次独立训练、验证和测试实验")
    print("=" * 80)

    # 1. 加载数据（数据划分固定）
    print("\n1. 加载数据集...")
    dataloaders, class_names, dataset_sizes, image_datasets = load_datasets()

    # 2. 实验配置
    model_type = 'mobilenet_v2'
    num_experiments = 3
    num_epochs = 7

    # 存储所有实验结果的列表
    all_experiment_results = []
    all_train_histories = []
    all_test_metrics = []
    all_roc_data = []  # 存储每次实验的ROC曲线数据
    all_auc_values = []  # 存储每次实验的AUC值

    # 3. 运行三次独立实验
    for exp_id in range(num_experiments):
        print(f"\n{'=' * 60}")
        print(f"实验 #{exp_id + 1}/{num_experiments}")
        print(f"{'=' * 60}")

        # 设置不同的随机种子以确保独立性
        set_seed(seed=42+ exp_id * 100)   #

        # 创建新的模型实例
        print(f"\n创建{model_type}模型...")
        model = create_model(model_type=model_type, num_classes=len(class_names), pretrained=False)
        model = model.to(device)

        # 定义损失函数和优化器
        class_counts = [0] * len(class_names)
        for _, label in image_datasets['train'].samples:
            class_counts[label] += 1

        print(f"训练集类别分布: {dict(zip(class_names, class_counts))}")

        criterion = get_weighted_cross_entropy(class_counts, device)
        print("使用加权交叉熵损失")

        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 训练模型
        train_result = train_model_with_sensitivity_criterion(
            model, criterion, optimizer, scheduler, dataloaders, dataset_sizes,
            class_names, model_type, num_epochs=num_epochs, experiment_id=exp_id
        )

        # 保存最佳模型权重
        best_model_weights_path = f'./results/{model_type}_best_weights.pth'

        # 重新创建模型并加载最佳权重进行测试
        model_best = create_model(model_type=model_type, num_classes=len(class_names), pretrained=False)
        model_best.load_state_dict(torch.load(best_model_weights_path))
        model_best = model_best.to(device)
        model_best.eval()

        # 在测试集上测试最佳模型
        print(f"\n实验 #{exp_id + 1} 在测试集上评估...")
        test_metrics = test_model_comprehensive(model_best, dataloaders['test'], class_names,
                                                phase=f'实验{exp_id + 1}测试')

        # 存储ROC曲线数据
        if 'fpr' in test_metrics and 'tpr' in test_metrics:
            all_roc_data.append((test_metrics['fpr'], test_metrics['tpr']))
            all_auc_values.append(test_metrics['roc_auc'])

        # 收集实验数据
        experiment_data = {
            'experiment_id': exp_id + 1,
            'train_history': train_result,
            'test_metrics': test_metrics,
            'best_sensitivity': train_result['best_sensitivity'],
            'best_accuracy': train_result['best_acc']
        }

        all_experiment_results.append(experiment_data)
        all_train_histories.append(train_result)
        all_test_metrics.append(test_metrics)

        # 打印本次实验的结果
        print(f"\n实验 #{exp_id + 1} 结果:")
        print(f"- 最佳验证敏感度: {train_result['best_sensitivity']:.4f}")
        print(f"- 最佳验证准确率: {train_result['best_acc']:.4f}")
        print(f"- 测试集准确率: {test_metrics['accuracy']:.4f}")
        print(f"- 测试集敏感度: {test_metrics['sensitivity']:.4f}")
        print(f"- 测试集AUC: {test_metrics['roc_auc']:.4f}")

    # 4. 计算和绘制平均ROC曲线±标准差区间
    print(f"\n{'=' * 80}")
    print("绘制平均ROC曲线和标准差区间")
    print(f"{'=' * 80}")

    if all_roc_data and len(all_roc_data) == num_experiments:
        mean_auc, std_auc = plot_mean_roc_curve_with_std(
            all_roc_data, all_auc_values,
            model_name=f"{model_type.upper()} (三次实验)",
            save_path='./results'
        )
    else:
        print("警告: 无法计算平均ROC曲线，ROC数据不完整")

    # 5. 绘制训练曲线的平均值和标准差
    # print(f"\n{'=' * 80}")
    # print("绘制训练和验证曲线的平均值和标准差")
    # print(f"{'=' * 80}")

    # plot_mean_training_curves(all_train_histories, save_path='./results')

    # 6. 生成综合报告
    print(f"\n{'=' * 80}")
    print("三次独立实验综合报告")
    print(f"{'=' * 80}")

    # 计算各项指标的平均值和标准差
    metrics_to_analyze = ['accuracy', 'sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc']

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

    # 打印报告
    print(f"\n模型: {model_type.upper()}")
    print(f"实验次数: {num_experiments}")
    print(f"训练轮数: {num_epochs}")
    print(f"类别: {class_names}")

    print(f"\n测试集性能统计 (均值 ± 标准差):")
    for metric, stats in metrics_summary.items():
        metric_names = {
            'accuracy': '准确率',
            'sensitivity': '敏感度',
            'specificity': '特异度',
            'precision': '精确率',
            'f1_score': 'F1分数',
            'roc_auc': 'ROC AUC'
        }
        print(f"  {metric_names.get(metric, metric)}: "
              f"{stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(范围: {stats['min']:.4f} - {stats['max']:.4f})")

    # 7. 保存所有实验结果到CSV文件
    print(f"\n{'=' * 80}")
    print("保存所有实验结果")
    print(f"{'=' * 80}")

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        '实验编号': [f'实验{i + 1}' for i in range(num_experiments)],
        '最佳验证敏感度': [exp['best_sensitivity'] for exp in all_experiment_results],
        '最佳验证准确率': [exp['best_accuracy'] for exp in all_experiment_results],
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
        '最佳验证敏感度': f"{np.mean([exp['best_sensitivity'] for exp in all_experiment_results]):.4f} ± {np.std([exp['best_sensitivity'] for exp in all_experiment_results]):.4f}",
        '最佳验证准确率': f"{np.mean([exp['best_accuracy'] for exp in all_experiment_results]):.4f} ± {np.std([exp['best_accuracy'] for exp in all_experiment_results]):.4f}",
        '测试准确率': f"{metrics_summary['accuracy']['mean']:.4f} ± {metrics_summary['accuracy']['std']:.4f}",
        '测试敏感度': f"{metrics_summary['sensitivity']['mean']:.4f} ± {metrics_summary['sensitivity']['std']:.4f}",
        '测试特异度': f"{metrics_summary['specificity']['mean']:.4f} ± {metrics_summary['specificity']['std']:.4f}",
        '测试精确率': f"{metrics_summary['precision']['mean']:.4f} ± {metrics_summary['precision']['std']:.4f}",
        '测试F1分数': f"{metrics_summary['f1_score']['mean']:.4f} ± {metrics_summary['f1_score']['std']:.4f}",
        '测试ROC_AUC': f"{metrics_summary['roc_auc']['mean']:.4f} ± {metrics_summary['roc_auc']['std']:.4f}"
    }

    results_df = pd.concat([results_df, pd.DataFrame([stats_row])], ignore_index=True)

    # 保存到CSV
    results_csv_path = './results/three_experiments_summary.csv'
    results_df.to_csv(results_csv_path, index=False, encoding='utf-8-sig')
    print(f"实验结果已保存到: {results_csv_path}")

    # 8. 保存完整的实验检查点
    final_checkpoint = {
        'all_experiment_results': all_experiment_results,
        'all_test_metrics': all_test_metrics,
        'all_roc_data': all_roc_data,
        'all_auc_values': all_auc_values,
        'metrics_summary': metrics_summary,
        'model_type': model_type,
        'class_names': class_names,
        'num_experiments': num_experiments,
        'num_epochs': num_epochs
    }

    checkpoint_path = './results/final_experiment_checkpoint.pth'
    torch.save(final_checkpoint, checkpoint_path)
    print(f"完整实验检查点已保存到: {checkpoint_path}")

    print(f"\n{'=' * 80}")
    print("三次独立实验完成!")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    main()