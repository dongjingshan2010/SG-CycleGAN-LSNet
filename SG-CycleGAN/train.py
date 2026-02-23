#!/usr/bin/python3

import argparse
import itertools
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from PIL import Image
import yaml
from rectified_flow import RectifiedFlow
from unet_model import MiniUnet
from models_conv_dann import Generator, SharedGenerator
from models_conv_dann import Discriminator, Domain_Discriminator
from utils import ReplayBuffer, LambdaLR, Logger, weights_init_normal
from datasets import ImageDataset
import torch
from unet_model import MiniUnet
from rectified_flow import RectifiedFlow
import cv2
import os, math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=30, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=30, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=6, help='size of the batches')
parser.add_argument('--depth4vit', type=int, default=3, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='./datasets/Med_shallowdeep/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=5,
                    help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--mtwg_fea', type=int, default=32, help='size of the batches')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--cuda', default=True, action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--world_size', type=int, default=1, help='number of nodes for distributed training')
parser.add_argument('--rank', type=int, default=0, help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:12345', help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', help='distributed backend')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use')
opt = parser.parse_args()

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


def setup(rank, world_size):
    # Initialize distributed training
    dist.init_process_group(
        backend=opt.dist_backend,
        init_method=opt.dist_url,
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


class GradientReversalLayer(torch.autograd.Function):
    """改进的梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.save_for_backward(torch.tensor(alpha))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return grad_output.neg() * alpha, None  # 明确使用neg()方法


def main_worker(rank, world_size):
    # Initialize distributed training
    if world_size > 1:
        setup(rank, world_size)

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and opt.cuda else "cpu")
    torch.cuda.set_device(device)

    # Set global batch size
    batch_size = opt.batchSize // world_size

    # 设置梯度累积步数
    accumulation_steps = 4

    # Networks
    shared_generator = SharedGenerator(opt.input_nc, opt.output_nc, opt.mtwg_fea, opt.size, opt.depth4vit).to(device)

    # checkpoint_path = 'output/shared_generator.pth'
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    #
    # # 处理DDP包装的模型权重（如果checkpoint是DDP保存的）
    # state_dict = checkpoint.get('shared_generator', checkpoint)  # 兼容不同保存方式
    # if all(key.startswith('module.') for key in state_dict.keys()):
    #     # 移除DDP添加的'module.'前缀
    #     state_dict = {k[7:]: v for k, v in state_dict.items()}
    # # 加载权重到模型
    # shared_generator.load_state_dict(state_dict, strict=True)  # strict=True确保权重完全匹配
    # print(f"成功从 {checkpoint_path} 加载SharedGenerator参数")

    def get_generator():
        """根据是否使用DDP返回正确的生成器实例"""
        return shared_generator.module if world_size > 1 else shared_generator

    def netG_A2B(x):
        return get_generator().forward_a2b(x)

    def netG_B2A(x):
        return get_generator().forward_b2a(x)

    # 分别定义优化器
    optimizer_G = optim.Adam(shared_generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    netD_A = Discriminator(opt.input_nc).to(device)
    netD_B = Discriminator(opt.output_nc).to(device)
    domain_cls = Domain_Discriminator(opt.mtwg_fea).to(device)

    # Apply weight initialization
    shared_generator.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    domain_cls.apply(weights_init_normal)

    # Wrap models for distributed training
    if world_size > 1:
        shared_generator = DDP(shared_generator, device_ids=[rank])
        netD_A = DDP(netD_A, device_ids=[rank])
        netD_B = DDP(netD_B, device_ids=[rank])
        domain_cls = DDP(domain_cls, device_ids=[rank])

    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_domain = nn.CrossEntropyLoss()

    # Optimizers & LR schedulers
    optimizer_D_A = optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_domain = optim.Adam(domain_cls.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G,
                                                 lr_lambda=LambdaLR(opt.epoch, 0, opt.decay_epoch).step)
    lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                   lr_lambda=LambdaLR(opt.epoch, 0, opt.decay_epoch).step)
    lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                   lr_lambda=LambdaLR(opt.epoch, 0, opt.decay_epoch).step)
    lr_scheduler_domain = optim.lr_scheduler.LambdaLR(optimizer_domain,
                                                      lr_lambda=LambdaLR(opt.epoch, 0, opt.decay_epoch).step)

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    input_A = Tensor(batch_size, opt.input_nc, opt.size, opt.size)
    input_B = Tensor(batch_size, opt.output_nc, opt.size, opt.size)
    target_real = Variable(Tensor(batch_size).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(batch_size).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # Dataset loader
    transforms_ = [
        transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(opt.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    dataset = ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True)

    # Distributed sampler if using multiple GPUs
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=opt.n_cpu,
        drop_last=True,
        sampler=sampler
    )

    # Train CycleGAN
    for epoch in range(opt.epoch, opt.epoch):
        print('Epoch {}/{}'.format(epoch, opt.epoch))
        if world_size > 1:
            sampler.set_epoch(epoch)

        for i, batch in enumerate(dataloader):
            # Set model input
            real_A = Variable(input_A.copy_(batch['A'])).to(device)
            real_B = Variable(input_B.copy_(batch['B'])).to(device)

            ###### Generators A2B and B2A ######
            # 只在累积步骤的开始时清零梯度
            if i % accumulation_steps == 0:
                optimizer_G.zero_grad()

            # Identity loss
            same_B, modefreeB = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            same_A, modefreeA = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # Domain adaptation loss with gradient reversal
            features = torch.cat([modefreeA, modefreeB], dim=0)
            label_A = torch.zeros(batch_size, dtype=torch.long, device=device)
            label_B = torch.ones(batch_size, dtype=torch.long, device=device)
            labels = torch.cat([label_A, label_B], dim=0)

            permutation = torch.randperm(features.size(0), device=device)
            features_shuffled = features[permutation]
            labels_shuffled = labels[permutation]
            # 梯度反转：让生成器学习生成域不变的特征
            p = epoch / opt.epoch
            lamba = 2 / (1 + math.exp(-5 * p)) - 1
            features_grl = GradientReversalLayer.apply(features_shuffled, lamba)
            pred_shuffled = domain_cls(features_grl)
            loss_domain = criterion_domain(pred_shuffled, labels_shuffled) * 5

            # GAN loss
            fake_B, shared_real_A = netG_A2B(real_A)

            # Get discriminator instance (handles DDP)
            netD_B_instance = netD_B.module if world_size > 1 else netD_B
            pred_fake = netD_B_instance(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A, shared_real_B = netG_B2A(real_B)
            netD_A_instance = netD_A.module if world_size > 1 else netD_A
            pred_fake = netD_A_instance(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A, shared_fake_B = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B, shared_fake_A = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Content consistency loss
            loss_consistency_A = criterion_identity(shared_real_A, shared_fake_B) * 5.0
            loss_consistency_B = criterion_identity(shared_real_B, shared_fake_A) * 5.0

            # Total loss
            loss_G = (loss_identity_A + loss_identity_B + loss_GAN_A2B +
                      loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB +
                      loss_consistency_A + loss_consistency_B + loss_domain)

            # 对损失进行归一化，因为我们要累积梯度
            loss_G = loss_G / accumulation_steps
            loss_G.backward()

            # 只在累积步骤结束时更新参数
            if (i + 1) % accumulation_steps == 0:
                optimizer_G.step()

            ###### Discriminator A ######
            # 只在累积步骤的开始时清零梯度
            if i % accumulation_steps == 0:
                optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A_instance(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A_instance(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            # 对损失进行归一化
            loss_D_A = loss_D_A / accumulation_steps
            loss_D_A.backward()

            # 只在累积步骤结束时更新参数
            if (i + 1) % accumulation_steps == 0:
                optimizer_D_A.step()

            ###### Discriminator B ######
            # 只在累积步骤的开始时清零梯度
            if i % accumulation_steps == 0:
                optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B_instance(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B_instance(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            # 对损失进行归一化
            loss_D_B = loss_D_B / accumulation_steps
            loss_D_B.backward()

            # 只在累积步骤结束时更新参数
            if (i + 1) % accumulation_steps == 0:
                optimizer_D_B.step()

            ###### Domain Classifier ######
            # 只在累积步骤的开始时清零梯度
            if i % accumulation_steps == 0:
                optimizer_domain.zero_grad()

            # 正常训练域分类器（不反转梯度）
            pred_normal = domain_cls(features_shuffled.detach())
            loss_domain_normal = criterion_domain(pred_normal, labels_shuffled)
            # 对损失进行归一化
            loss_domain_normal = loss_domain_normal / accumulation_steps
            loss_domain_normal.backward()

            # 只在累积步骤结束时更新参数
            if (i + 1) % accumulation_steps == 0:
                optimizer_domain.step()

            if rank == 0 and i % 200 == 0:
                domain_acc = (pred_shuffled.argmax(dim=1) == labels_shuffled).float().mean()
                print(f'[Epoch {epoch}] [batch {i}] '
                      f'Domain Loss: {loss_domain.item() * accumulation_steps:.4f}, Domain Acc: {domain_acc.item():.4f}',
                      f'Loss_total: {loss_G.item() * accumulation_steps:.4f}')  #

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        lr_scheduler_domain.step()

        # Save models checkpoints (only on master process)
        if rank == 0:
            # Handle DDP model saving
            shared_generator_model = shared_generator.module if world_size > 1 else shared_generator
            netD_A_model = netD_A.module if world_size > 1 else netD_A
            netD_B_model = netD_B.module if world_size > 1 else netD_B

            shared_generator_shared_layers_model = shared_generator.module.shared_layers if world_size > 1 else shared_generator.shared_layers

            torch.save(shared_generator_model.state_dict(), 'output/shared_generator.pth')
            torch.save(netD_A_model.state_dict(), 'output/netD_A.pth')
            torch.save(netD_B_model.state_dict(), 'output/netD_B.pth')

            torch.save(shared_generator_shared_layers_model.state_dict(), 'output/shared_generator_sharedlayers.pth')

    # ============== 释放CycleGAN相关的GPU资源 ==============
    # 1. 删除不再使用的判别器、优化器、调度器等（这些在Rectified Flow阶段不需要）
    del netD_A, netD_B, optimizer_D_A, optimizer_D_B
    del lr_scheduler_D_A, lr_scheduler_D_B, criterion_GAN, criterion_cycle, criterion_identity
    # 2. 删除输入缓存和缓冲区（CycleGAN特有的中间变量）
    del target_real, target_fake, fake_A_buffer, fake_B_buffer
    # 3. 强制Python垃圾回收（删除引用计数为0的对象）
    import gc
    gc.collect()
    # 4. 清空CUDA未使用的缓存（释放GPU显存）
    torch.cuda.empty_cache()

    # Train Rectified Flow

    # checkpoint_path = 'output/shared_generator_deep.pth'
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    # # 提取state_dict（兼容不同保存方式）
    # state_dict = checkpoint.get('shared_generator', checkpoint)
    #
    # # 判断当前模型是否被DDP包装
    # is_ddp_model = isinstance(shared_generator, torch.nn.parallel.DistributedDataParallel)
    #
    # # 处理前缀不匹配问题
    # if is_ddp_model:
    #     # 模型是DDP的（需要带module.前缀），如果state_dict没有前缀，则添加
    #     if not all(key.startswith('module.') for key in state_dict.keys()):
    #         state_dict = {f"module.{k}": v for k, v in state_dict.items()}
    # else:
    #     # 模型不是DDP的（不需要module.前缀），如果state_dict有前缀，则移除
    #     if all(key.startswith('module.') for key in state_dict.keys()):
    #         state_dict = {k[7:]: v for k, v in state_dict.items()}
    #
    # # 加载权重
    # shared_generator.load_state_dict(state_dict, strict=True)
    # print(f"成功从 {checkpoint_path} 加载SharedGenerator参数")
    #
    # print(f"Successfully load SharedGenerator from {checkpoint_path} for Rectified Flow training")
    # # Freeze SharedGenerator
    # for param in shared_generator.parameters():
    #     param.requires_grad = False

    # Load Rectified Flow config
    config = './config/train_config.yaml'
    config = yaml.load(open(config, 'rb'), Loader=yaml.FullLoader)
    base_channels = config.get('base_channels', 256)
    # epochs = config.get('epochs', 10)
    # batch_size = config.get('batch_size', 64)
    lr_adjust_epoch = config.get('lr_adjust_epoch', 50)
    batch_print_interval = config.get('batch_print_interval', 100)
    # checkpoint_save_interval = config.get('checkpoint_save_interval', 1)
    save_path = config.get('save_path', './checkpoints')
    use_cfg = config.get('use_cfg', False)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and opt.cuda else "cpu")

    # Create and wrap Rectified Flow model
    Unetmodel = MiniUnet(base_channels).to(device)

    # ============== 新增：计算并输出MiniUnet参数量 ==============
    def count_parameters(model):
        """计算模型总参数量和可训练参数量"""
        total_params = sum(p.numel() for p in model.parameters())  # 总参数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量
        return total_params, trainable_params

    # 计算参数量
    total_params, trainable_params = count_parameters(Unetmodel)
    # 格式化输出（转换为百万(M)单位，保留2位小数）
    print(f"MiniUnet 模型参数量统计:")
    print(f"  总参数量: {total_params:,} 个 ({total_params / 1e6:.2f}M)")
    print(f"  可训练参数量: {trainable_params:,} 个 ({trainable_params / 1e6:.2f}M)")
    # ==========================================================

    if world_size > 1:
        Unetmodel = DDP(Unetmodel, device_ids=[rank])

    # Optimizer and scheduler
    optimizer = optim.AdamW(Unetmodel.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    rf = RectifiedFlow()
    loss_list = []

    # Train Rectified Flow (2)
    # Get generator instance (handles DDP)
    shared_generator_instance = shared_generator.module if world_size > 1 else shared_generator
    for epoch in range(opt.epoch, opt.n_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        if rank == 0:
            print(f'Rectified Flow Epoch {epoch}/{opt.n_epochs}')
            # 初始化每个epoch的loss记录
            epoch_loss = 0.0

        for i, batch in enumerate(dataloader):
            real_A = Variable(input_A.copy_(batch['A'])).to(device)
            real_B = Variable(input_B.copy_(batch['B'])).to(device)

            fake_B, shared_real_B = shared_generator_instance.forward_a2b(real_B)

            # Prepare data for Rectified Flow
            false_batch = fake_B
            true_batch = real_B
            x_1 = true_batch
            t = torch.rand(x_1.size(0)).to(device)

            # Generate flow
            x_t, _ = rf.create_flow(x_1, t)
            x_0 = false_batch

            # Move data to device
            x_t = x_t.to(device)
            x_0 = x_0.to(device)
            x_1 = x_1.to(device)

            optimizer.zero_grad()

            if use_cfg:
                # CFG handling
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                t = torch.cat([t, t.clone()], dim=0)
                y = torch.cat([torch.ones(x_1.size(0)), -torch.ones(x_1.size(0))], dim=0)
                x_1 = torch.cat([x_1, x_1.clone()], dim=0)
                x_0 = torch.cat([x_0, x_0.clone()], dim=0)
                y = y.to(device)
            else:
                y = None

            # Forward pass
            v_pred = Unetmodel(x=x_t, t=t, y=y)

            # Calculate loss
            loss = rf.mse_loss(v_pred, x_1, x_0)
            # 累加每个batch的loss
            if rank == 0:
                epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print training info (only on master process)
            if rank == 0 and i % batch_print_interval == 0:
                print(f'[Epoch {epoch}] [batch {i}] loss: {loss.item()}')

            # Record loss (only on master process)
            if rank == 0:
                loss_list.append(loss.item())

        # 打印每个epoch的平均loss
        if rank == 0:
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f'[Epoch {epoch} finished] average_loss: {avg_epoch_loss:.6f}')
        # Update learning rate
        scheduler.step()

        # 保存检查点（仅在主GPU上）
        # Save checkpoint (only on master process)
        if epoch == opt.n_epochs - 1:
            print(f'Saving model {epoch} to {save_path}...')
            save_dict = dict(
                # 使用当前模型状态（已处理DDP包装）
                Unetmodel=Unetmodel.module.state_dict() if hasattr(Unetmodel, 'module') else Unetmodel.state_dict(),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                epoch=epoch,
                loss_list=loss_list
            )
            torch.save(save_dict, os.path.join(save_path, f'miniunet_finalepoch.pth'))

    del shared_generator
    # Cleanup distributed training
    if world_size > 1:
        cleanup()

def infer(
        checkpoint_path,
        base_channels=16,
        step=50,
        num_imgs=1,
        org_img=None,
        filenames=None,
        y=None,
        cfg_scale=7.0,
        save_path='./results',
        save_path_motaiwuguan='./results',
        save_noise_path=None,
        device='cuda',
        shared_generator=None,
        img=None,
        world_size=1):
    os.makedirs(save_path, exist_ok=True)
    if save_noise_path is not None:
        os.makedirs(save_noise_path, exist_ok=True)

    if y is not None:
        assert len(y.shape) == 1 or len(y.shape) == 2, 'y must be 1D or 2D tensor'
        assert y.shape[0] == num_imgs or y.shape[0] == 1, 'y.shape[0] must match num_imgs or be 1'
        if y.shape[0] == 1:
            y = y.repeat(num_imgs, 1).reshape(num_imgs)
        y = y.to(device)

    model = MiniUnet(base_channels=base_channels)
    model.to(device)
    model.eval()

    rf = RectifiedFlow()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['Unetmodel']
    if all(key.startswith('module.') for key in state_dict.keys()):
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    shared_generator_instance = shared_generator.module if world_size > 1 else shared_generator
    shared_generator_instance.eval()  # 确保生成器处于评估模式

    # with torch.no_grad():
    #     for i in range(num_imgs):
    #         print(f'Generating {i}th image...')
    #         dt = 1.0 / step
    #
    #         if img=='uto':
    #             # 生成 fake_B 并保存
    #             fake_org, shared_real_org = shared_generator_instance.forward_a2b(org_img.to(device))  # 原生成fake_B的代码
    #         elif img=='mri':
    #             fake_org, shared_real_org = shared_generator_instance.forward_b2a(org_img.to(device))  # 原生成fake_B的代码
    #         # ====================== 新增：保存 fake_B 为图片 ======================
    #         # 1. 取第一个样本（移除batch维度）
    #         fake_org_single = fake_org[0]  # 形状: [C, H, W]
    #
    #         # 2. 反归一化（和x_t处理一致）
    #         mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    #         std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    #         fake_org_denorm = fake_org_single * std + mean  # 反归一化
    #
    #         # 3. 线性缩放到[0, 1]（保留纹理，不截断）
    #         # 展平计算min/max（为单张图片添加batch维度便于复用代码）
    #         fake_org_flat = fake_org_denorm.unsqueeze(0).view(1, 3, -1)  # 形状: [1, 3, H*W]
    #         min_val = fake_org_flat.min(dim=2, keepdim=True)[0].view(1, 3, 1, 1)
    #         max_val = fake_org_flat.max(dim=2, keepdim=True)[0].view(1, 3, 1, 1)
    #         range_val = max_val - min_val
    #         range_val[range_val == 0] = 1.0  # 避免除零
    #         fake_org_01 = (fake_org_denorm - min_val[0]) / range_val[0]  # 缩放到[0,1]
    #
    #         # 4. 转换为OpenCV格式并保存
    #         fake_org_np = fake_org_01.detach().cpu().numpy()  # [C, H, W]
    #         fake_org_np = fake_org_np.transpose(1, 2, 0)  # 转换为[H, W, C]
    #         fake_org_np = cv2.cvtColor(fake_org_np, cv2.COLOR_RGB2BGR)  # 转换为BGR（OpenCV默认格式）
    #         fake_org_np = (fake_org_np * 255).astype('uint8')  # 转换为0-255的uint8
    #
    #         # 5. 保存（文件名添加"_fake_B"后缀区分）
    #         if img == 'uto':
    #             fake_org_filename = f'{os.path.splitext(filenames[0])[0]}.png'  #2Mri
    #             print(f"save transfered images by rectified flow further, not here")
    #         elif img == 'mri':
    #             fake_org_filename = f'{os.path.splitext(filenames[0])[0]}.png'  #2Uto
    #             cv2.imwrite(os.path.join(save_path, fake_org_filename), fake_org_np)
    #             print(f"已保存 fake_B 至: {os.path.join(save_path, fake_org_filename)}")
    #         # ==================================================================
    #
    #         ## +++++++++++++++++保存模态无关特征图++++++++++++++++++++++++++++++++++++++++++++++++++++
    #         # 1. 取第一个样本（移除batch维度），形状为 [32, 256, 256]
    #         fake_org_single = (shared_real_org[0] + 1) / 2  # 形状: [32, 256, 256]
    #
    #         # 2. 创建可视化方案：将32个通道排列成网格
    #         def visualize_32ch_as_grid(feature_maps, grid_cols=8):
    #             """
    #             将32通道的特征图排列成网格进行可视化
    #             feature_maps: [32, H, W] 张量
    #             grid_cols: 网格列数
    #             """
    #             channels, H, W = feature_maps.shape
    #             grid_rows = (channels + grid_cols - 1) // grid_cols  # 计算需要的行数
    #
    #             # 创建空白画布
    #             grid_h = grid_rows * H
    #             grid_w = grid_cols * W
    #             grid_image = torch.zeros((3, grid_h, grid_w), device=feature_maps.device)
    #
    #             # 将每个通道的特征图放置到网格中
    #             for ch_idx in range(channels):
    #                 row = ch_idx // grid_cols
    #                 col = ch_idx % grid_cols
    #
    #                 # 获取当前通道的特征图
    #                 single_channel = feature_maps[ch_idx]  # [H, W]
    #
    #                 # 对单个通道进行归一化到[0,1]
    #                 ch_min = single_channel.min()
    #                 ch_max = single_channel.max()
    #                 if ch_max - ch_min > 1e-6:
    #                     normalized_ch = (single_channel - ch_min) / (ch_max - ch_min)
    #                 else:
    #                     normalized_ch = torch.zeros_like(single_channel)
    #
    #                 # 将单通道复制到RGB三个通道（创建灰度图）
    #                 rgb_channel = normalized_ch.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    #
    #                 # 放置到网格对应位置
    #                 start_h = row * H
    #                 start_w = col * W
    #                 grid_image[:, start_h:start_h + H, start_w:start_w + W] = rgb_channel
    #
    #             return grid_image
    #
    #         # 3. 生成网格可视化
    #         grid_visualization = visualize_32ch_as_grid(fake_org_single, grid_cols=8)
    #
    #         # 4. 转换为OpenCV格式并保存
    #         def save_grid_visualization(grid_tensor, save_path, filename):
    #             """保存网格可视化结果"""
    #             # 转换为numpy
    #             grid_np = grid_tensor.detach().cpu().numpy()  # [3, grid_h, grid_w]
    #             grid_np = grid_np.transpose(1, 2, 0)  # 转换为 [grid_h, grid_w, 3]
    #
    #             # 转换为BGR并缩放到0-255
    #             grid_np = cv2.cvtColor(grid_np, cv2.COLOR_RGB2BGR)
    #             grid_np = (grid_np * 255).astype(np.uint8)
    #
    #             # 保存
    #             os.makedirs(save_path, exist_ok=True)
    #             save_full_path = os.path.join(save_path, filename)
    #             cv2.imwrite(save_full_path, grid_np)
    #
    #             return save_full_path, grid_np.shape
    #
    #         # 5. 保存网格图
    #         filename = f'{os.path.splitext(filenames[0])[0]}_32ch_grid.png'
    #         save_path, final_shape = save_grid_visualization(
    #             grid_visualization,
    #             save_path_motaiwuguan,
    #             filename
    #         )
    #         print(f"✅ 已保存32通道网格可视化图至：{save_path}")
    #         print(f"📐 最终图像尺寸：{final_shape[1]}×{final_shape[0]} (宽×高)")
    #         print(f"🔢 通道排列：4行×8列 (共32个通道)")
    #
    #         def create_pca_summary(feature_maps, save_path, filename):
    #             """使用PCA将32通道降维为3通道彩色图"""
    #             # 重塑数据: [32, H*W] -> [H*W, 32]
    #             features_flat = feature_maps.view(opt.mtwg_fea, -1).T.detach().cpu().numpy()  # [H*W, 32]
    #
    #             # 应用PCA
    #             from sklearn.decomposition import PCA
    #             pca = PCA(n_components=3)
    #             pca_result = pca.fit_transform(features_flat)  # [H*W, 3]
    #
    #             # 归一化到[0,1]
    #             pca_min = pca_result.min(axis=0)
    #             pca_max = pca_result.max(axis=0)
    #             pca_normalized = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)
    #
    #             # 重塑回图像尺寸
    #             H, W = feature_maps.shape[1], feature_maps.shape[2]
    #             pca_image = pca_normalized.reshape(H, W, 3)
    #
    #             # 转换为BGR并保存
    #             pca_image_bgr = cv2.cvtColor((pca_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    #             pca_filename = f'{filename}_pca_summary.png'
    #             cv2.imwrite(os.path.join(save_path, pca_filename), pca_image_bgr)
    #
    #             print(f"✅ 已保存PCA摘要图至：{os.path.join(save_path, pca_filename)}")
    #             print(f"📊 PCA解释方差比：{pca.explained_variance_ratio_}")
    #
    #         # 可选：取消注释以生成PCA摘要图
    #         create_pca_summary(fake_org_single, save_path_motaiwuguan,
    #                           os.path.splitext(filenames[0])[0])
    #
    #         if opt.epoch != opt.n_epochs and img == 'uto':
    #         # not training rectified flow, and do not generate images by it.
    #             # 继续处理后续生成步骤（原代码逻辑）
    #             x_t = fake_org  # 用fake_B作为初始x_t
    #             noise = x_t.detach().cpu().numpy()
    #
    #             if y is not None:
    #                 y_i = y[i].unsqueeze(0)
    #
    #             for j in range(step):
    #                 if j % 10 == 0:
    #                     print(f'Generating {i}th image, step {j}...')
    #                 t = j * dt
    #                 t = torch.tensor([t]).to(device)
    #
    #                 if y is not None:
    #                     v_pred_uncond = model(x=x_t, t=t)
    #                     v_pred_cond = model(x=x_t, t=t, y=y_i)
    #                     v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
    #                 else:
    #                     v_pred = model(x=x_t, t=t)
    #
    #                 x_t = rf.euler(x_t, v_pred, dt)
    #
    #             # 处理并保存最终生成的x_t（原代码逻辑，保持不变）
    #             x_t = x_t[0]
    #             filename = filenames[0]
    #
    #             mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    #             std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
    #             x_t = x_t * std + mean
    #             batch_size, channels, h, w = x_t.unsqueeze(0).shape
    #             x_flat = x_t.view(batch_size, channels, -1)
    #             min_val = x_flat.min(dim=2, keepdim=True)[0].view(batch_size, channels, 1, 1)
    #             max_val = x_flat.max(dim=2, keepdim=True)[0].view(batch_size, channels, 1, 1)
    #             range_val = max_val - min_val
    #             range_val[range_val == 0] = 1.0
    #             x_t_01 = (x_t - min_val) / range_val
    #             img = x_t_01.detach().cpu().numpy()
    #
    #             img = img.squeeze().transpose(1, 2, 0)  # 转换为[H, W, C]
    #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # 转换为BGR
    #             img = (img * 255).astype('uint8')
    #             cv2.imwrite(os.path.join(save_path, f'{filename}.png'), img)
    #             if save_noise_path is not None:
    #                 np.save(os.path.join(save_noise_path, f'{i}.npy'), noise)
    with torch.no_grad():
        for i in range(num_imgs):
            print(f'Generating {i}th image...')
            dt = 1.0 / step

            if img == 'uto':
                # 生成 fake_B 并保存
                fake_org, shared_real_org = shared_generator_instance.forward_a2b(org_img.to(device))
            elif img == 'mri':
                fake_org, shared_real_org = shared_generator_instance.forward_b2a(org_img.to(device))

            # ====================== 增强对比度优化版：保存 fake_B 为图片 ======================
            # 1. 取第一个样本（移除batch维度）
            fake_org_single = fake_org[0]  # 形状: [C, H, W]

            # 2. 反归一化（和x_t处理一致）
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
            fake_org_denorm = fake_org_single * std + mean  # 反归一化

            # 打印原始数据统计
            print(f"📊 原始数据统计:")
            print(f"  最小值: {fake_org_denorm.min().item():.6f}")
            print(f"  最大值: {fake_org_denorm.max().item():.6f}")
            print(f"  均值: {fake_org_denorm.mean().item():.6f}")

            # 3. 智能对比度增强的线性缩放
            def smart_contrast_stretch(image_tensor, percentile_low=2, percentile_high=98):
                """智能对比度拉伸，使用百分位数避免极端值影响"""
                # 转换为numpy处理
                img_np = image_tensor.detach().cpu().numpy()

                # 计算每个通道的百分位数
                channels = img_np.shape[0]
                enhanced_channels = []

                for c in range(channels):
                    channel_data = img_np[c].flatten()

                    # 计算百分位数
                    low_val = np.percentile(channel_data, percentile_low)
                    high_val = np.percentile(channel_data, percentile_high)

                    # 如果百分位数范围太小，使用最小最大值
                    if high_val - low_val < 0.01:
                        low_val = channel_data.min()
                        high_val = channel_data.max()

                    # 线性拉伸到[0, 1]
                    if high_val - low_val > 0:
                        channel_stretched = np.clip((img_np[c] - low_val) / (high_val - low_val), 0, 1)
                    else:
                        channel_stretched = np.zeros_like(img_np[c])

                    enhanced_channels.append(channel_stretched)

                # 合并通道
                enhanced_np = np.stack(enhanced_channels, axis=0)
                return torch.from_numpy(enhanced_np).float().to(image_tensor.device)

            # 应用智能对比度拉伸
            fake_org_stretched = smart_contrast_stretch(fake_org_denorm, percentile_low=1, percentile_high=99)

            # 4. 应用Gamma校正进一步增强对比度
            def apply_gamma_correction(image_tensor, gamma=1.5):
                """应用Gamma校正增强对比度"""
                # Gamma校正公式: output = input^(1/gamma)
                # 对于gamma>1，增强暗部细节；对于gamma<1，增强亮部细节
                gamma_corrected = torch.pow(image_tensor, 1.0 / gamma)
                return gamma_corrected

            fake_org_gamma = apply_gamma_correction(fake_org_stretched, gamma=1.2)

            # 5. 转换为OpenCV格式并应用进一步增强
            fake_org_np = fake_org_gamma.detach().cpu().numpy()  # [C, H, W]
            fake_org_np = fake_org_np.transpose(1, 2, 0)  # 转换为[H, W, C]

            # 确保值在[0, 1]范围内
            fake_org_np = np.clip(fake_org_np, 0, 1)

            # 转换为BGR和0-255
            fake_org_np = cv2.cvtColor(fake_org_np, cv2.COLOR_RGB2BGR)
            fake_org_np_8bit = (fake_org_np * 255).astype('uint8')

            # 6. 应用OpenCV对比度增强技术
            def enhance_contrast_opencv(image):
                """使用OpenCV技术增强图像对比度"""
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # 方法1: CLAHE（对比度受限的自适应直方图均衡化）
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)

                    # 应用CLAHE到L通道
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l_enhanced = clahe.apply(l)

                    # 合并通道并转换回BGR
                    lab_enhanced = cv2.merge([l_enhanced, a, b])
                    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

                    # 方法2: 直方图均衡化（YCrCb空间）
                    ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YCrCb)
                    y, cr, cb = cv2.split(ycrcb)
                    y_eq = cv2.equalizeHist(y)
                    ycrcb_eq = cv2.merge([y_eq, cr, cb])
                    enhanced = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

                    # 方法3: 锐化滤波器增强纹理
                    kernel = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])
                    enhanced = cv2.filter2D(enhanced, -1, kernel)

                    return enhanced
                else:
                    # 灰度图像处理
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(image)
                    enhanced = cv2.equalizeHist(enhanced)
                    return enhanced

            # 应用OpenCV对比度增强
            fake_org_enhanced = enhance_contrast_opencv(fake_org_np_8bit)

            # 7. 保存图像
            if img == 'uto':
                fake_org_filename = f'{os.path.splitext(filenames[0])[0]}_enhanced.png'
                print(f"保存转换后的图像（使用校正流进一步处理）")
            elif img == 'mri':
                fake_org_filename = f'{os.path.splitext(filenames[0])[0]}_enhanced.png'

                # 保存增强版本
                cv2.imwrite(os.path.join(save_path, fake_org_filename), fake_org_enhanced)

                # 同时保存原始增强版本用于对比
                cv2.imwrite(os.path.join(save_path, f'{os.path.splitext(filenames[0])[0]}_basic.png'), fake_org_np_8bit)

                # 打印增强效果统计
                print(f"✅ 已保存增强版 fake_B 至: {os.path.join(save_path, fake_org_filename)}")
                print(f"🎨 增强效果:")
                print(f"  原始范围: [{fake_org_np_8bit.min()}, {fake_org_np_8bit.max()}]")
                print(f"  增强后范围: [{fake_org_enhanced.min()}, {fake_org_enhanced.max()}]")
                print(f"  对比度提升: {(fake_org_enhanced.std() / max(fake_org_np_8bit.std(), 1)):.2f}倍")

            # ==================================================================

            ## +++++++++++++++++保存模态无关特征图（32张独立图片）++++++++++++++++++++++++++++++++++++++++++++++++++++
            # 1. 取第一个样本（移除batch维度），形状为 [32, 256, 256]
            feature_map_original = shared_real_org[0]  # 形状: [32, 256, 256]

            # 2. 增强单个通道对比度的函数
            def enhance_single_channel(channel):
                """增强单个通道的对比度，使纹理更清晰"""
                # 转换为numpy
                ch_np = channel.detach().cpu().numpy()

                # 计算百分位数
                p1 = np.percentile(ch_np, 1)
                p99 = np.percentile(ch_np, 99)

                # 如果百分位数范围太小，使用最小最大值
                if p99 - p1 < 0.001:
                    p1 = ch_np.min()
                    p99 = ch_np.max()

                # 线性拉伸
                if p99 - p1 > 0:
                    channel_stretched = np.clip((ch_np - p1) / (p99 - p1), 0, 1)
                else:
                    channel_stretched = np.zeros_like(ch_np)

                # Gamma校正增强纹理
                gamma = 0.5  # 强伽马值增强暗部细节
                channel_gamma = np.power(channel_stretched, gamma)

                # 转换为8位图像
                channel_8bit = (channel_gamma * 255).astype(np.uint8)

                # 直方图均衡化
                channel_eq = cv2.equalizeHist(channel_8bit)

                # 局部对比度增强（CLAHE）
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                channel_clahe = clahe.apply(channel_eq)

                return channel_clahe

            # 3. 保存32张独立通道图片
            def save_individual_channels(feature_maps, save_path, filename_prefix):
                """保存所有通道为独立的图片"""
                os.makedirs(save_path, exist_ok=True)

                channels = feature_maps.shape[0]
                saved_files = []

                for ch_idx in range(channels):
                    # 获取当前通道
                    channel = feature_maps[ch_idx]  # [H, W]

                    # 增强通道对比度
                    channel_enhanced = enhance_single_channel(channel)

                    # 应用伪彩色映射增强可视性（可选）
                    if False:
                        # 使用JET伪彩色映射
                        colored_ch = cv2.applyColorMap(channel_enhanced, cv2.COLORMAP_JET)
                    else:
                        # 灰度图像
                        colored_ch = cv2.cvtColor(channel_enhanced, cv2.COLOR_GRAY2BGR)

                    # 保存图片
                    save_filename = f'{filename_prefix}_ch{ch_idx:02d}.png'
                    save_path_full = os.path.join(save_path, save_filename)
                    cv2.imwrite(save_path_full, colored_ch)

                    saved_files.append(save_path_full)

                return saved_files

            # 4. 保存所有32个通道
            print(f"💾 保存32个独立通道图片...")
            saved_files = save_individual_channels(feature_map_original, save_path_motaiwuguan,
                                                   os.path.splitext(filenames[0])[0])

            print(f"✅ 已保存32个独立通道图片至: {save_path_motaiwuguan}")
            print(f"📊 通道统计:")
            print(f"  总通道数: {feature_map_original.shape[0]}")
            print(f"  图像尺寸: {feature_map_original.shape[1]}×{feature_map_original.shape[2]}")

            # 5. 可选：生成通道对比度排名
            def calculate_channel_contrast(feature_maps):
                """计算每个通道的对比度（标准差）"""
                contrast_scores = []
                for ch_idx in range(feature_maps.shape[0]):
                    channel = feature_maps[ch_idx]
                    # 计算标准差作为对比度指标
                    std_dev = channel.std().item()
                    contrast_scores.append((ch_idx, std_dev))
                return contrast_scores

            if False:
                contrast_scores = calculate_channel_contrast(feature_map_original)
                contrast_scores.sort(key=lambda x: x[1], reverse=True)

                print(f"🏆 通道对比度排名 (前10):")
                for rank, (ch_idx, contrast) in enumerate(contrast_scores[:10]):
                    print(f"  排名{rank + 1}: 通道{ch_idx:02d}, 对比度={contrast:.6f}")

            # 6. 可选：生成特征摘要图（使用PCA）
            def create_pca_summary_enhanced(feature_maps, save_path, filename_prefix):
                """使用PCA将32通道降维为3通道彩色图（增强版）"""
                try:
                    from sklearn.decomposition import PCA

                    # 重塑数据: [32, H*W] -> [H*W, 32]
                    features_flat = feature_maps.view(feature_maps.shape[0], -1).T.detach().cpu().numpy()

                    # 应用PCA
                    pca = PCA(n_components=3)
                    pca_result = pca.fit_transform(features_flat)

                    # 归一化到[0,1]
                    pca_min = pca_result.min(axis=0)
                    pca_max = pca_result.max(axis=0)
                    pca_normalized = (pca_result - pca_min) / (pca_max - pca_min + 1e-8)

                    # 重塑回图像尺寸
                    H, W = feature_maps.shape[1], feature_maps.shape[2]
                    pca_image = pca_normalized.reshape(H, W, 3)

                    # 转换为BGR
                    pca_image_bgr = cv2.cvtColor((pca_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

                    # 应用对比度增强
                    lab = cv2.cvtColor(pca_image_bgr, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l_enhanced = clahe.apply(l)
                    lab_enhanced = cv2.merge([l_enhanced, a, b])
                    pca_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

                    # 保存
                    pca_filename = f'{filename_prefix}_pca_summary_enhanced.png'
                    cv2.imwrite(os.path.join(save_path, pca_filename), pca_enhanced)

                    print(f"✅ 已保存PCA摘要图至：{os.path.join(save_path, pca_filename)}")
                    print(f"📊 PCA解释方差比：{pca.explained_variance_ratio_}")

                except Exception as e:
                    print(f"⚠️ PCA摘要图生成失败: {e}")

            # 生成PCA摘要图（可选）
            if False:
                create_pca_summary_enhanced(feature_map_original, save_path_motaiwuguan,
                                            os.path.splitext(filenames[0])[0])

            # ==================================================================

            if opt.epoch != opt.n_epochs and img == 'uto':
                # not training rectified flow, and do not generate images by it.
                # 继续处理后续生成步骤（原代码逻辑）
                x_t = fake_org  # 用fake_B作为初始x_t
                noise = x_t.detach().cpu().numpy()

                if y is not None:
                    y_i = y[i].unsqueeze(0)

                for j in range(step):
                    if j % 10 == 0:
                        print(f'Generating {i}th image, step {j}...')
                    t = j * dt
                    t = torch.tensor([t]).to(device)

                    if y is not None:
                        v_pred_uncond = model(x=x_t, t=t)
                        v_pred_cond = model(x=x_t, t=t, y=y_i)
                        v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
                    else:
                        v_pred = model(x=x_t, t=t)

                    x_t = rf.euler(x_t, v_pred, dt)

                # 处理并保存最终生成的x_t（使用增强方法）
                x_t_single = x_t[0]
                filename = filenames[0]

                # 应用与前面相同的增强方法
                mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
                std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(device)
                x_t_denorm = x_t_single * std + mean

                # 智能对比度拉伸
                x_t_stretched = smart_contrast_stretch(x_t_denorm, percentile_low=1, percentile_high=99)

                # Gamma校正
                x_t_gamma = apply_gamma_correction(x_t_stretched, gamma=1.2)

                # 转换为OpenCV格式
                x_t_np = x_t_gamma.detach().cpu().numpy()
                x_t_np = x_t_np.transpose(1, 2, 0)
                x_t_np = np.clip(x_t_np, 0, 1)
                x_t_np = cv2.cvtColor(x_t_np, cv2.COLOR_RGB2BGR)
                x_t_np_8bit = (x_t_np * 255).astype('uint8')

                # 应用OpenCV增强
                x_t_enhanced = enhance_contrast_opencv(x_t_np_8bit)

                # 保存图像
                cv2.imwrite(os.path.join(save_path, f'{filename}.png'), x_t_enhanced)

                if save_noise_path is not None:
                    np.save(os.path.join(save_noise_path, f'{i}.npy'), noise)

                print(f"✅ 已保存增强版校正流生成图像: {os.path.join(save_path, f'{filename}.png')}")

class SingleFolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.image_files[idx]  # 返回图像和文件名


# 创建数据加载器的函数
def create_single_folder_dataloader(image_dir, batch_size=1, resize=256):
    transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = SingleFolderDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def main_infer(rank, world_size):
    # ###### infer ##########
    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() and opt.cuda else "cpu")
    torch.cuda.set_device(device)

    image_dir = ".\datasets/test\MRI"
    transfer_dir = './datasets/Transfer/MRI2Uto/fake_SG-cycle-3WUto'
    transfer_dir_motaiwuguan = './datasets/Transfer/MRI2Uto/shallow_motaiwuguan'
    if 'uto' in image_dir.lower():
        img = 'uto'
    elif 'mri' in image_dir.lower():
        img = 'mri'
    else:
        img = None  # 或根据需求设置默认值
        print("路径中既不包含Uto也不包含MRI")
    dataloader = create_single_folder_dataloader(image_dir)

    shared_generator = SharedGenerator(opt.input_nc, opt.output_nc, opt.mtwg_fea, opt.size, opt.depth4vit).to(device)
    checkpoint_path = 'output/shared_generator.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 处理DDP包装的模型权重（如果checkpoint是DDP保存的）
    state_dict = checkpoint.get('shared_generator', checkpoint)  # 兼容不同保存方式
    if all(key.startswith('module.') for key in state_dict.keys()):
        # 移除DDP添加的'module.'前缀
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    # 加载权重到模型
    shared_generator.load_state_dict(state_dict, strict=True)  # strict=True确保权重完全匹配
    print(f"成功从 {checkpoint_path} 加载SharedGenerator参数")


    # 打印第一批数据的形状
    for images, filenames in dataloader:
        infer(checkpoint_path='./checkpoints/v1.1-cfg/miniunet_finalepoch.pth',
              base_channels=256,
              step=2,
              num_imgs=1,
              org_img=images,
              filenames=filenames,
              y=None,  # torch.tensor(y)
              cfg_scale=5.0,
              save_path=transfer_dir,
              save_path_motaiwuguan=transfer_dir_motaiwuguan,
              device='cuda',
              shared_generator=shared_generator,
              img=img,
              world_size=world_size)


def main():
    # Determine number of available GPUs
    world_size = torch.cuda.device_count() if opt.cuda else 1
    print(f"检测到 {world_size} 个GPU")

    if world_size > 1:
        # Use multiprocessing for distributed training
        mp.spawn(
            main_worker,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        main_worker(0, world_size)
    main_infer(0, world_size)


if __name__ == '__main__':
    mp.freeze_support()
    main()

# when opt.epoch == opt.n_epochs, training of rectified flow was forbidened,
# and default parameters for rectified flow are load.
# only the generated (tranformed) images by GAN are useful, while the images generated by rectified flow
# are formalistic