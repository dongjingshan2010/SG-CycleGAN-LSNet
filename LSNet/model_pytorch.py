import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import transforms as T


# helper functions


def default(val, def_val):
    return def_val if val is None else val


def flatten(t):
    return t.reshape(t.shape[0], -1)


def singleton(cache_key):
    def inner_fn(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            instance = getattr(self, cache_key)
            if instance is not None:
                return instance

            instance = fn(self, *args, **kwargs)
            setattr(self, cache_key, instance)
            return instance

        return wrapper

    return inner_fn


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def MaybeSyncBatchnorm(is_distributed=None):
    is_distributed = default(is_distributed, dist.is_initialized() and dist.get_world_size() > 1)
    return nn.SyncBatchNorm if is_distributed else nn.BatchNorm1d


# loss fn
def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# augmentation utils
class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


# exponential moving average


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# MLP class for projector and predictor


def MLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )


def SimSiamMLP(dim, projection_size, hidden_size=4096, sync_batchnorm=None):
    return nn.Sequential(
        nn.Linear(dim, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, hidden_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size, bias=False),
        MaybeSyncBatchnorm(sync_batchnorm)(projection_size, affine=False)
    )


# a wrapper class for the base neural network
# will manage the interception of the hidden layer output
# and pipe it into the projecter and predictor nets


class NetWrapper(nn.Module):
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2, use_simsiam_mlp=False,
                 sync_batchnorm=None):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.use_simsiam_mlp = use_simsiam_mlp
        self.sync_batchnorm = sync_batchnorm

        self.hidden = {}
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def _get_projector(self, hidden):
        _, dim = hidden.shape
        create_mlp_fn = MLP if not self.use_simsiam_mlp else SimSiamMLP
        projector = create_mlp_fn(dim, self.projection_size, self.projection_hidden_size,
                                  sync_batchnorm=self.sync_batchnorm)
        return projector.to(hidden)

    def get_representation(self, x):
        pred, c_results = self.net(x)
        return pred, c_results

        # if self.layer == -1:
        #     return self.net(x)
        #
        # if not self.hook_registered:
        #     self._register_hook()
        #
        # self.hidden.clear()
        # _ = self.net(x)
        # hidden = self.hidden[x.device]
        # self.hidden.clear()
        #
        # assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        # return hidden

    def forward(self, x, return_projection=True):
        representation, c_results = self.get_representation(x)

        if not return_projection:
            return representation

        projector = self._get_projector(representation)
        projection = projector(representation)
        return projection, c_results


# main class
class Model(nn.Module):
    def __init__(
            self,
            net,
            net_sparse,
            image_size,
            hidden_layer,
            projection_size,
            sparse_projection_size,
            projection_hidden_size,
            moving_average_decay=0.99,
            use_momentum=True,
            sync_batchnorm=None,
            class_weights=None,  # 新增：类别权重参数
    ):
        super().__init__()

        # default SimCLR augmentation

        DEFAULT_AUG = torch.nn.Sequential(
            # T.Scale(int(1.2*image_size)),
            # RandomApply(
            #     T.ColorJitter(0.8, 0.8, 0.8, 0.2),
            #     p = 0.3
            # ),
            # T.RandomGrayscale(p=0.2),
            # T.RandomHorizontalFlip(),
            # RandomApply(
            #     T.GaussianBlur((3, 3), (1.0, 2.0)),
            #     p = 0.2
            # ),
            T.RandomRotation(degrees=45),  # , fill=(255, 255, 255)
            T.RandomResizedCrop(size=image_size, scale=(0.8, 0.99)),  # , ratio=(0.99, 1.0)
            # T.RandomResizedCrop((image_size, image_size)),
            # T.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225])),
        )

        augment_fn = None
        self.augment1 = default(augment_fn, DEFAULT_AUG)
        self.augment2 = default(augment_fn, self.augment1)

        self.online_encoder = NetWrapper(
            net,
            projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm
        )

        self.sparse_encoder = NetWrapper(
            net_sparse,
            sparse_projection_size,
            projection_hidden_size,
            layer=hidden_layer,
            use_simsiam_mlp=not use_momentum,
            sync_batchnorm=sync_batchnorm
        )

        # self.gen_trans = gen_trans

        self.use_momentum = use_momentum
        self.target_encoder = None
        self.target_gen_vit = None

        self.online_predictor = MLP(projection_size, sparse_projection_size, projection_hidden_size)
        self.sparse_predictor = MLP(sparse_projection_size, sparse_projection_size, projection_hidden_size)

        # get device of network and make wrapper same device
        device = get_module_device(net)
        self.to(device)

        # 保存类别权重
        self.class_weights = class_weights
        self.use_weighted_loss = class_weights is not None

        # 根据是否使用权重初始化损失函数
        if self.use_weighted_loss:
            # 确保权重张量在正确的设备上
            weights_tensor = torch.tensor(self.class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor).to(device)
        else:
            self.criterion = nn.CrossEntropyLoss().to(device)

        # send a mock image tensor to instantiate singleton parameters
        # self.forward(torch.randn(2, 3, image_size, image_size, device=device), torch.randn(2, 3, image_size, image_size, device=device))

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def set_class_weights(self, class_weights):
        """
        动态设置类别权重
        参数:
            class_weights: 类别权重列表，如 [1.0, 2.0, 0.5] 或 None
        """
        device = get_module_device(self.online_encoder.net)

        if class_weights is None:
            # 重置为标准CrossEntropyLoss
            self.criterion = nn.CrossEntropyLoss().to(device)
            self.use_weighted_loss = False
            print("已重置为无权重损失函数")
        else:
            # 应用新的类别权重
            weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor).to(device)
            self.use_weighted_loss = True
            print(f"已设置类别权重: {class_weights}")

    def forward(
            self,
            img,
            label,
            return_embedding=False,
            return_projection=True
    ):
        assert not (self.training and img.shape[
            0] == 1), 'you must have greater than 1 sample when training, due to the batchnorm in the projection layer'

        if return_embedding:
            return self.online_encoder(img1, return_projection=return_projection)

        image_us = img  # self.augment1(img), self.augment2(img)

        b, _, _, _ = image_us.shape

        online_proj, c_results = self.online_encoder(image_us)  # # provide back gradients
        online_proj = self.online_predictor(online_proj)
        loss_CE = self.criterion(c_results, label)
        # 定位第一个Attention层
        first_attn_layer = self.online_encoder.net.mvit[0].transformer.branch1_layers[0][0].fn
        scd_attn_layer = self.online_encoder.net.mvit[1].transformer.branch1_layers[0][0].fn
        trd_attn_layer = self.online_encoder.net.mvit[2].transformer.branch1_layers[0][0].fn

        # 确保注意力分数已被保存
        assert hasattr(first_attn_layer, 'attn_score'), "请先在Attention类中保存attn_score"
        # 计算loss_CE对注意力分数的梯度
        loss_CE.backward(retain_graph=True)  # retain_graph=True保留计算图（如果后续还需反向传播）
        attn_score_grad0 = first_attn_layer.attn_score.grad.squeeze()  # 获取大模型梯度
        attn_score_grad1 = scd_attn_layer.attn_score.grad.squeeze()  # 获取大模型梯度
        attn_score_grad2 = trd_attn_layer.attn_score.grad.squeeze()  # 获取大模型梯度

        sparse_proj, c_results_sparse = self.sparse_encoder(image_us)  # sparsifing based on generated grad
        sparse_proj = self.sparse_predictor(sparse_proj)
        loss_CE2 = self.criterion(c_results_sparse, label)
        gen_grad_map0 = self.sparse_encoder.net.mvit[0].transformer.first_attn_score_grad
        gen_grad_map1 = self.sparse_encoder.net.mvit[1].transformer.first_attn_score_grad
        gen_grad_map2 = self.sparse_encoder.net.mvit[2].transformer.first_attn_score_grad

        loss_grad = (F.mse_loss(gen_grad_map0.squeeze(), attn_score_grad0) + F.mse_loss(gen_grad_map1.squeeze(),
                                                                                        attn_score_grad1)
                     + F.mse_loss(gen_grad_map2.squeeze(), attn_score_grad2))

        lossfn = loss_fn(online_proj, sparse_proj).mean()

        # # 2. 分类logits蒸馏 - 使用KL散度（标准做法）
        # # 现跑的没使用该损失。
        # temperature = 3.0
        # teacher_logits = c_results / temperature
        # student_logits = c_results_sparse / temperature
        # soft_targets = F.softmax(teacher_logits, dim=-1)
        # soft_prob = F.log_softmax(student_logits, dim=-1)
        # loss_logits_distill = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)

        # loss = loss_CE + loss_CE2 #+ loss_grad + loss_logits_distill

        loss = lossfn + loss_CE + loss_CE2 + loss_grad

        return loss