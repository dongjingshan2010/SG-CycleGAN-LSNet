import numpy as np
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def create_random_maskfromIndex(changeAdj, shape=(224, 224), mask_ratio=0.1):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_ratio: the ratio of masked area
    :return:
    '''
    h, w = shape[0], shape[1]
    mask_format = np.zeros(h * w)
    mask_format[: int(h * w * mask_ratio)] = 1
    mask_format_zero = np.zeros(h * w)

    mask_matrix = []
    for e in changeAdj:
        if e == 1:
            np.random.shuffle(mask_format)
            mask = mask_format.reshape(h, w)

            mask_temp = mask + mask.T #mask.dot(mask.T)  #
            np.putmask(mask_temp, mask_temp >= 1, 1)
            # rate = np.sum(mask_temp == 1)/(h * w)
            # print('rate=%f'%rate)
            mask_matrix.append(mask_temp)
        else:
            mask_matrix.append(mask_format_zero.reshape(h, w))
    mask_matrix = np.array(mask_matrix, dtype=int)
    np.array(mask_matrix, np.bool)

    return mask_matrix


def create_random_mask_duichen(shape=(512, 224, 224), mask_ratio=0.1):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 0 means to mask, 1 means not to mask
    :param mask_ratio: the ratio of masked area
    :return:
    '''
    bsz, h, w = shape[0], shape[1], shape[2]
    mask_format = np.zeros(h * w)
    mask_format[: int(h * w * mask_ratio)] = 1

    mask_matrix = []
    for _ in range(bsz):
        np.random.shuffle(mask_format)
        mask_matrix.append(mask_format.reshape(h, w))
    mask_matrix = np.array(mask_matrix, dtype=int)
    trans_mask_matrix = np.transpose(mask_matrix, (0, 2, 1))

    mask_temp = mask_matrix+trans_mask_matrix  #mask_matrix.dot(trans_mask_matrix)
    np.putmask(mask_temp, mask_temp >= 1, 1)

    mask_temp = 1- mask_temp

    return mask_temp


def create_random_mask(shape=(512, 224, 224), mask_ratio=0.1):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_ratio: the ratio of masked area
    :return:
    '''
    bsz, h, w = shape[0], shape[1], shape[2]
    mask_format = np.zeros(h * w)
    mask_format[: int(h * w * mask_ratio)] = 1

    mask_matrix = []
    for _ in range(bsz):
        np.random.shuffle(mask_format)
        mask_matrix.append(mask_format.reshape(h, w))
    mask_matrix = np.array(mask_matrix, dtype=int)

    return mask_matrix


def create_rectangle_mask(shape=(512, 224, 224), mask_shape=(16, 16)):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_shape: The shape size of the masked area
    :return:
    '''
    bsz, h, w = shape[0], shape[1], shape[2]
    assert h == w
    xs = np.random.randint(w, size=bsz)
    ys = np.random.randint(h, size=bsz)

    mask_matrix = []
    for i in range(bsz):
        x = xs[i]
        y = ys[i]
        mask_format = np.zeros((h, w))
        mask_format[x: x + mask_shape[0], y: y + mask_shape[1]] = 1
        mask_matrix.append(mask_format)

    mask_matrix = np.array(mask_matrix, dtype=int)

    return mask_matrix


def create_multirectangle_mask(number4rec, shape=(512, 224, 224), mask_shape=(16, 16)):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_shape: The shape size of the masked area
    :return:
    '''
    _, h, w = shape[0], shape[1], shape[2]
    bsz = number4rec
    assert h == w
    xs = np.random.randint(w, size=bsz)
    ys = np.random.randint(h, size=bsz)

    mask_format = torch.zeros((h, w))
    for i in range(bsz):
        x = xs[i]
        y = ys[i]

        mask_format[x: x + mask_shape[0], y: y + mask_shape[1]] = 1
        # mask_matrix.append(mask_format)

    mask_matrix = mask_format

    return mask_matrix


def create_rectangle_mask2coords(selected_rows, shape=(1, 224, 224), mask_shape=(16, 16)):
    '''
    Get a mask image with shape (224,224). 512 represents batchsize.
    :param shape: image shape (batchsize, h, w), which are 1 and 0, 1 means to mask, 0 means not to mask
    :param mask_shape: The shape size of the masked area
    :return:
    '''
    bsz4rows = selected_rows.size(0)
    _, h, w = shape[0], shape[1], shape[2]
    mask_format = torch.zeros((h, w))
    for i in range(bsz4rows):
        x, y = selected_rows[i]
        top = max(0, y - mask_shape[0] // 2)
        bottom = min(shape[-2], y + mask_shape[0] // 2 + 1)
        left = max(0, x - mask_shape[1] // 2)
        right = min(shape[-1], x + mask_shape[1] // 2 + 1)
        top, bottom, left, right = int(top), int(bottom), int(left), int(right)
        mask_format[top:bottom, left:right] = 1

    return mask_format


def create_bond_mask2coords(coords, selected_rows, shape=(1, 224, 224)):
    bsz4rows = selected_rows.size(0)
    _, h, w = shape[0], shape[1], shape[2]
    mask_format = torch.zeros((h, w))
    for i in range(bsz4rows):
        onecoord = selected_rows[i]
        distances = torch.sqrt(torch.sum((coords - onecoord) ** 2, dim=1))
        distances[distances == 0] = float('inf')
        min_index = torch.argmin(distances)

        nearestpoint = coords[min_index]

        top = min(onecoord[0], nearestpoint[0])
        bottom = max(onecoord[0], nearestpoint[0])
        left = min(onecoord[1], nearestpoint[1])
        right = max(onecoord[1], nearestpoint[1])
        top = max(0, top-2)
        bottom = min(shape[-2], bottom + 2)
        left = max(0, left-2)
        right = min(shape[-1], right + 2)

        top, bottom, left, right = int(top), int(bottom), int(left), int(right)
        mask_format[top:bottom, left:right] = 1
    return mask_format

import random
def create_subgraph_mask2coords(coords, selected_rows, shape=(1, 224, 224)):
    bsz4rows = selected_rows.size(0)
    _, h, w = shape[0], shape[1], shape[2]
    mask_format = torch.zeros((h, w))
    temp = coords.size(0)
    if temp < 5:
        number4adjnotes = temp
    else:
        number4adjnotes = random.randint(2, 5)
    for i in range(bsz4rows):
        onecoord = selected_rows[i]
        distances = torch.sqrt(torch.sum((coords - onecoord) ** 2, dim=1))
        distances[distances == 0] = float('inf')
        _, min_index = torch.topk(distances, k=number4adjnotes, largest=False)

        nearestpoint = coords[min_index]

        # aa = nearestpoint[:, 0]

        top = min(nearestpoint[:, 0])
        bottom = max(nearestpoint[:, 0])
        left = min(nearestpoint[:, 1])
        right = max(nearestpoint[:, 1])
        # top = max(0, top-2)
        # bottom = min(shape[-2], bottom + 2)
        # left = max(0, left-2)
        # right = min(shape[-1], right + 2)

        top, bottom, left, right = int(top), int(bottom), int(left), int(right)
        mask_format[top:bottom, left:right] = 1
    return mask_format
