
'''
———————————————— 模型训练函数 ————————————————
net：要训练的摩西那个
device： 训练使用的设备，默认为CPU
epochs:  训练的轮数，默认为5
batch_size：每个批次的输入样本数，默认为1
lr： 学习率，默认为0.001
val_percent：验证集所占的百分比，默认为0.1
svae_cp：是否保存检查点，默认为 True
img_scale：输入图像的缩放比例，默认为0.5
'''
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from unet import UNet
from utils import basicdataset
from train import eval_net

dir_img = 'C:\\Users\\25372\\Desktop\\dataset\\_image'
dir_mask = 'C:\\Users\\25372\\Desktop\\dataset\\_mask'
dir_checkpoint = 'C:\\Users\\25372\\Desktop\\model\\dataset\\test\\checkpoints'

''' 
包配置 
tensorboard
matplotlib
'''

'''
终端指令
python3 train.py -e 200 -b 20 -l 0.0001 -s 0.5 -v 0.3
'''

def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):
    # 构建数据集对象
    dataset = basicdataset(dir_img, dir_mask, img_scale)
    
    dt = dataset.getTensor()
    img = dt['image']
    mask = dt['mask']
    
    length = img.shape[0]
    # length = img.size(0)
    
    n_val = int(length * val_percent)
    n_train = length - n_val
    
    # 定义负责样本拆分的随机采样器
    indices = list(range(length))
    train_indices, val_indices = indices[n_val:], indices[:n_val]
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 定义数据加载器
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
    
    # 创建日志对象
    # 用于记录训练过程中的统计信息和日志。
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    # 打印训练的初步信息
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 定义一个 RMSprop 优化器
    # 传入 net.parameters() 来指定需要进行优化的模型参数
    # lr 表示学习率
    # weight_decay 表示权重衰减系数
    # momentum 表示动量
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义学习率调整器，动态的调整学习率
    # optimizer 要进行学习率调整的优化器对象
    # 如果指定的阈值检测方式是 ‘min’，
    # 即多类别分类问题的情况下，当验证指标不再下降时，学习率会被降低；
    # 如果是 ‘max’，
    # 即二元分类问题的情况下，当验证指标不再上升时，学习率会被降低。
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    # 依据模型类别数选择相应的损失函数
    # 多类别：交叉熵损失 nn.CrossEntropyLoss()
    # 单类别：带逻辑回归的二元交叉熵损失 nn.BCEWithLogitsLoss()
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()


    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        
        # tqdm 用于在命令行界面显示进度条的库
        # total 参数设置迭代的总数，在这里是训练数据集的总数 n_train
        # desc 参数设置显示在进度条前面的文本，用来表示当前是第几个 epoch
        # unit 参数设置进度条的单位，这里是 ‘img’ 表示每个迭代单位是图像
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                # 从batch中获取图像数据和真实掩码数据
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                # 将图像数据转移到指定的计算设备上
                # 并将数据类型设为 torch.float32
                imgs = imgs.to(device=device, dtype=torch.float32)
                # 根据模型的类别数 net.n_classes，
                # 确定真实掩码数据的数据类型 mask_type。
                # 如果类别数为 1，说明是二元分类问题，则将数据类型设置为 torch.float32，
                # 表示每个像素对应一个二分类结果；
                # 否则，将数据类型设置为 torch.long，表示掩码数据是一组分类标签
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                # 将真实掩码数据转移到指定的计算设备上
                # 并将数据类型设为 mask_type
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # 对图像掩码进行预测
                masks_pred = net(imgs)
                # 计算损失
                loss = criterion(masks_pred, true_masks)
                # 保存该集损失，用于计算总损失
                epoch_loss += loss.item()
                # 对训练过程中的损失值进行记录和可视化
                writer.add_scalar('Loss/train', loss.item(), global_step)

                # 在进度条中显示当前batch的损失值
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # 清空梯度信息
                optimizer.zero_grad()
                # 反向传播计算梯度
                # 计算出每个可训练参数的梯度信息
                loss.backward()
                # 进行梯度裁剪，防止梯度爆炸
                # 将所有可训练参数的梯度值进行一个最大值的限制，
                # 即将梯度值裁剪到 [-0.1, 0.1] 的范围之内。
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                # 使用优化器 optimizer 更新模型的参数，
                # 利用 optimizer.step() 方法实现。
                # 根据计算得到的梯度信息以及优化算法的更新规则，对模型中的可训练参数进行更新。
                optimizer.step()
                
                # 更新进度条中已经处理的图片数量
                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    '''
                    # 可以在训练过程中实时地记录参数的权重和梯度信息，
                    # 并可视化到TensorBoard中，以便进一步分析和调试网络。
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    '''
                    val_score = eval_net(net, val_loader, device)
                    
                    # 根据验证集的得分来更新学习率。
                    # 如果验证集得分比之前保存的数值更好，则可以适当提高学习率；
                    # 反之，如果得分没有提高，则适当降低学习率。
                    # 这样可以提高模型的收敛速度，增强模型的泛化性能
                    scheduler.step(val_score)
                    # 记录新的学习率
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        # 记录在验证集上的得分
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        # 记录在验证集上的得分
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    # 可视化模型实际的预测效果
                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        ''' 保存训练过程中的模型检查点 checkpoint '''
        # 判断是否需要保存检查点
        if save_cp:
            # 尝试创建保存检查点的文件夹
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            # 使用torch.save()方法保存网络的状态字典到检查点文件中
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            # 使用日志记录器（logging）记录检查点已保存的消息
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

def get_args():
    ''' 包含命令行参数信息 '''
    ''' 生成预测掩码，通过命令函参数指定输入图像等参数 '''
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    '''
    指定训练的轮数, 默认为5
    -e
    -epochs
    '''
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    '''
    指定批次大小，默认为1
    -b
    --batch-size
    '''
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    '''
    指定学习率，默认为0.0001
    -l
    --learning-rate
    '''
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    '''
    用于指定是否从.pth文件中加载模型，默认为False。
    -f
    --load
    '''
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    '''
    指定图像的缩放因子，默认为0.5
    -s
    --scale
    '''
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    '''
    指定用作验证数据的百分比
    -v
    --validation
    '''
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()    
    
    

'''
    train_net(net=net,
              epochs=100,
              batch_size=8,
              lr=0.01,
              device=device,
              img_scale=1,
              val_percent=0.3)
'''

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)