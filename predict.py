import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils import plot_img_and_mask
from utils import basicdataset

'''
python3 predict.py -m C:\\Users\\25372\\Desktop\\Unet_12\\predict_test\\model.pth -i C:\\Users\\25372\\Desktop\\Unet_12\\predict_test\\inpt_1.jpg
'''

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    # 将网络设置为评估模式
    net.eval()
    # 使用预处理函数 BasicDataset.preprocess 对输入图像进行预处理，
    # 并将其转换为PyTorch张量
    img = torch.from_numpy(basicdataset.preprocess(full_img, scale_factor))
    # 将输入图像的维度扩展为(batch_size, channels, height, width)
    img = img.unsqueeze(0)
    # 将图像张量传输到指定的设备（通常是CPU或GPU）
    img = img.to(device=device, dtype=torch.float32)

    # 使用无梯度计算的上下文环境
    with torch.no_grad():
        # 将图像张量输入神经网络并获取输出
        output = net(img)

        # 依据不同的分类类别选择分类函数
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        # 将概率张量从(batch_size, n_classes, height, width)
        # 调整为(n_classes, height, width)
        probs = probs.squeeze(0)
        
        # 创建一个图像转换函数，将概率张量转换回图像
        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        # 调用图像转换函数将概率张量转换为PIL图像，再将其转换回张量
        probs = tf(probs.cpu())
        
        # 将概率张量从(batch_size, channels, height, width)
        # 调整为(height, width)
        full_mask = probs.squeeze().cpu().numpy()

    # 布尔掩码，其中像素值大于out_threshold被视为对象（前景），反之被视为背景
    return full_mask > out_threshold

def get_args():
    ''' 包含命令行参数信息 '''
    ''' 生成预测掩码，通过命令函参数指定输入图像等参数 '''
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    '''
    指定模型文件的保存位置
    --model MODEL.pth
    -m model.pth
    '''
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help="Specify the file in which the model is stored")
    '''
    指定输入图像文件的路径和名称
    --input input1.png input2.png
    -i path/input1.png path/input2.png
    '''
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    '''
    指定生成的掩码图像的路径和文件名
    --output output1.png output2.png
    -o path/output1.png path/output2.png
    '''
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    '''
    处理输入图像时将每个输入图像及其生成的掩码逐一可视化
    --viz
    -v
    '''
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    '''
    表示在处理输入图像时不保存生成的掩码
    -no-save
    -n
    '''
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    '''
    指定生成掩码时采用二值化的阈值
    --mask-threshold
    -t
    '''
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    '''
    指定输入图像缩小的比例，以便更快地处理大图像
    --scale
    -s
    '''
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)
    '''
    参考指令↓
    [-h] [--model FILE] --input INPUT [INPUT ...]
    [--output INPUT [INPUT ...]] [--viz] [--no-save]
    [--mask-threshold MASK_THRESHOLD] [--scale SCALE]
    
    python image_processing_script.py --model my_model.pth --input input1.jpg input2.png input3.jpeg \ 
    --output output_dir/output1.png output_dir/output2.png output_dir/output3.png \ 
    --mask-threshold 0.7 --scale 0.25
    '''
    return parser.parse_args()


def get_output_filenames(args):
    # 根据命令函参数确定输入文件名
    in_files = args.input
    out_files = []
    
    if not args.output:
        # 如果没有指定输入，则从文件路径中分离出文件名和扩展名，
        # 然后使用"_OUT"作为后缀，生成一个新的文件名
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    # 读取命令行参数
    args = get_args()
    # 指定输入路径
    in_files = args.input
    # 指定输出路径
    out_files = get_output_filenames(args)
    # 构建网络
    net = UNet(n_channels=3, n_classes=1)
    # 加载模型参数
    logging.info("Loading model {}".format(args.model))
    
    # 检测是否有可使用的GPU
    # 并根据需求调整设备模式
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 记录正在加载的模型文件的路径
    logging.info(f'Using device {device}')
    # 将模型移动到所选设备上
    net.to(device=device)
    # 加载模型的状态字典
    # 并通过map_location=device参数将模型移动到所选设备上
    net.load_state_dict(torch.load(args.model, map_location=device))

    # 记录模型已经加载
    logging.info("Model loaded !")

    for i, fn in enumerate(in_files):
        # 记录当前正在预测的图像文件名
        logging.info("\nPredicting image {} ...".format(fn))
        # 打开图像文件
        img = Image.open(fn)
        # 预测图像
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        if not args.no_save:
            # 读取输出文件名
            out_fn = out_files[i]
            # 将mask转为图片
            result = mask_to_image(mask)
            # 保存图片
            result.save(out_files[i])
            # 记录文件已保存
            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            # 成功保存掩膜的文件路径
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            # 可视化预测结果
            plot_img_and_mask(img, mask)