import paddle
import os
import random
import numpy as np
import cv2
import json
import glob
import scipy.io as scio
try:
    from make_density import make_density
except:
    from work.make_density import make_density


def get_points(gt_path):
    """获取位置坐标和人数"""
    image_info = scio.loadmat(gt_path)['image_info']
    annPoints = image_info[0,0]['location'][0][0]
    person_counts = len(annPoints)
    return annPoints, person_counts
    
    
def downsample(points, size):
    """
    将裁剪的区域按照scale缩放（样本下采样）
    ：param points：人群位置坐标
    ：param scale：缩放比例
    :return 缩放后的图像，缩放后的位置坐标
    """
    for i in range(len(points)):
        points[i][1] = points[i][1] / size[0]
        points[i][0] = points[i][0] / size[1]
        
    return points
    
    
# 对读入的图像数据进行预处理
def transform_img(img):
    
    # 读入的图像数据格式是[H, W, C]
    # 使用转置操作将其变成[C, H, W]
    img = np.transpose(img, (2,0,1))
    img = img.astype('float32')
    # 将数据范围调整到[0, 1.0]之间
    img = (img - 127.5) / 128
    #img = img / 255.
    return img
    

    
def SH_data_loader(datadir, size=[512, 1024], scale=4, mode='train'):
    # size(h, w)  将datadir目录下的文件列出来，每条文件都要读入
    # datadir1 = None
    # if 'part_A_final' in datadir:
    #     datadir1 = datadir.replace("part_A_final", "part_B_final")
    # elif 'part_B_final' in datadir:
    #     datadir1 = datadir.replace("part_B_final", "part_A_final")
    # datadir1 = None
    # if datadir1 is None:
    #     filenames = glob.glob(datadir + '*.jpg')
    # else:
    #     filenames = glob.glob(datadir + '*.jpg') + glob.glob(datadir1 + '*.jpg')
    print('SH')
    filenames = glob.glob(datadir + '*.jpg')
    def reader():
        if mode == 'train':
            # 训练时随机打乱数据顺序
            random.shuffle(filenames)
       
            for img_path in filenames:
               
                gt_path = img_path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
                image = cv2.imread(img_path)
                annPoints, person_counts = get_points(gt_path)
                #print(image.shape)
                h, w = image.shape[0], image.shape[1]
                
                points = downsample(annPoints, (h / size[0], w / size[1]))
                image = cv2.resize(image, (size[1], size[0]))
                density = make_density(image, points)
                density = cv2.resize(density, (density.shape[1]//scale,density.shape[0]//scale),interpolation = cv2.INTER_CUBIC)*(scale*scale)
                density = density / (density.sum()+0.000000001) * person_counts  # 归一化
                
                prob = np.random.uniform(0, 1)
                if prob < 0.35:
                    image = cv2.flip(image, 1) # 水平翻转
                    density = cv2.flip(density, 1)
                
                image = transform_img(image)
                
               
                yield image, density.reshape(1, density.shape[0], density.shape[1])
               
        elif mode == 'val':
            #random.shuffle(filenames)
            for img_path in filenames:
                #print(img_path)
                gt_path = img_path.replace('images', 'ground_truth').replace('IMG', 'GT_IMG').replace('jpg', 'mat')
                image = cv2.imread(img_path)
                annPoints, person_counts = get_points(gt_path)
                #print(image.shape)
                h, w = image.shape[0], image.shape[1]
                
                points = downsample(annPoints, (h / size[0], w / size[1]))
                image = cv2.resize(image, (size[1], size[0]))
                density = make_density(image, points)
                density = cv2.resize(density, (density.shape[1]//scale,density.shape[0]//scale),interpolation = cv2.INTER_CUBIC)*(scale*scale)
                density = density / (density.sum()+0.000000001) * person_counts  # 归一化
                
                prob = np.random.uniform(0, 1)
                if prob < 0.35:
                    image = cv2.flip(image, 1) # 水平翻转
                    density = cv2.flip(density, 1)
                image = transform_img(image)
                
                yield image, density.reshape(1, density.shape[0], density.shape[1])
        
    return reader
    


if __name__ == '__main__':
    pass
        