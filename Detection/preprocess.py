import os
import torch
import numpy as np
import cv2
import pydicom as pdcm
from tqdm import tqdm
import torchvision.transforms.functional as F
from skimage import io

def preprocess(input_path,output_path,output_size=[224,224]):
    # 遍历输入路径下的所有文件夹
    for folder_name in tqdm(os.listdir(input_path)):
        folder_path = os.path.join(input_path, folder_name)
        target_path=os.path.join(output_path, folder_name)
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        if os.path.isdir(folder_path):
            img_path = os.path.join(folder_path, 'img.jpg')
            mask_path = os.path.join(folder_path, 'mask.png')
            bboxes_path = os.path.join(folder_path, 'bboxes.npy')

            if os.path.exists(img_path) and os.path.exists(mask_path) and os.path.exists(bboxes_path):
                # 读取图像
                # img_dcm = pdcm.dcmread(img_path)
                # img = img_dcm.pixel_array
                img=io.imread(img_path)

                # 读取掩码
                mask = io.imread(mask_path)
                
                # 读取边界框
                bboxes = np.load(bboxes_path, allow_pickle=True)

                original_size = img.shape
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

                # Normalize图像
                # img = F.normalize(torch.tensor(img, dtype=torch.float32).unsqueeze(0), [0.5], [0.5])

                # Adjust bounding boxes according to the resize
                h_scale = output_size[0] / original_size[0]
                w_scale = output_size[1] / original_size[1]
                bboxes = [(x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale) for (x1, y1, x2, y2) in bboxes]

                # 转换为tensor
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # 添加通道维度
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # 添加通道维度
                bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

                # 保存为单独的pt文件
                img_save_path = os.path.join(target_path, 'img.pt')
                mask_save_path = os.path.join(target_path, 'mask.pt')
                bboxes_save_path = os.path.join(target_path, 'bboxes.pt')
                if not os.path.exists(img_save_path):
                    torch.save(img_tensor, img_save_path)
                if not os.path.exists(mask_save_path):
                    torch.save(mask_tensor, mask_save_path)
                if not os.path.exists(bboxes_save_path):
                    torch.save(bboxes_tensor, bboxes_save_path)
                
            elif os.path.exists(img_path) and os.path.exists(bboxes_path):
                # 读取图像
                # img_dcm = pdcm.dcmread(img_path)
                # img = img_dcm.pixel_array
                img=io.imread(img_path)

                bboxes = np.load(bboxes_path, allow_pickle=True)

                original_size = img.shape
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)

                # Normalize图像
                # img = F.normalize(torch.tensor(img, dtype=torch.float32).unsqueeze(0), [0.5], [0.5])

                # Adjust bounding boxes according to the resize
                h_scale = output_size[0] / original_size[0]
                w_scale = output_size[1] / original_size[1]
                bboxes = [(x1 * w_scale, y1 * h_scale, x2 * w_scale, y2 * h_scale) for (x1, y1, x2, y2) in bboxes]

                # 转换为tensor
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # 添加通道维度
                bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32)

                # 保存为单独的pt文件
                img_save_path = os.path.join(target_path, 'img.pt')
                bboxes_save_path = os.path.join(target_path, 'bboxes.pt')
                if not os.path.exists(img_save_path):
                    torch.save(img_tensor, img_save_path)
                if not os.path.exists(bboxes_save_path):
                    torch.save(bboxes_tensor, bboxes_save_path)
                
            elif os.path.exists(img_path) and os.path.exists(mask_path):
                # 读取图像
                # img_dcm = pdcm.dcmread(img_path)
                # img = img_dcm.pixel_array
                img=io.imread(img_path)

                # 读取掩码
                mask = io.imread(mask_path)
                
             

                original_size = img.shape
                img = cv2.resize(img, output_size, interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, output_size, interpolation=cv2.INTER_NEAREST)

              
                # 转换为tensor
                img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) # 添加通道维度
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # 添加通道维度

                # 保存为单独的pt文件
                img_save_path = os.path.join(target_path, 'img.pt')
                mask_save_path = os.path.join(target_path, 'mask.pt')
              
                if not os.path.exists(img_save_path):
                    torch.save(img_tensor, img_save_path)
                if not os.path.exists(mask_save_path):
                    torch.save(mask_tensor, mask_save_path)

                
                # print(f'Saved {img_save_path}, {mask_save_path}, {bboxes_save_path}')
# if __name__ == "__main__":
#     output_size=[224,224]
#     input_path = '/home/jiayi/Baseline/segdetdata/CBIS-DDSM/Train'  # 替换为你的实际路径
#     output_path = '/home/jiayi/Baseline/segdetdata/CBIS-DDSM/Train_cache_'+str(output_size[0])
#     preprocess(input_path,output_path,output_size)
#     input_path = '/home/jiayi/Baseline/segdetdata/CBIS-DDSM/Test'  # 替换为你的实际路径
#     output_path = '/home/jiayi/Baseline/segdetdata/CBIS-DDSM/Test_cache_'+str(output_size[0])
#     preprocess(input_path,output_path,output_size)