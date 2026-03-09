import os
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from typing import Literal

class FLDataset(Dataset):
    def __init__(self,
                 client_dir: str,
                 task_type: Literal[-2, -1, 0, 1, 2, 3, 4] = 0,
                 split: Literal["train", "test"] = "train"):
        """
        单个客户端的数据集类

        参数:
        - client_dir: 客户端数据目录路径
        - task_type: 任务类型
            -2: 返回task1和task4的图像及task0的标签,
            -1: 返回task1和task3的图像及task0的标签,
            0: 三分类,
            1: 正常/肿瘤分类,
            2: 肿瘤分割,
            3: 胶质瘤/转移瘤分类(使用task3数据),
            4: 胶质瘤/转移瘤分类(使用task1数据)
        - split: 数据拆分(train/test)
        """
        self.client_dir = client_dir
        self.task_type = task_type
        self.split = split
        self.samples = []

        # 根据任务类型初始化
        if task_type in [0, 1]:
            self._init_classification_task()
        elif task_type == 2:
            self._init_task2()
        elif task_type == 3:
            self._init_task3()
        elif task_type == -1:
            self._init_combined()
        elif task_type == 4:  
            self._init_task4()
        elif task_type == -2:  
            self._init_combined_task4()
        else:
            raise ValueError("任务类型必须是[-1, 0, 1, 2, 3]")

    def _init_classification_task(self):
        """初始化分类任务数据"""
        # 三分类标签映射
        label_map_3class = {
            "Normal": 0,
            "Glioma": 1,
            "BM": 2
        }
        
        # 二分类标签映射
        label_map_2class = {
            "Normal": 0,
            "Glioma": 1,
            "BM": 1
        }
        
        # 根据任务类型选择映射方式
        label_map = label_map_3class if self.task_type == 0 else label_map_2class
        
        for label_folder in os.listdir(os.path.join(self.client_dir, self.split)):

            if label_folder not in label_map.keys():
                continue
                
            label_dir = os.path.join(self.client_dir, self.split, label_folder)
            if not os.path.isdir(label_dir):
                continue
                
            for sample_id in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                task_file = os.path.join(sample_path, "task1", f"{sample_id}_task1.npy")
                
                if os.path.exists(task_file):
                    self.samples.append({
                        "data_path": task_file,
                        "label": label_map[label_folder]  # 应用选择的标签映射
                    })

    def _init_task3(self):
        """初始化肿瘤子类别分类数据"""
        label_map = {
            "Glioma": 0,
            "BM": 1
        }

        for label_folder in os.listdir(os.path.join(self.client_dir, self.split)):
            if label_folder not in label_map.keys():
                continue

            label_dir = os.path.join(self.client_dir, self.split, label_folder)
            if not os.path.isdir(label_dir):
                continue

            for sample_id in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                task_file = os.path.join(sample_path, "task3", f"{sample_id}_task3.npy")
                
                if os.path.exists(task_file):
                    self.samples.append({
                        "data_path": task_file,
                        "label": label_map[label_folder]  
                    })
    def _init_task4(self):
        """初始化肿瘤子类别分类(使用task1数据)"""
        label_map = {
            "Glioma": 0,
            "BM": 1
        }

        # 只处理Glioma和BM样本
        for label_folder in label_map.keys():
            label_dir = os.path.join(self.client_dir, self.split, label_folder)
            if not os.path.isdir(label_dir):
                continue

            for sample_id in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                
                # 使用task1的数据文件
                task_file = os.path.join(sample_path, "task1", f"{sample_id}_task1.npy")
                
                if os.path.exists(task_file):
                    self.samples.append({
                        "data_path": task_file,
                        "label": label_map[label_folder]
                    })

    def _init_combined(self):
        label_map_3class = {"Normal": 0, "Glioma": 1, "BM": 2}
        for label_folder in os.listdir(os.path.join(self.client_dir, self.split)):

            if label_folder not in label_map_3class.keys():
                continue
                
            label_dir = os.path.join(self.client_dir, self.split, label_folder)
            if not os.path.isdir(label_dir):
                continue
                
            for sample_id in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_id)
                if not os.path.isdir(sample_path):
                    continue
                task1_file = os.path.join(sample_path, "task1", f"{sample_id}_task1.npy")
                task3_file = os.path.join(sample_path, "task3", f"{sample_id}_task3.npy")
                    
                if os.path.exists(task1_file) and os.path.exists(task3_file):
                    self.samples.append({
                        "task1_path": task1_file,
                        "task3_path": task3_file,
                        "task0_label": label_map_3class[label_folder]
                    })

    def _init_combined_task4(self):
        """初始化组合任务数据（task1 + task4）"""
        label_map_3class = {"Normal": 0, "Glioma": 1, "BM": 2}
        for label_folder in os.listdir(os.path.join(self.client_dir, self.split)):

            if label_folder not in label_map_3class.keys():
                continue

            label_dir = os.path.join(self.client_dir, self.split, label_folder)
            if not os.path.isdir(label_dir):
                continue

            for sample_id in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_id)
                if not os.path.isdir(sample_path):
                    continue

                # 加载task1和task4的数据文件
                task1_file = os.path.join(sample_path, "task1", f"{sample_id}_task1.npy")
                task4_file = os.path.join(sample_path, "task1", f"{sample_id}_task1.npy")  # task4使用task1的数据

                if os.path.exists(task1_file) and os.path.exists(task4_file):
                    self.samples.append({
                        "task1_path": task1_file,  # task1数据路径
                        "task4_path": task4_file,  # task4数据路径（与task1相同）
                        "task0_label": label_map_3class[label_folder]  # 三分类标签
                    })

    def _process_mask(self, mask: np.ndarray) -> torch.Tensor:
        """处理分割标签为多通道格式"""
        # 创建WT和TC标签
        WT_Label = mask.copy()
        WT_Label[(mask == 1) | (mask == 2) | (mask == 4)] = 1.0
        WT_Label[WT_Label != 1] = 0.0
        
        TC_Label = mask.copy()
        TC_Label[(mask == 1) | (mask == 4)] = 1.0
        TC_Label[TC_Label != 1] = 0.0
        
        # 合并通道并转换维度
        nplabel = np.stack([WT_Label, TC_Label], axis=-1)  # (H,W,2)
        
        return torch.from_numpy(nplabel.transpose(2,0,1)).float()  # (2,H,W)
    
    def _init_task2(self):
        """初始化分割任务数据"""
        for label in ["Glioma", "BM"]: # 只处理肿瘤样本
            label_dir = os.path.join(self.client_dir, self.split, label)
            if not os.path.exists(label_dir):
                continue
                
            for sample_id in os.listdir(label_dir):
                sample_path = os.path.join(label_dir, sample_id)
                task2_dir = os.path.join(sample_path, "task2")
                
                if os.path.exists(task2_dir):
                    slices = [f for f in os.listdir(task2_dir) if f.endswith("_image.npy")]
                    for img_file in slices:
                        base_name = img_file.replace("_image.npy", "")
                        mask_file = f"{base_name}_mask.npy"
                        mask_path = os.path.join(task2_dir, mask_file)

                        if self.split == "train":
                            roi_file = f"{base_name}_roi.npy"
                            roi_path = os.path.join(task2_dir, roi_file)

                        if os.path.exists(mask_path):
                            if self.split == "train":
                                self.samples.append({
                                    "data_path": os.path.join(task2_dir, img_file),
                                    "mask_path": mask_path,
                                    "roi_path": roi_path
                                })
                            else:
                                self.samples.append({
                                    "data_path": os.path.join(task2_dir, img_file),
                                    "mask_path": mask_path
                                })
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if self.task_type in [0, 1, 4]:
            # 分类任务
            item = self.samples[index]
            data = np.load(item["data_path"])  # (D,H,W,2)
            
            # 转换为 (C,D,H,W)
            data = torch.from_numpy(data).permute(3,0,1,2).float()
            return data, item["label"]
        
        elif self.task_type == 2:
            # 分割任务
            item = self.samples[index]
            image = np.load(item["data_path"])  # (H,W,2)
            mask = np.load(item["mask_path"])    # (H,W)

            # 转换图像格式
            image = torch.from_numpy(image.transpose(2,0,1)).float()  # (2,H,W)
            
            # 处理mask
            processed_mask = self._process_mask(mask)

            if self.split == "train":
                roi = np.load(item['roi_path'])
                roi = torch.from_numpy(roi).float()
                return image, processed_mask, roi

            return image, processed_mask

        elif self.task_type == 3:
            item = self.samples[index]
            data = np.load(item["data_path"])  # (4,D,H,W)
            data = torch.from_numpy(data).float()
            return data, item['label']

        elif self.task_type == -1:
            item = self.samples[index]
            # 加载task1图像 (D, H, W, 2) -> (2, D, H, W)
            task1_data = torch.from_numpy(np.load(item["task1_path"])).permute(3,0,1,2).float()
            # 加载task3图像 (4, D, H, W)
            task3_data = torch.from_numpy(np.load(item["task3_path"])).float()
            return task1_data, task3_data, item["task0_label"]

        elif self.task_type == -2:  # 新增-2任务类型
            item = self.samples[index]
            # 加载task1图像 (D, H, W, 2) -> (2, D, H, W)
            task1_data = torch.from_numpy(np.load(item["task1_path"])).permute(3,0,1,2).float()
            # 加载task4图像（与task1相同）
            task4_data = torch.from_numpy(np.load(item["task4_path"])).permute(3,0,1,2).float()
            return task1_data, task4_data, item["task0_label"]
            

def LoadData(root_dir, task_type, split, merge):
    all_clients_dataset = []
    for client_folder in os.listdir(root_dir):
            client_path = os.path.join(root_dir, client_folder)
            if not os.path.isdir(client_path):
                continue
            client_dataset = FLDataset(
                client_dir=client_path,
                task_type=task_type,
                split=split
            )
            all_clients_dataset.append(client_dataset)
    if merge == False:
        return all_clients_dataset
    else:
        return ConcatDataset(all_clients_dataset)

