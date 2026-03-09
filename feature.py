import torch
from tqdm import tqdm
import numpy as np
from typing import Dict, Tuple, Optional

from sklearn.cluster import MiniBatchKMeans



def extract_multi_scale_features(model: torch.nn.Module,
                                 data_loader: torch.utils.data.DataLoader,
                                 device: str):
    """
    提取多尺度特征。

    参数:
        model: 模型
        data_loader: 数据加载器
        device: 设备 'cuda' 或 'cpu'

    返回:
        features: 字典，键为特征层名称，值为该层的特征矩阵 (num_samples, feature_dim)features: 字典，键为特征层名称，值为该层的特征矩阵 (num_samples, feature_dim)
                e.g.
                    features = {
                        'layer1': np.array([[0.1, 0.2, 0.3, 0.4],  # 样本1的特征
                                            [0.5, 0.6, 0.7, 0.8],  # 样本2的特征
                                            [0.9, 1.0, 1.1, 1.2]]),  # 样本3的特征
                        'layer2': np.array([[0.01, 0.02, 0.03, 0.04, 0.05, 0.06],  # 样本1的特征
                                            [0.07, 0.08, 0.09, 0.10, 0.11, 0.12],  # 样本2的特征
                                            [0.13, 0.14, 0.15, 0.16, 0.17, 0.18]])  # 样本3的特征
                    }
        labels: 对应的样本标签 (num_samples,)
    """
    features = {}
    labels = []  # 存储样本标签

    model.eval()  # 切换到评估模式
    with torch.no_grad():  # 禁用梯度计算
        for images, batch_labels in tqdm(data_loader, desc="Extracting Features", unit="batch"):
            images = images.to(device)
            batch_labels = batch_labels.cpu().numpy()
            # 前向传播，获取模型特征字典
            _, feature_dict = model(images)

            # 提取指定层的特征
            for layer, feat in feature_dict.items():
                
                if layer not in features:
                    features[layer] = []  # 初始化该层的特征列表
                features[layer].append(feat.cpu())

            # 保存标签
            labels.append(batch_labels)

    # 拼接特征和标签
    for layer in features:
        if features[layer]:  # 确保特征不为空
            features[layer] = torch.cat(features[layer], dim=0).numpy()  # (num_samples, feature_dim)

    labels = np.concatenate(labels)  # (num_samples,)

    return features, labels



def generate_prototypes(features: Dict[str, np.ndarray],
                        labels: np.ndarray,
                        num_clusters: int):
    """
    利用 K-Means 聚类为每个特征层和每个类别生成原型。

    参数:
        features: 字典，键为特征层名称，值为该层的特征矩阵 (num_samples, feature_dim)
        labels: 样本标签数组 (num_samples,)
        num_clusters: 每个类别的原型数量

    返回:
        prototypes: 字典，键为特征层名称，值为另一个字典
                   - 内层字典的键为类别标签，值为该类的原型数组 (num_clusters, feature_dim)
    """
    prototypes = {}  # 存储所有层的原型

    # 遍历每个特征层
    for layer in tqdm(features.keys(), desc="Processing Layers", unit="layer"):
        layer_prototypes = {}  # 存储当前层的原型

        # 遍历每个类别，并使用进度条
        for cls in tqdm(np.unique(labels), desc=f"Clustering Layer: {layer}", unit="class"):
            cls_mask = (labels == cls)
            cls_feat = features[layer][cls_mask]

            if len(cls_feat) < num_clusters:
                print(f"[Warning] Class {cls} in layer '{layer}' has fewer samples ({len(cls_feat)}) than clusters ({num_clusters}). Skipping.")
                continue

            # 标准化特征
            cls_feat = (cls_feat - cls_feat.mean(axis=0)) / (cls_feat.std(axis=0) + 1e-8)

            # 使用 K-Means 聚类生成原型
            kmeans = MiniBatchKMeans(n_clusters=num_clusters,
                                     n_init=3,
                                     batch_size=256,
                                     init_size=3*256,
                                     max_iter=100,
                                     reassignment_ratio=0.01,
                                     random_state=42)
            kmeans.fit(cls_feat)  # 计算聚类中心

            # 保存聚类中心作为原型
            layer_prototypes[int(cls)] = kmeans.cluster_centers_

        # 保存当前层的原型
        prototypes[layer] = layer_prototypes

    return prototypes



def precompute_prototypes(
    global_memory_bank: Dict[int, Dict[str, Dict[int, np.ndarray]]],
    layer_names: list,
    device: str = 'cuda'
    ) -> Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    """
    预计算所有层的原型矩阵和标签，返回字典
    
    参数:
        global_memory_bank: 全局原型库
        layer_names: 需要处理的层名列表
        device: 目标设备
    
    返回:
        {
            层名: (原型张量, 原型标签),
            ...
        }
    """
    prototypes_dict = {}
    
    for layer in layer_names:
        all_prototypes = []
        all_proto_labels = []
        
        # 收集所有客户端的该层原型
        for client_mem in global_memory_bank.values():
            layer_prototypes = client_mem.get(layer, {})
            for label, prototypes in layer_prototypes.items():
                if prototypes.shape[0] > 0:
                    all_prototypes.append(prototypes)
                    all_proto_labels.extend([label] * prototypes.shape[0])
        
        if len(all_prototypes) == 0:
            prototypes_dict[layer] = (None, None)
            continue
        
        # 转换为Tensor
        prototypes_matrix = np.vstack(all_prototypes)
        prototypes_tensor = torch.tensor(prototypes_matrix, dtype=torch.float32, device=device)
        prototypes_tensor = torch.nn.functional.normalize(prototypes_tensor, p=2, dim=1)
        proto_labels = torch.tensor(all_proto_labels, dtype=torch.long, device=device)
        
        prototypes_dict[layer] = (prototypes_tensor, proto_labels)
    
    return prototypes_dict