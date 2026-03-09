import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import torch.nn.functional as F

def contrastive_layer_loss(
    layer_features: torch.Tensor,
    labels: torch.Tensor,
    prototypes_tensor: Optional[torch.Tensor],
    proto_labels: Optional[torch.Tensor],
    tau: float = 0.5,
    epsilon: float = 1e-8
) -> torch.Tensor:
    """
    计算单层对比损失
    
    参数:
        layer_features: (B, D)
        labels: (B,)
        prototypes_tensor: (P, D) 或 None
        proto_labels: (P,) 或 None
        tau: 温度系数
        epsilon: 数值稳定项
    
    返回:
        对比损失标量
    """
    if prototypes_tensor is None or proto_labels is None:
        return torch.tensor(0.0, device=layer_features.device)
    
    # 特征归一化
    layer_features = torch.nn.functional.normalize(layer_features, p=2, dim=1)
    
    # 计算相似度矩阵 (B, P)
    sim_matrix = torch.mm(layer_features, prototypes_tensor.T)
    
    # 构建正样本掩码 (B, P)
    pos_mask = proto_labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # 计算损失
    exp_sim = torch.exp(sim_matrix / tau)
    positive_sum = (exp_sim * pos_mask).sum(dim=1)  # (B,)
    total_sum = exp_sim.sum(dim=1)                  # (B,)
    
    ratio = (positive_sum) / (total_sum + epsilon)
    loss = -torch.log(ratio)
    
    return loss.mean()



def contrastive_loss(
    features_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    precomputed_prototypes: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    tau: float
    ) -> torch.Tensor:
    """
    计算多层对比损失的平均值
    
    参数:
        features_dict: 各层特征字典 {层名: (B, D)}
        labels: 标签 (B,)
        precomputed_prototypes: 预计算的原型字典
        tau: 温度系数
    
    返回:
        平均对比损失标量
    """
    total_loss = 0.0
    valid_layer_count = 0
    
    for layer_name, layer_features in features_dict.items():
        # 获取该层预计算的原型
        prototypes_tensor, proto_labels = precomputed_prototypes.get(layer_name, (None, None))
        
        # 计算损失
        layer_loss = contrastive_layer_loss(
            layer_features=layer_features,
            labels=labels,
            prototypes_tensor=prototypes_tensor,
            proto_labels=proto_labels,
            tau=tau
        )
        
        # 累加损失
        valid_layer_count += 1
        total_loss += layer_loss
    
    return total_loss / valid_layer_count if valid_layer_count > 0 else torch.tensor(0.0, device=labels.device)



class SegLoss(nn.Module):
    def __init__(self, mu=0.6, smooth=1e-5, reduction='mean'):
        """
        多分类 ROI 加权 Dice Loss

        Args:
            mu: ROI 部分的权重系数，默认 0.6
            smooth: 避免除零错误
            reduction: 'mean'（默认）或 'sum'
        """
        super(SegLoss, self).__init__()
        self.mu = mu
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target, roi_mask):
        """
        Args:
            input: 模型输出 (B, C, H, W) (未经过 Sigmoid)
            target: One-hot 编码的真实标签 (B, C, H, W)
            roi_mask: ROI 区域掩膜 (B, H, W), 1 表示 ROI, 0 表示 non-ROI
        """
        # BCE 损失（逐通道计算）
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')  # (B, C, H, W)
        bce_loss = bce_loss.sum(dim=(2, 3))  # (B, C)
        bce_loss = bce_loss.sum(dim=1)  # (B,)

        # Dice 损失（多分类 ROI 加权）
        dice_loss = self._multiclass_dice_loss(input, target, roi_mask)  # (B,)

        # 总损失 = BCE + Dice
        total_loss = 0.0 * bce_loss + dice_loss

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        else:
            return total_loss
        
    def _multiclass_dice_loss(self, input, target, roi_mask):

        # 输入处理
        predict = torch.sigmoid(input)  # (B, C, H, W)
        target = target.float()  # 确保为浮点类型

        # 扩展 roi_mask 维度以适配 (B, C, H, W)
        roi_mask = roi_mask.unsqueeze(1)  # (B, 1, H, W)
        non_roi_mask = 1 - roi_mask  # 非 ROI 部分

        # 计算 Dice loss 分子（交集）
        intersection_roi = torch.sum(predict * target * roi_mask, dim=(2, 3))  # 计算 ROI 区域的 p * q
        intersection_non_roi = torch.sum(predict * target * non_roi_mask, dim=(2, 3))  # 计算非 ROI 区域的 p * q

        # 计算 Dice loss 分母（并集）
        union_roi = torch.sum((predict + target) * roi_mask, dim=(2, 3))  # 计算 ROI 区域的 p + q
        union_non_roi = torch.sum((predict + target) * non_roi_mask, dim=(2, 3))  # 计算非 ROI 区域的 p + q

        # 计算 ROI 加权 Dice Loss
        num = 2 * (self.mu * intersection_roi + (1 - self.mu) * intersection_non_roi) + self.smooth
        den = (self.mu * union_roi + (1 - self.mu) * union_non_roi) + self.smooth

        dice_loss = 1 - num / den  # (B, C)

        dice_loss = dice_loss.mean(dim=1)

        return dice_loss # (B,)


