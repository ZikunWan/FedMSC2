from client.clientbase import Client
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.feature import *
from utils.PCAU import PCAU
from utils.losses import contrastive_loss


class clientFedMSC2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super(clientFedMSC2, self).__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.num_clusters = args.num_clusters        # 每个类别的原型数K
        self.tau = args.tau                          # 对比学习温度系数
        self.lamda = args.lamda                      # 对比损失权重系数

        # 存储上一轮的本地原型 用于参数自适应更新
        self.prev_local_prototypes = None

        self.precomputed_prototypes = None # 结构为 {层名: (prototypes_tensor, proto_labels)}
        """
        e.g.
            prev_local_prototypes = {
                'layer1': {
                    0: np.array([[0.1, 0.2], [0.3, 0.4]]),  # 类别 0 的 2 个原型
                    1: np.array([[0.5, 0.6], [0.7, 0.8]]),  # 类别 1 的 2 个原型
                    2: np.array([[0.9, 1.0], [1.1, 1.2]])   # 类别 2 的 2 个原型
                },
                'layer2': {
                    0: np.array([[0.2, 0.3], [0.4, 0.5]]),  # 类别 0 的 2 个原型
                    1: np.array([[0.6, 0.7], [0.8, 0.9]]),  # 类别 1 的 2 个原型
                    2: np.array([[1.0, 1.1], [1.2, 1.3]])   # 类别 2 的 2 个原型
                }
            }

            global_memory_bank = {
                client_id_1: {
                    'layer1': {0: np.array([[...], [...]]), 1: np.array([[...], [...]])},
                    'layer2': {0: np.array([[...], [...]]), 1: np.array([[...], [...]])},
                },
                client_id_2: { ... },
                ...
            }
        """
    def adaptive_local_update(self,
                              global_model: torch.nn.Module):
        current_state = self.model.state_dict()
        updated_state = PCAU(
            global_model=global_model,
            local_model_state=current_state,
            precomputed_prototypes=self.precomputed_prototypes,
            device=self.device,
            prev_local_prototypes=self.prev_local_prototypes,
            num_classes=self.num_classes,
            num_clusters=self.num_clusters
        )
        self.model.load_state_dict(updated_state)

    def cls_train(self):
        train_loader = self.load_train_data()
        self.model.train()
        sup_loss_fn = nn.CrossEntropyLoss()
        accumulation_steps = 4
        for epoch in range(self.local_epochs):

            total_sup_loss = 0.0  # 累计监督损失
            total_con_loss = 0.0  # 累计对比损失
            total_batches = 0  # 累计批次数量

            with tqdm(train_loader, desc=f"[Client {self.id}] Epoch {epoch+1}/{self.local_epochs}", unit="batch") as pbar:
                self.optimizer.zero_grad()
                for batch_idx, (images, labels) in enumerate(pbar):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # 前向传播
                    outputs, feature_dict = self.model(images)
                
                    # 计算监督损失（交叉熵损失）
                    loss_sup = sup_loss_fn(outputs, labels)
                    
                    
            
                    # 计算对比损失
                    loss_con = contrastive_loss(feature_dict,
                                                labels,
                                                self.precomputed_prototypes,
                                                self.tau) if self.precomputed_prototypes else torch.tensor(0.0, device=self.device)
                    
                    

                    # 总损失
                    loss = (loss_sup + self.lamda * loss_con) / accumulation_steps  # 平均损失
                    loss.backward()
                    # 每 accumulation_steps 步更新一次参数
                    if (batch_idx + 1) % accumulation_steps == 0:
                        self.optimizer.step()  # 更新参数
                        self.optimizer.zero_grad()  # 清空梯度

                    total_sup_loss += loss_sup.item()
                    total_con_loss += loss_con.item()
                    total_batches += 1

                    pbar.set_postfix({
                        "Loss": f"{loss.item():.8f}",
                        "Avg Sup Loss": f"{(total_sup_loss / total_batches):.8f}",
                        "Avg Con Loss": f"{(total_con_loss / total_batches):.8f}"
                    })
        
            # 学习率衰减
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
    
        # 计算训练集的多尺度原型，并更新 prev_local_prototypes
        train_features, train_labels = extract_multi_scale_features(self.model, train_loader, self.device)
        self.prev_local_prototypes = generate_prototypes(train_features, train_labels, self.num_clusters)

    
    def cls_train_metrics(self):
        """
        在分类任务的本地训练完成后：
        计算训练集损失值，并统计样本数。

        返回:
            train_loss: 训练集的总损失值
            train_num: 训练集的样本数
        """
        train_loader = self.load_train_data()
        self.model.eval()  # 切换到评估模式

        # 定义监督损失函数（交叉熵损失）
        sup_loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0  # 累计总损失
        train_num = 0  # 累计样本数
        
        with torch.no_grad():  # 禁用梯度计算
            for images, labels in train_loader:
                # 将数据移动到设备
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs, feature_dict = self.model(images)

                # 计算监督损失（交叉熵损失）
                loss_sup = sup_loss_fn(outputs, labels)
                # 计算对比损失
                loss_con = contrastive_loss(feature_dict, labels, self.precomputed_prototypes, self.tau) if self.precomputed_prototypes else torch.tensor(0.0, device=self.device)

                # 总损失 = 监督损失 + 对比损失
                loss = loss_sup + self.lamda * loss_con

                # 累计损失和样本数
                total_loss += loss.item() * labels.shape[0]
                train_num += labels.shape[0]
  
        return total_loss, train_num

