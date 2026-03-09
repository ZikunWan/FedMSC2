import torch
import os
from utils.losses import SegLoss
from utils.metrics import iou_score, dice_coefficient
from dataset.dataset import FLDataset
from torch.utils.data import DataLoader
import copy
from tqdm import tqdm

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.args = args
        self.model = copy.deepcopy(args.model)  # Passed in model
        self.id = id
        self.device = args.device
        self.save_folder_name = args.save_folder_name
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.learning_rate_decay = args.learning_rate_decay  # 默认False
        self.task = args.task
        
        
        self.loss_mu = args.loss_mu
        # Loss function and optimizer
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Learning rate scheduler
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)

        # 客户端的数据样本数
        self.train_samples = train_samples
        self.test_samples = test_samples

    def load_train_data(self, batch_size=None):
        """Load training data for the client."""
        if batch_size is None:
            batch_size = self.batch_size

        train_dataset = FLDataset(
            client_dir=os.path.join(self.args.root_dir, f"client_{self.id}"),
            task_type=self.args.task,
            split='train'
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )

        return train_loader

    def load_test_data(self, batch_size=None):
        """Load test data for the client."""
        if batch_size is None:
            batch_size = self.batch_size

        test_dataset = FLDataset(
            client_dir=os.path.join(self.args.root_dir, f"client_{self.id}"),
            task_type=self.args.task,
            split='test'
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

        return test_loader

    def set_parameters(self, model):
        """Set model parameters from another model."""
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def update_parameters(self, new_params):
        """Update model parameters with new values."""
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def cls_test_metrics(self):
        """Evaluate the model on the test data."""
        test_loader = self.load_test_data()
        self.model.eval()

        test_num = 0  # 累计样本数
        correct = 0  # 累计正确预测的样本数

        with torch.no_grad():  # 禁用梯度计算
            for images, labels in test_loader:
                # 将数据移动到设备
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs, _ = self.model(images)  # 分类输出
                test_num += labels.size(0)

                # 计算正确预测的样本数
                _, predicted = torch.max(outputs, 1)  # 获取预测类别
                correct += (predicted == labels).sum().item()

        return test_num, correct

    def seg_test_metrics(self):
        test_loader = self.load_test_data()
        self.model.eval()

        test_num = 0  # 累计样本数
        total_dice = 0.0  # 累计 Dice 系数
        total_iou = 0.0  # 累计 IoU

        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                # 前向传播
                logits = self.model(images)

                # 计算 Dice 系数
                dice_score = dice_coefficient(logits, masks)
                total_dice += dice_score.item() * masks.size(0)

                # 计算 IoU
                iou = iou_score(logits, masks)
                total_iou += iou.item() * masks.size(0)

                # 累计样本数
                test_num += masks.size(0)

        return test_num, total_dice, total_iou

    def cls_train_metrics(self):
        """Evaluate the model on the training data."""
        trainloader = self.load_train_data()
        self.model.eval()

        total_loss = 0.0  # 累计总损失
        train_num = 0  # 累计样本数

        with torch.no_grad():
            for images, labels in trainloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                outputs, _ = self.model(images)
                loss = self.loss(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                train_num += labels.size(0)

        return total_loss, train_num

    def seg_train_metrics(self):
        train_loader = self.load_train_data()
        self.model.eval()  # 切换到评估模式
        criterion = SegLoss(mu=self.loss_mu)

        total_loss = 0.0  # 累计总损失
        train_num = 0  # 累计样本数

        with torch.no_grad():  # 禁用梯度计算
            for images, masks, rois in train_loader:
                # 将数据移动到设备
                images = images.to(self.device)
                masks = masks.to(self.device)
                rois = rois.to(self.device)

                # 前向传播
                logits = self.model(images)

                # 计算监督损失（加权 Dice 损失）
                loss_sup = criterion(logits, masks, rois)

                # 累计损失和样本数
                total_loss += loss_sup.item() * masks.size(0)
                train_num += masks.size(0)

        return total_loss, train_num
    
    def save_item(self, item, item_name):
        """Save model or other items."""
        if self.save_folder_name and not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, f"client_{self.id}_{item_name}.pt"))

    def load_item(self, item_name):
        """Load saved items."""
        return torch.load(os.path.join(self.save_folder_name, f"client_{self.id}_{item_name}.pt"))

    def seg_train(self):
        train_loader = self.load_train_data()
        self.model.train()
        criterion = SegLoss(mu=self.loss_mu)

        for epoch in range(self.local_epochs):
            total_sup_loss = 0.0  # 累计监督损失
            total_batches = 0  # 累计批次数量

            with tqdm(train_loader, desc=f"[Client {self.id}] Epoch {epoch+1}/{self.local_epochs}", unit="batch") as pbar:
                for _, (images, masks, rois) in enumerate(pbar):
                    # 将数据移动到设备
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    rois = rois.to(self.device)

                    # 前向传播
                    logits = self.model(images)

                    # 计算监督损失（Dice损失）
                    loss_sup = criterion(logits, masks, rois)

                    # 反向传播与优化
                    self.optimizer.zero_grad()
                    loss_sup.backward()
                    self.optimizer.step()

                    # 累加损失
                    total_sup_loss += loss_sup.item()
                    total_batches += 1

                    # 更新进度条信息
                    pbar.set_postfix({
                        "Sup Loss": f"{loss_sup.item():.8f}",
                        "Avg Sup Loss": f"{(total_sup_loss / total_batches):.8f}"
                    })

            # 学习率衰减
            if self.learning_rate_decay:
                self.learning_rate_scheduler.step()
