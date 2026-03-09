import torch
import os
import numpy as np
import copy
from dataset.dataset import FLDataset
import h5py


class Server(object):
    def __init__(self, args):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients

        self.algorithm = args.algorithm
        self.stage = args.stage
        self.task = args.task

        self.goal = args.goal
        
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break
        self.clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_train_cls_loss = []

        self.rs_test_dice = []
        self.rs_test_iou = []
        self.rs_train_seg_loss = []

        self.eval_gap = args.eval_gap

        self.save_personalized = args.save_personalized

        
        self.best_metric = -float('inf')  # 跟踪最优指标
        self.best_models = {}  # 保存各客户端的最优模型


    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            train_data = FLDataset(client_dir=os.path.join(self.args.root_dir, f"client_{i}"),
                                   task_type=self.task,
                                   split="train")
            test_data = FLDataset(client_dir=os.path.join(self.args.root_dir, f"client_{i}"),
                                   task_type=self.task,
                                   split="test")
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data))
            self.clients.append(client)

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters(self.global_model)

    def receive_models(self):
        assert (len(self.clients) > 0)

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in self.clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = os.path.join('model', self.algorithm, f"task_{self.task}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model_path = os.path.join(model_path, f"{self.algorithm}_task_{self.task}.pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join('model', self.algorithm, f"task_{self.task}")
        model_path = os.path.join(model_path, f"{self.algorithm}_task_{self.task}.pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join('model', self.algorithm, f"task_{self.task}")
        model_path = os.path.join(model_path, f"{self.algorithm}_task_{self.task}.pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = self.algorithm
        result_path = "results"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if len(self.rs_test_acc):  # 检查是否有分类任务的测试集准确率数据
            algo = algo + "_" + self.goal
            if self.args.task is None:
                file_path = os.path.join(result_path, f"{algo}.h5")
            else:
                file_path = os.path.join(result_path, f"{algo}_{self.task}.h5")
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                # 保存分类任务的指标
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_train_cls_loss', data=self.rs_train_cls_loss)

        if len(self.rs_test_dice):  # 检查是否有分割任务的测试集 Dice 系数数据
            algo = algo + "_" + self.goal
            file_path = os.path.join(result_path, f"{algo}_seg.h5")
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                # 保存分割任务的指标
                hf.create_dataset('rs_test_dice', data=self.rs_test_dice)
                hf.create_dataset('rs_test_iou', data=self.rs_test_iou)
                hf.create_dataset('rs_train_seg_loss', data=self.rs_train_seg_loss)

    def save_item(self, item, item_name):
        model_path = os.path.join('model', self.algorithm, f"task_{self.task}")
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(item, os.path.join(model_path, item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, item_name + ".pt"))

    def test_metrics(self):

        num_samples = []
        metrics = []

        for client in self.clients:
            if self.stage == 'cls':
                test_num, correct = client.cls_test_metrics()
                metrics.append([correct])
            elif self.stage == 'seg':
                test_num, total_dice, total_iou = client.seg_test_metrics()
                metrics.append([total_dice, total_iou])
            else:
                raise ValueError(f"Unknown stage: {self.stage}. Expected 'cls' or 'seg'.")
            
            num_samples.append(test_num)

        ids = [c.id for c in self.clients]

        return ids, num_samples, metrics

    def train_metrics(self):
        
        num_samples = []
        losses = []
        for client in self.clients:
            if self.stage == 'cls':
                # 分类任务：获取训练集样本数和损失值
                total_loss, train_num = client.cls_train_metrics()
            elif self.stage == 'seg':
                # 分割任务：获取训练集样本数和损失值
                total_loss, train_num = client.seg_train_metrics()
            else:
                raise ValueError(f"Unknown stage: {self.stage}. Expected 'cls' or 'seg'.")
            
            num_samples.append(train_num)
            losses.append(total_loss)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        if self.stage == 'cls':
            # 分类任务评估
            ids, test_num_samples, test_metrics = self.test_metrics()  # 获取测试集指标
            _, train_num_samples, train_losses = self.train_metrics()  # 获取训练集指标

            # 计算全局测试集准确率
            total_correct = sum([metric[0] for metric in test_metrics]) * 1.0  # correct
            total_test_samples = sum(test_num_samples)  # test_num
            test_acc = total_correct / total_test_samples

            # 计算全局训练集损失值
            total_train_loss = sum(train_losses)  # total_loss
            total_train_samples = sum(train_num_samples)  # train_num
            train_loss = total_train_loss / total_train_samples

            # 记录测试集准确率和训练集损失值
            if acc is None:
                self.rs_test_acc.append(test_acc)
            else:
                acc.append(test_acc)

            if loss is None:
                self.rs_train_cls_loss.append(train_loss)
            else:
                loss.append(train_loss)
        
            # 打印结果
            print("Averaged Train Loss: {:.8f}".format(train_loss))
            print("Averaged Test Accuracy: {:.8f}".format(test_acc))

            if self.save_personalized:
                if test_acc > self.best_metric:
                    self.best_metric = test_acc
                    self.best_models = {client.id: copy.deepcopy(client.model) for client in self.clients}

        elif self.stage == 'seg':
            # 分割任务评估
            ids, test_num_samples, test_metrics = self.test_metrics()  # 获取测试集指标
            _, train_num_samples, train_losses = self.train_metrics()  # 获取训练集指标

            # 计算全局测试集 Dice 系数和 IoU
            total_dice = sum([metric[0] for metric in test_metrics])  # total_dice
            total_iou = sum([metric[1] for metric in test_metrics])  # total_iou
            total_test_samples = sum(test_num_samples)  
            test_dice = total_dice / total_test_samples  
            test_iou = total_iou / total_test_samples  

            # 计算全局训练集损失值
            total_train_loss = sum(train_losses)  # total_loss
            total_train_samples = sum(train_num_samples)  # train_num
            train_loss = total_train_loss / total_train_samples

            # 记录测试集 mIoU 系数和训练集损失值
            if acc is None:
                self.rs_test_iou.append(test_iou)
            else:
                acc.append(test_iou)

            if loss is None:
                self.rs_train_seg_loss.append(train_loss)
            else:
                loss.append(train_loss)

            # 打印结果
            print("Averaged Train Loss: {:.8f}".format(train_loss))
            print("Averaged Test Dice: {:.8f}".format(test_dice))
            print("Averaged Test IoU: {:.8f}".format(test_iou))

            
            if test_iou > self.best_metric:
                self.best_metric = test_iou
                self.best_models = {client.id: copy.deepcopy(client.model) for client in self.clients}

        else:
            raise ValueError(f"Unknown stage: {self.stage}. Expected 'cls' or 'seg'.")

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            # 根据任务类型来决定使用哪个指标进行检查
            if self.stage == 'cls':
                acc_ls = self.rs_test_acc  # 分类任务时使用 test accuracy
            elif self.stage == 'seg':
                acc_ls = self.rs_test_iou  # 分割任务时使用 test mIoU
            else:
                raise ValueError(f"Unknown stage: {self.stage}. Expected 'cls' or 'seg'.")

            # 如果设置了 top_cnt 和 div_value
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            # 如果只设置了 top_cnt
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            # 如果只设置了 div_value
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    
