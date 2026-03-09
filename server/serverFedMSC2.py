from server.serverbase import Server
from client.clientFedMSC2 import clientFedMSC2
from utils.feature import precompute_prototypes

class FedMSC2(Server):
    def __init__(self, args):
        super().__init__(args)

        self.seg_early_stop_counter = 0
        self.prev_best_iou = -float('inf')
    
        self.set_clients(clientFedMSC2)
        print("Finished creating server and clients.")

        if self.task != 2:
            # 获取特征层列表 ----------------------------------------------------------
            first_client = self.clients[0] # 获取一个样本
            batch_data = next(iter(first_client.load_train_data()))  # 获取批次数据（可能是元组或列表）
            images = batch_data[0].to(self.device)  # 提取第一个元素（图像张量）并移动设备
            _, feature_dict = first_client.model(images) 
            self.feature_layers = list(feature_dict.keys())  # 提取特征层的键

            # 初始化全局原型库和预计算缓存 ----------------------------------------------
            self.global_memory_bank = {i: {layer: {} for layer in self.feature_layers} for i in range(self.num_clients)}
            self.precomputed_prototypes = None  # 新增预计算结果缓存

    def train(self):
        # 初始化所有客户端模型
        last_best_metric = -float('inf')
        self.send_models()
        for i in range(self.global_rounds):
            print(f"\n------------- Round number: {i + 1} -------------")
            
            for client in self.clients:
                if self.stage == 'cls':
                    client.cls_train()  # 分类任务训练
                elif self.stage == 'seg':
                    client.seg_train()  # 分割任务训练
                else:
                    raise ValueError(f"Unknown stage: {self.stage}. Expected 'cls' or 'seg'.")

            if self.stage == 'cls':
                # 接收客户端上传的原型
                self.receive_protos()
                self.precomputed_prototypes = precompute_prototypes(
                    global_memory_bank=self.global_memory_bank,
                    layer_names=self.feature_layers,
                    device=self.device
                )
                # 接收客户端上传的模型
                self.receive_models()
                # 更新全局模型
                self.aggregate_parameters()
                # 发送 Global Memory Bank 到客户端
                self.send_protos()
                # 本地模型自适应更新
                for client in self.clients:
                    client.adaptive_local_update(self.global_model)
            elif self.stage == 'seg':
                # 仅接收模型并聚合参数
                self.receive_models()
                self.aggregate_parameters()

            if i % self.eval_gap == 0: 
                print("\nEvaluate personalized models")
                self.evaluate()

                
                current_metric = self.best_metric
                if current_metric > last_best_metric:
                    last_best_metric = current_metric
                    
                    if self.stage == 'cls':
                        # 保存所有客户端个性化模型
                        for client_id, model in self.best_models.items():
                            item_name = f"{self.args.algorithm}_task_{self.task}_client_{client_id}"
                            self.save_item(model, item_name)
                    elif self.stage == 'seg':
                        self.save_global_model()
                

            # 早停机制增强
            if self.auto_break:
                if self.stage == 'seg':
                    # 分割任务专用早停逻辑
                    if len(self.rs_test_iou) >= 1:
                        current_iou = self.rs_test_iou[-1]
                        if current_iou - self.prev_best_iou <= 0.0001:
                            self.seg_early_stop_counter += 1
                        else:
                            self.seg_early_stop_counter = 0
                            self.prev_best_iou = current_iou
                            
                        if self.seg_early_stop_counter >= 5:
                            print("Early stopping triggered for segmentation task")
                            break
                else:
                    # 原有早停逻辑
                    if self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                        break

        # 输出最佳结果
        print("\nBest accuracy.")
        if self.stage == 'cls':
            best_acc = max(self.rs_test_acc)
            print(f"Best Test Accuracy: {best_acc:.8f}")
        elif self.stage == 'seg':
            best_iou = max(self.rs_test_iou)  
            print(f"Best Test mIoU: {best_iou:.8f}")
        
        self.save_results()
        '''
        if self.stage == 'cls' and hasattr(self, 'best_models') and self.best_models:
            # 遍历所有客户端的最优模型并保存
            for client_id, model in self.best_models.items():
                item_name = f"{self.args.algorithm}_task_{self.task}_client_{client_id}"
                self.save_item(model, item_name)
        elif self.stage == 'seg':
            # 直接保存全局模型
            self.save_global_model()
        '''

    

    def send_protos(self):
        """
        将Global Memory Bank发送给每个客户端
        """
        assert len(self.clients) > 0

        for client in self.clients:
            client.precomputed_prototypes = self.precomputed_prototypes

    def receive_protos(self):
        """
        接收每个客户端上传的原型，并更新 global_memory_bank
        """
        assert len(self.clients) > 0
        for client in self.clients:
            # 直接使用客户端已经计算好的 prev_local_prototypes
            if client.prev_local_prototypes is not None:
                # 更新全局原型库中与当前客户端相关的部分
                for layer in client.prev_local_prototypes:
                    self.global_memory_bank[client.id][layer] = client.prev_local_prototypes[layer]
        
