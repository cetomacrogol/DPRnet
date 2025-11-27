import torch
import torch.nn as nn
import copy
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, confusion_matrix, precision_recall_curve, \
    precision_score
from sklearn.metrics import f1_score
from models import binary_cross_entropy, cross_entropy_logits, entropy_logits, RandomLayer, DRPNet  # 引入新的模型类
from prettytable import PrettyTable
from tqdm import tqdm
import configs
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

chemberta_model_path = './ChemBERTa-zinc-base-v1'


class Trainer(object):
    def __init__(self, model, optim, device, train_dataloader, val_dataloader, test_dataloader, experiment=None,
                 **config):
        self.model = model
        self.optim = optim
        self.device = device
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.custom_output_dim = config["CHEMBERTA"]["OUTPUT_DIM"]
        self.current_epoch = 0
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.best_model = None
        self.best_epoch = None
        self.best_auroc = 0
        self.smiles_extractor = None
        self.tokenizer = AutoTokenizer.from_pretrained(chemberta_model_path)
        self.train_loss_epoch = []
        self.val_loss_epoch, self.val_auroc_epoch = [], []
        self.test_metrics = {}
        self.config = config
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]

        valid_metric_header = ["# Epoch", "AUROC", "AUPRC", "Val_loss"]
        test_metric_header = ["# Best Epoch", "AUROC", "AUPRC", "F1", "Sensitivity", "Specificity", "Accuracy",
                              "Threshold", "Test_loss"]
        train_metric_header = ["# Epoch", "Train_loss"]
        self.val_table = PrettyTable(valid_metric_header)
        self.test_table = PrettyTable(test_metric_header)
        self.train_table = PrettyTable(train_metric_header)
        self.step = 0
        self.experiment = experiment

    def train(self):
        float2str = lambda x: '%0.4f' % x
        for i in range(self.epochs):
            self.current_epoch += 1
            train_loss = self.train_epoch()
            train_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [train_loss]))
            if self.experiment:
                self.experiment.log_metric("train_epoch model loss", train_loss, epoch=self.current_epoch)
            self.train_table.add_row(train_lst)
            self.train_loss_epoch.append(train_loss)
            auroc, auprc, val_loss = self.test(dataloader="val")
            if self.experiment:
                self.experiment.log_metric("valid_epoch model loss", val_loss, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auroc", auroc, epoch=self.current_epoch)
                self.experiment.log_metric("valid_epoch auprc", auprc, epoch=self.current_epoch)
            val_lst = ["epoch " + str(self.current_epoch)] + list(map(float2str, [auroc, auprc, val_loss]))
            self.val_table.add_row(val_lst)
            self.val_loss_epoch.append(val_loss)
            self.val_auroc_epoch.append(auroc)
            if auroc >= self.best_auroc:
                self.best_model = copy.deepcopy(self.model)
                self.best_auroc = auroc
                self.best_epoch = self.current_epoch
            print('Validation at Epoch ' + str(self.current_epoch) + ' with validation loss ' + str(val_loss), " AUROC "
                  + str(auroc) + " AUPRC " + str(auprc))
        auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, _, precision = self.test(dataloader="test")
        test_lst = ["epoch " + str(self.best_epoch)] + list(map(float2str, [auroc, auprc, f1, sensitivity, specificity,
                                                                            accuracy, 0.5, test_loss]))
        self.test_table.add_row(test_lst)
        print('Test at Best Model of Epoch ' + str(self.best_epoch) + ' with test loss ' + str(test_loss), " AUROC "
              + str(auroc) + " AUPRC " + str(auprc) + " Sensitivity " + str(sensitivity) + " Specificity " +
              str(specificity) + " Accuracy " + str(accuracy) + " Thred_optim " + "0.5")
        self.test_metrics["auroc"] = auroc
        self.test_metrics["auprc"] = auprc
        self.test_metrics["test_loss"] = test_loss
        self.test_metrics["sensitivity"] = sensitivity
        self.test_metrics["specificity"] = specificity
        self.test_metrics["accuracy"] = accuracy
        self.test_metrics["thred_optim"] = 0.5  # 固定阈值为0.5
        self.test_metrics["best_epoch"] = self.best_epoch
        self.test_metrics["F1"] = f1
        self.test_metrics["Precision"] = precision
        self.save_result()
        if self.experiment:
            self.experiment.log_metric("valid_best_auroc", self.best_auroc)
            self.experiment.log_metric("valid_best_epoch", self.best_epoch)
            self.experiment.log_metric("test_auroc", self.test_metrics["auroc"])
            self.experiment.log_metric("test_auprc", self.test_metrics["auprc"])
            self.experiment.log_metric("test_sensitivity", self.test_metrics["sensitivity"])
            self.experiment.log_metric("test_specificity", self.test_metrics["specificity"])
            self.experiment.log_metric("test_accuracy", self.test_metrics["accuracy"])
            self.experiment.log_metric("test_threshold", 0.5)
            self.experiment.log_metric("test_f1", self.test_metrics["F1"])
            self.experiment.log_metric("test_precision", self.test_metrics["Precision"])
        return self.test_metrics

    def save_result(self):
        if self.config["RESULT"]["SAVE_MODEL"]:
            torch.save(self.best_model.state_dict(),
                       os.path.join(self.output_dir, f"best_model_epoch_{self.best_epoch}.pth"))
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"model_epoch_{self.current_epoch}.pth"))
        state = {
            "train_epoch_loss": self.train_loss_epoch,
            "val_epoch_loss": self.val_loss_epoch,
            "test_metrics": self.test_metrics,
            "config": self.config
        }
        torch.save(state, os.path.join(self.output_dir, f"result_metrics.pt"))

        val_prettytable_file = os.path.join(self.output_dir, "valid_markdowntable.txt")
        test_prettytable_file = os.path.join(self.output_dir, "test_markdowntable.txt")
        train_prettytable_file = os.path.join(self.output_dir, "train_markdowntable.txt")
        with open(val_prettytable_file, 'w') as fp:
            fp.write(self.val_table.get_string())
        with open(test_prettytable_file, 'w') as fp:
            fp.write(self.test_table.get_string())
        with open(train_prettytable_file, "w") as fp:
            fp.write(self.train_table.get_string())

    def train_epoch(self):
        self.model.train()
        loss_epoch = 0
        num_batches = len(self.train_dataloader)
        for i, (compound_graph, compound_smiles, drug_graph, drug_smiles, labels, cell_expression_tensor) in enumerate(
                tqdm(self.train_dataloader)):
            self.step += 1
            compound_graph = compound_graph.to(self.device)
            drug_graph = drug_graph.to(self.device)
            labels = labels.float().to(self.device)

            # 化合物SMILES处理
            compound_smiles_list = list(compound_smiles)
            compound_tokenized_smiles = [
                self.tokenizer(smile, return_tensors='pt', padding=True, truncation=True, max_length=144) for smile in
                compound_smiles_list]

            compound_input_ids = pad_sequence([ts['input_ids'].squeeze(0) for ts in compound_tokenized_smiles],
                                              batch_first=True,
                                              padding_value=self.tokenizer.pad_token_id).to(self.device)
            compound_attention_masks = pad_sequence(
                [ts['attention_mask'].squeeze(0) for ts in compound_tokenized_smiles],
                batch_first=True, padding_value=0).to(self.device)

            compound_batched_input = {
                'input_ids': compound_input_ids,
                'attention_mask': compound_attention_masks
            }

            # 药物SMILES处理
            drug_smiles_list = list(drug_smiles)
            drug_tokenized_smiles = [
                self.tokenizer(smile, return_tensors='pt', padding=True, truncation=True, max_length=144) for smile in
                drug_smiles_list]

            drug_input_ids = pad_sequence([ts['input_ids'].squeeze(0) for ts in drug_tokenized_smiles],
                                          batch_first=True,
                                          padding_value=self.tokenizer.pad_token_id).to(self.device)
            drug_attention_masks = pad_sequence([ts['attention_mask'].squeeze(0) for ts in drug_tokenized_smiles],
                                                batch_first=True, padding_value=0).to(self.device)

            drug_batched_input = {
                'input_ids': drug_input_ids,
                'attention_mask': drug_attention_masks
            }

            cell_expression_tensor = cell_expression_tensor.to(self.device)

            self.optim.zero_grad()
            # 更新模型的输入参数
            score, _ = self.model(compound_graph, compound_batched_input, drug_graph, drug_batched_input,
                                  cell_expression_tensor)
            if self.config["DECODER"]["BINARY"] == 1:
                n, loss = binary_cross_entropy(score, labels)
            else:
                n, loss = cross_entropy_logits(score, labels)
            loss.backward()
            self.optim.step()
            loss_epoch += loss.item()
            if self.experiment:
                self.experiment.log_metric("train_step model loss", loss.item(), step=self.step)
        loss_epoch = loss_epoch / num_batches
        print('Training at Epoch ' + str(self.current_epoch) + ' with training loss ' + str(loss_epoch))
        return loss_epoch

    def test(self, dataloader="test"):
        test_loss = 0
        y_label, y_pred = [], []

        if dataloader == "test":
            data_loader = self.test_dataloader
        elif dataloader == "val":
            data_loader = self.val_dataloader
        else:
            raise ValueError(f"Error key value {dataloader}")

        num_batches = len(data_loader)
        with torch.no_grad():
            self.model.eval()
            for i, (
                    compound_graph, compound_smiles, drug_graph, drug_smiles, labels,
                    cell_expression_tensor) in enumerate(
                data_loader):
                compound_graph = compound_graph.to(self.device)
                drug_graph = drug_graph.to(self.device)
                labels = labels.float().to(self.device)

                # 化合物SMILES处理
                compound_smiles_list = list(compound_smiles)
                compound_tokenized_smiles = [
                    self.tokenizer(smile, return_tensors='pt', padding=True, truncation=True, max_length=144) for smile
                    in
                    compound_smiles_list]

                compound_input_ids = pad_sequence([ts['input_ids'].squeeze(0) for ts in compound_tokenized_smiles],
                                                  batch_first=True,
                                                  padding_value=self.tokenizer.pad_token_id).to(self.device)
                compound_attention_masks = pad_sequence(
                    [ts['attention_mask'].squeeze(0) for ts in compound_tokenized_smiles],
                    batch_first=True, padding_value=0).to(self.device)

                compound_batched_input = {
                    'input_ids': compound_input_ids,
                    'attention_mask': compound_attention_masks
                }

                # 药物SMILES处理
                drug_smiles_list = list(drug_smiles)
                drug_tokenized_smiles = [
                    self.tokenizer(smile, return_tensors='pt', padding=True, truncation=True, max_length=144) for smile
                    in
                    drug_smiles_list]

                drug_input_ids = pad_sequence([ts['input_ids'].squeeze(0) for ts in drug_tokenized_smiles],
                                              batch_first=True,
                                              padding_value=self.tokenizer.pad_token_id).to(self.device)
                drug_attention_masks = pad_sequence([ts['attention_mask'].squeeze(0) for ts in drug_tokenized_smiles],
                                                    batch_first=True, padding_value=0).to(self.device)

                drug_batched_input = {
                    'input_ids': drug_input_ids,
                    'attention_mask': drug_attention_masks
                }

                cell_expression_tensor = cell_expression_tensor.to(self.device)

                if dataloader == "val":
                    score, _ = self.model(compound_graph, compound_batched_input, drug_graph, drug_batched_input,
                                          cell_expression_tensor)
                elif dataloader == "test":
                    score, _ = self.best_model(compound_graph, compound_batched_input, drug_graph, drug_batched_input,
                                               cell_expression_tensor)

                if self.config["DECODER"]["BINARY"] == 1:
                    n, loss = binary_cross_entropy(score, labels)
                else:
                    n, loss = cross_entropy_logits(score, labels)

                test_loss += loss.item()
                y_label.extend(labels.cpu().tolist())
                y_pred.extend(n.cpu().tolist())

        # Calculate metrics
        test_loss /= num_batches
        auroc = roc_auc_score(y_label, y_pred)
        auprc = average_precision_score(y_label, y_pred)

        if dataloader == "test":
            # 使用固定阈值0.5生成二分类预测
            y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]

            # 计算所有指标
            cm = confusion_matrix(y_label, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            sensitivity = tp / (tp + fn + 1e-10)  # 召回率/敏感度
            specificity = tn / (tn + fp + 1e-10)

            # 直接调用sklearn的f1_score
            f1 = f1_score(y_label, y_pred_binary)
            precision = tp / (tp + fp + 1e-10)  # 精确率

            return auroc, auprc, f1, sensitivity, specificity, accuracy, test_loss, 0.5, precision
        else:
            return auroc, auprc, test_loss