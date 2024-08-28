import os
import json
import numpy as np
from sklearn.metrics import recall_score,precision_score
import matplotlib.pyplot as plt


def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = np.sum(np.logical_and(y_true[i], y_pred[i]))
        q = np.sum(np.logical_or(y_true[i], y_pred[i]))
        if q != 0:
            count += p / q
        else:
            count += 1
    return count / y_true.shape[0]


def category_accuracy_r(y_true, y_pred):
    num_classes = y_true.shape[1]
    accuracies = []

    for i in range(num_classes):
        acc = sum(t == p for t, p in zip(y_true[:, i], y_pred[:, i])) / len(y_true[:, i])
        accuracies.append(acc)

    return accuracies

def category_recall_r(y_true, y_pred):
    num_classes = y_true.shape[1]
    recalls = []

    for i in range(num_classes):
        true_positives = sum(t == 1 and p == 1 for t, p in zip(y_true[:, i], y_pred[:, i]))
        actual_positives = sum(t == 1 for t in y_true[:, i])
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = true_positives / actual_positives
        recalls.append(recall)

    return recalls

def category_precision_r(y_true, y_pred):
    num_classes = y_true.shape[1]
    precisions = []

    for i in range(num_classes):
        true_positives = sum(t == 1 and p == 1 for t, p in zip(y_true[:, i], y_pred[:, i]))
        predicted_positives = sum(p == 1 for p in y_pred[:, i])
        if predicted_positives == 0:
            precision = 0.0
        else:
            precision = true_positives / predicted_positives
        precisions.append(precision)

    return precisions



# def Accuracy(y_true, y_pred):
#     count = 0
#     for i in range(y_true.shape[0]):
#         p = sum(np.logical_and(y_true[i], y_pred[i]))
#         q = sum(np.logical_or(y_true[i], y_pred[i]))
#         count += p / q
#     return count / y_true.shape[0]

def Precision(y_true, y_pred):
    count = 0
    valid_samples = 0
    for i in range(y_true.shape[0]):
        pred_sum = np.sum(y_pred[i])
        if pred_sum == 0:
            continue  # 跳过没有任何预测的样本
        count += np.sum(np.logical_and(y_true[i], y_pred[i])) / pred_sum
        valid_samples += 1
    return count / valid_samples if valid_samples > 0 else 0

def MacroRecall(y_true, y_pred):
    count = 0
    valid_samples = 0
    for i in range(y_true.shape[0]):
        true_sum = np.sum(y_true[i])
        if true_sum == 0:
            continue
        count += np.sum(np.logical_and(y_true[i], y_pred[i])) / true_sum
        valid_samples += 1
    return count / valid_samples if valid_samples > 0 else 0

def MicroRecall(y_true, y_pred):
    true_positives = np.sum(np.logical_and(y_true, y_pred))
    all_positives = np.sum(y_true)
    return true_positives / all_positives if all_positives > 0 else 0

def su_recall_score(y_true, y_pred):
    if all(y == 0 for y in y_true) and all(y == 0 for y in y_pred):
        return 1.0
    else:
        return recall_score(y_true, y_pred, average='binary')

def bool2binary(bool_dict):
    return [1 if i else 0 for i in bool_dict.values()]

def load_annotations(json_path):
    with open(json_path,'r') as j:
            onehot = json.load(j)
    return onehot

def su_evaluate(json_path):
    images = [i for i in json_path.keys()]
    annotations = [j for j in json_path.values()]

def eval_from_json(json_file):
    cnt=0

    with open(json_file,'r') as j:

        data = json.load(j)
        all_pred,all_gt = [],[]
        all_acc = []
        for img, res_dict in data.items():
            cnt+=1

            pred = res_dict['pred']
            ground_truth = res_dict['ground_truth']

            acc_sample = Accuracy(np.array(ground_truth),np.array(pred))
            all_acc.append(acc_sample)

            all_pred.append(pred)
            all_gt.append(ground_truth)

        recall_macro = recall_score(np.array(all_gt),np.array(all_pred),average='macro')
        recall_micro = recall_score(np.array(all_gt),np.array(all_pred),average='micro')        
        precision_macro = precision_score(np.array(all_gt), np.array(all_pred), average='macro')
        precision_micro = precision_score(np.array(all_gt), np.array(all_pred), average='micro')
        acc = Accuracy(np.array(all_gt),np.array(all_pred))

        # print(f'Recall (macro): {recall_macro * 100:.2f}%')
        # print(f'Recall (micro): {recall_micro * 100:.2f}%')

        # print(f'Precision (macro): {precision_macro * 100:.2f}%')
        # print(f'Precision (micro): {precision_micro * 100:.2f}%')

        # print(f'Accuracy: {acc * 100:.2f}%')

        recalls = category_recall_r(np.array(all_gt),np.array(all_pred))
        formatted_recalls = [f'{recall*100:.2f}%' for recall in recalls]
        print(f'Category Recalls: {formatted_recalls}')
        recall_mean = np.mean(category_recall_r(np.array(all_gt),np.array(all_pred)))
        print(f'category_recall_mean: {recall_mean*100:.2f}%')

        precisions = category_precision_r(np.array(all_gt),np.array(all_pred))
        formatted_precisions = [f'{pre*100:.2f}%' for pre in precisions]
        print(f'Category precisions: {formatted_precisions}')
        precision_mean = np.mean(category_precision_r(np.array(all_gt),np.array(all_pred)))
        print(f'category_precision_mean: {precision_mean*100:.2f}%')

        accuracies = category_accuracy_r(np.array(all_gt),np.array(all_pred))
        formatted_accuracies = [f'{acc*100:.2f}%' for acc in accuracies]
        print(f'Category Accuracies: {formatted_accuracies}')
        accuracy_mean = np.mean(category_accuracy_r(np.array(all_gt),np.array(all_pred)))
        print(f'category_accuracy_mean: {accuracy_mean*100:.2f}%')

        recalls.append(recall_mean) 
        precisions.append(precision_mean)
        accuracies.append(accuracy_mean)

        return recalls, precisions, accuracies


def draw_figure(model1_recall,model1_precision,model1_accuracy,model2_recall,model2_precision,model2_accuracy,model3_recall,model3_precision,model3_accuracy):
    categories = ["smoking","waving hands and hailing","ped on lawn","crowded","fire or flood","fallen leaves/trash","illegal parking","global average"]
    model_names = ['LLaVA-v1.6', 'LLaVA-DINOv2-Vicuna-7b (ours)', 'LLaVA-unfreezed-DINOv2-Vicuna-7b (ours)']
    # 将数据放入一个字典中，便于处理

    print(len(model1_recall),len(model2_recall),len(model3_recall),len(model1_precision))

    data = {
        'recall': [model1_recall, model2_recall, model3_recall],
        'precision': [model1_precision, model2_precision, model3_precision],
        'accuracy': [model1_accuracy, model2_accuracy, model3_accuracy]
    }

    x = np.arange(len(model_names))

    fig, ax = plt.subplots(3, 1, figsize=(12, 18))

    for i, metric in enumerate(['recall', 'precision', 'accuracy']):
        for j, category in enumerate(categories):
            y = [data[metric][0][j], data[metric][1][j], data[metric][2][j]]
            if category == 'global average':
                ax[i].plot(model_names, y, marker='o', linestyle='dashdot', linewidth=3, markersize=10, label=f'{category}')
            else:
                ax[i].plot(model_names, y, marker='o', linestyle='--', label=f'{category}')
        
        ax[i].set_title(f'{metric.capitalize()} by Category and Model')
        ax[i].set_xlabel('Model')
        ax[i].set_ylabel(metric.capitalize())
        ax[i].legend()
        ax[i].grid(True)

    fig.tight_layout()
    plt.savefig("/root/TinyLLaVA_Factory/su_data/res.png")

if __name__=='__main__':

    print('category : 人员吸烟，人员招手/拦车，人员践踏草坪，人员聚集，火情和汛情，地面落叶/大块垃圾，违停车辆, 行人晕倒躺卧')
    # print(Accuracy(np.array([[1,1,1],[0,0,0]]),np.array([[1,1,1],[0,1,0]])))
    print("\n开源llava eval")
    recalls1, precisions1, accuracies1 = eval_from_json('/root/LLaVA_SU/su_data/outputs/llava_8cls.json')
    print("\nDINOv2-LLaVA eval")
    recalls2, precisions2, accuracies2 = eval_from_json('/root/TinyLLaVA_Factory/su_data/outputs/0611_1820-tiny-llava-vicuna-7b-v1.5-dinov2-giant-base-finetune.json')
    print("\n解冻训练DINOv2-LLaVA eval")
    recalls3, precisions3, accuracies3 = eval_from_json('/root/TinyLLaVA_Factory/su_data/outputs/0702_2002-llava_checkpoints_vit_finetune-prompt.json.json')
    
    print(recalls1,precisions1,accuracies1)
    # draw_figure(recalls1, precisions1, accuracies1, recalls2, precisions2, accuracies2, recalls3, precisions3, accuracies3)