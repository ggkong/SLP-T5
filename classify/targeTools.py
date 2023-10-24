import numpy as np
import torch
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = torch.logical_and(y_true[i], y_pred[i]).sum().item()
        q = torch.logical_or(y_true[i], y_pred[i]).sum().item()
        count += p / q
    return count / y_true.shape[0]


def OAA(y_true, y_pred):
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()
    return accuracy_score(y_true_np, y_pred_np)


def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if torch.sum(y_pred[i]) == 0:
            continue
        count += torch.sum(torch.logical_and(y_true[i], y_pred[i])) / torch.sum(y_pred[i])
    return count / y_true.shape[0]


def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if torch.sum(y_true[i]) == 0:
            continue
        count += torch.sum(torch.logical_and(y_true[i], y_pred[i])) / torch.sum(y_true[i])
    return count / y_true.shape[0]


def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (torch.sum(y_true[i]) == 0) and (torch.sum(y_pred[i]) == 0):
            continue
        p = torch.sum(torch.logical_and(y_true[i], y_pred[i]))
        q = torch.sum(y_true[i]) + torch.sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]


def countScore(y_true, y_pred):
    oaa = OAA(y_true, y_pred)
    acc = Accuracy(y_true, y_pred)
    p = Precision(y_true, y_pred)
    r = Recall(y_true, y_pred)
    f1 = F1Measure(y_true, y_pred)
    GM = (oaa + acc + p + r + f1) / 5
    return [GM, oaa, acc, p, r, f1]


def singleThresholdFive(epoch, model, dataloaderTest):
    for idxTest, dataTest in enumerate(dataloaderTest, 0):
        inputsTest, targetsFive = dataTest
        inputsTest = inputsTest.float()
        output = model(inputsTest)
    outputs = (output == output.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
    GMScore = countScore(targetsFive.int(), outputs)
    print(f"epoch:{epoch}, GM:{GMScore[0].item()}, OAA:{GMScore[1]}, ACC:{GMScore[2]}, F1:{GMScore[5].item()}")
    return GMScore


def testThresholdFive(epoch, model, dataloaderTest, class_num=4):
    model.eval()
    with torch.no_grad():
        for idxTest, dataTest in enumerate(dataloaderTest, 0):
            inputsTest, targetsFive = dataTest
            inputsTest = inputsTest.float()
            output = model(inputsTest)
        ThresholdList = selectMaxACC(targetsFive, output, class_num)
        outputsThreshold =[]
        for item in range(0, class_num):
            outputsItem = output[:, item:item+1]
            labels_item = torch.where(outputsItem > ThresholdList[item], torch.tensor(1), torch.tensor(0))
            outputsThreshold.append(labels_item)
        result = torch.cat(outputsThreshold, dim=1)
    # tempScore = {}
    #
    # for item in range(0, class_num):
    #     tempResult = result.clone()
    #     for i in range(tempResult.size(0)):
    #         if torch.all(tempResult[i] == 0):
    #             tempResult[i, item] = 1
    #     tempGMScore = countScore(targetsFive.int(), tempResult)
    #     tempScore[item] = tempGMScore
    # best = max(tempScore.items(), key=lambda x: x[1][0])
    # best = best[1]
    GMScore = countScore(targetsFive.int(), result)
    # print(f"epoch:{epoch}, bestThreshold:{ThresholdList}, GM:{best[0].item()}, OAA:{best[1]}, ACC:{best[2]}, F1:{best[5].item()}")
    # return best
    print(f"epoch:{epoch}, bestThreshold:{ThresholdList}, GM:{GMScore[0].item()}, OAA:{GMScore[1]}, ACC:{GMScore[2]}, F1:{GMScore[5].item()}")
    return GMScore


# 是不是可以改进成最佳匹配法
def selectMaxACC(target, outputs, class_num=4):
    bestThresholdItemList = []
    for item in range(0, class_num):
        targetItem = target[:, item:item+1]
        outputsItem = outputs[:, item:item+1]
        thresholdItemScore = {}

        for num in range(1, 99, 1):
            thresholdTest = round(num * 0.01, 2)
            labels_cov_test_item = torch.where(outputsItem > thresholdTest, torch.tensor(1), torch.tensor(0))
            scoreItem = accuracy_score(targetItem.int(), labels_cov_test_item) + recall_score(targetItem.int(), labels_cov_test_item) + precision_score(targetItem.int(), labels_cov_test_item) + f1_score(targetItem.int(), labels_cov_test_item)
            thresholdItemScore[thresholdTest] = scoreItem
        best = max(thresholdItemScore.items(), key=lambda x: x[1])
        bestThresholdItemList.append(best[0])
    return bestThresholdItemList


