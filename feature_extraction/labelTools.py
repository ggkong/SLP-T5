import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from feature_extraction.AAC import AAC
from feature_extraction.DPC import DPC
from feature_extraction.FileTools import readfasta
from feature_extraction.ProtT5 import readH5File


def transform_labels(label):
    labels = label.split(',')
    labels = [l.strip() for l in labels]
    return labels


def labelList2Mutil(label_list):
    LabelList = [item.split("|")[1] for item in label_list]
    series = pd.Series(LabelList, name='labels')
    df_label = pd.DataFrame(series)

    trainLabelList = []
    for index, row in df_label.iterrows():
        value = row['labels']
        trainLabelList.append(transform_labels(value))

    mlb = MultiLabelBinarizer()
    encoded_labels = mlb.fit_transform(trainLabelList)
    mutil_labels = pd.DataFrame(encoded_labels, columns=mlb.classes_)

    return mutil_labels


def getACCT5DF(h5_path, fasta_path):
    T5_122, label_122 = readH5File(h5_path)
    label_122 = [s.split()[0] for s in label_122]
    index_list, seq_list = readfasta(fasta_path)
    index_list = [s.split()[0] for s in index_list]
    test_122_dict  = {index_list[i]: seq_list[i] for i in range(len(index_list))}
    seq_sorted_by_122 = []
    for item in label_122:
        seq_sorted_by_122.append(test_122_dict[item])
    ACC_122 = np.array(AAC(seq_sorted_by_122))
    AAC_T5_envelope_122 = pd.DataFrame(np.hstack((ACC_122, T5_122)))
    return AAC_T5_envelope_122


def getDPCT5DF(h5_path, fasta_path):
    T5_122, label_122 = readH5File(h5_path)
    label_122 = [s.split()[0] for s in label_122]
    index_list, seq_list = readfasta(fasta_path)
    index_list = [s.split()[0] for s in index_list]
    test_122_dict = {index_list[i]: seq_list[i] for i in range(len(index_list))}
    seq_sorted_by_122 = []
    for item in label_122:
        seq_sorted_by_122.append(test_122_dict[item])
    DPC_122 = np.array(DPC(seq_sorted_by_122))
    DPC_T5_envelope_122 = pd.DataFrame(np.hstack((DPC_122, T5_122)))
    return DPC_T5_envelope_122


def getAACDF(h5_path, fasta_path):
    T5_122, label_122 = readH5File(h5_path)
    label_122 = [s.split()[0] for s in label_122]
    index_list, seq_list = readfasta(fasta_path)
    index_list = [s.split()[0] for s in index_list]
    test_122_dict = {index_list[i]: seq_list[i] for i in range(len(index_list))}
    seq_sorted_by_122 = []
    for item in label_122:
        seq_sorted_by_122.append(test_122_dict[item])
    AAC_122 = np.array(AAC(seq_sorted_by_122))
    AAC_122_DF = pd.DataFrame(AAC_122)
    return AAC_122_DF


def getACCDPCT5DF(h5_path, fasta_path):
    T5_122, label_122 = readH5File(h5_path)
    label_122 = [s.split()[0] for s in label_122]
    index_list, seq_list = readfasta(fasta_path)
    index_list = [s.split()[0] for s in index_list]
    test_122_dict = {index_list[i]: seq_list[i] for i in range(len(index_list))}
    seq_sorted_by_122 = []
    for item in label_122:
        seq_sorted_by_122.append(test_122_dict[item])
    AAC_122 = np.array(AAC(seq_sorted_by_122))
    DPC_122 = np.array(DPC(seq_sorted_by_122))
    AAC_DPC_T5_envelope_122 = pd.DataFrame(np.hstack((AAC_122, DPC_122, T5_122)))
    return AAC_DPC_T5_envelope_122


def getT5DF(h5_path):
    T5_122, _ = readH5File(h5_path)
    T5_envelope_122 = pd.DataFrame(T5_122)
    return T5_envelope_122


def getTestLabel(feature_envelope_test, feature_lumen_test, feature_stroma_test, feature_membrane_test):
    data_1_test_label = {'envelope': [1] * len(feature_envelope_test),
                         'lumen': [0] * len(feature_envelope_test),
                         'stroma': [0] * len(feature_envelope_test),
                         'thylakoid_membrane': [0] * len(feature_envelope_test)}
    df1_test = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(feature_lumen_test),
                         'lumen': [1] * len(feature_lumen_test),
                         'stroma': [0] * len(feature_lumen_test),
                         'thylakoid_membrane': [0] * len(feature_lumen_test)}
    df2_test = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(feature_stroma_test),
                         'lumen': [0] * len(feature_stroma_test),
                         'stroma': [1] * len(feature_stroma_test),
                         'thylakoid_membrane': [0] * len(feature_stroma_test)}
    df3_test = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(feature_membrane_test),
                         'lumen': [0] * len(feature_membrane_test),
                         'stroma': [0] * len(feature_membrane_test),
                         'thylakoid_membrane': [1] * len(feature_membrane_test)}
    df4_test = pd.DataFrame(data_4_test_label)

    all_test_label = pd.concat([df1_test, df2_test, df3_test, df4_test], axis=0)
    all_test_feature = pd.concat(
        [pd.DataFrame(feature_envelope_test), pd.DataFrame(feature_lumen_test), pd.DataFrame(feature_stroma_test),
         pd.DataFrame(feature_membrane_test)], axis=0)
    test_df = pd.concat([all_test_feature, all_test_label], axis=1)
    sorted_df = test_df.sort_values(by=test_df.columns[:1044].tolist())
    return sorted_df


def getTestLabelByFeatureNum(feature_envelope_test, feature_lumen_test, feature_stroma_test, feature_membrane_test, feature_num):
    data_1_test_label = {'envelope': [1] * len(feature_envelope_test),
                         'lumen': [0] * len(feature_envelope_test),
                         'stroma': [0] * len(feature_envelope_test),
                         'thylakoid_membrane': [0] * len(feature_envelope_test)}
    df1_test = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(feature_lumen_test),
                         'lumen': [1] * len(feature_lumen_test),
                         'stroma': [0] * len(feature_lumen_test),
                         'thylakoid_membrane': [0] * len(feature_lumen_test)}
    df2_test = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(feature_stroma_test),
                         'lumen': [0] * len(feature_stroma_test),
                         'stroma': [1] * len(feature_stroma_test),
                         'thylakoid_membrane': [0] * len(feature_stroma_test)}
    df3_test = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(feature_membrane_test),
                         'lumen': [0] * len(feature_membrane_test),
                         'stroma': [0] * len(feature_membrane_test),
                         'thylakoid_membrane': [1] * len(feature_membrane_test)}
    df4_test = pd.DataFrame(data_4_test_label)

    all_test_label = pd.concat([df1_test, df2_test, df3_test, df4_test], axis=0)
    all_test_feature = pd.concat(
        [pd.DataFrame(feature_envelope_test), pd.DataFrame(feature_lumen_test), pd.DataFrame(feature_stroma_test),
         pd.DataFrame(feature_membrane_test)], axis=0)
    test_df = pd.concat([all_test_feature, all_test_label], axis=1)
    sorted_df = test_df.sort_values(by=test_df.columns[:feature_num].tolist())
    return sorted_df