import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler, ADASYN

import random
import numpy as np


# 多标签数据增强工具
def selectSameLabel(real_label, class_num=4):
    if class_num == 4:
        label_columns = ['envelope', 'lumen', 'stroma', 'thylakoid_membrane']
    else:
        label_columns = ['envelope', 'lumen', 'plastoglobule', 'stroma', 'thylakoid_membrane']

    # 创建一个字典，用于存储每个标签对应的样本
    label_samples = {}

    # 遍历 DataFrame 中的每一行
    for index, row in real_label.iterrows():
        # 获取该行的标签值，存储在一个元组中
        labels = tuple(row[label_columns].values)

        # 如果标签值不存在于 label_samples 字典中，则创建一个空列表作为值
        if labels not in label_samples:
            label_samples[labels] = []

        # 将当前行的索引添加到对应标签值的列表中
        label_samples[labels].append(index)

    # 打印每个标签对应的样本索引
    return label_samples


# 多标签数据增强工具
def KG_MLAug(real_data, real_label, multiple):
    label_sames = selectSameLabel(real_label)
    new_X = np.zeros((len(real_data) * multiple, real_data.shape[1]))
    new_y = np.zeros((len(real_data) * multiple, real_label.shape[1]))
    i = 0
    for labels, samples in label_sames.items():
        if len(samples) > 1:
            for item in samples:
                for _ in range(multiple):
                    # 随机选取邻居
                    reference = random.randint(0, len(samples) - 1)
                    index = samples[reference]
                    gap = real_data.loc[item] - real_data.loc[index]
                    ratio = random.random()
                    new_X[i] = np.array(real_data.loc[item] + ratio * gap)
                    new_y[i] = np.array(labels)
                    i = i + 1
        else:
            true_feature = real_data.loc[samples]
            for _ in range(multiple):
                ratio = random.uniform(0, 0.5)
                new_X[i] = np.array(true_feature + true_feature * ratio)
                new_y[i] = np.array(labels)
                i = i + 1
    new_y = new_y.astype(int)
    new_X = pd.DataFrame(new_X)
    new_X_column_names = [str(i) for i in range(real_data.shape[1])]
    new_X.columns = new_X_column_names
    new_y = pd.DataFrame(new_y)
    new_y.columns = ['envelope', 'lumen', 'stroma', 'thylakoid_membrane']
    return new_X, new_y


def MLDA(real_data, real_label, multiple):
    label_sames = selectSameLabel(real_label, class_num=5)
    new_X = np.zeros((len(real_data) * multiple, real_data.shape[1]))
    new_y = np.zeros((len(real_data) * multiple, real_label.shape[1]))
    i = 0
    for labels, samples in label_sames.items():
        if len(samples) > 1:
            for item in samples:
                for _ in range(multiple):
                    # 随机选取邻居
                    reference = random.randint(0, len(samples) - 1)
                    index = samples[reference]
                    gap = real_data.loc[item] - real_data.loc[index]
                    ratio = random.random()
                    new_X[i] = np.array(real_data.loc[item] + ratio * gap)
                    new_y[i] = np.array(labels)
                    i = i + 1
        else:
            true_feature = real_data.loc[samples]
            for _ in range(multiple):
                ratio = random.uniform(0, 0.5)
                new_X[i] = np.array(true_feature + true_feature * ratio)
                new_y[i] = np.array(labels)
                i = i + 1

    new_y = new_y.astype(int)
    new_X = pd.DataFrame(new_X)
    new_X_column_names = [str(i) for i in range(real_data.shape[1])]
    new_X.columns = new_X_column_names
    new_y = pd.DataFrame(new_y)
    new_y.columns = ['envelope', 'lumen', 'plastoglobule', 'stroma', 'thylakoid_membrane']
    return new_X, new_y


def dataAugForTestBorderlineSMOTE(features_pd, label_pd, multiple, feature_num):
    feature_with_label = pd.concat([features_pd, label_pd], axis=1)

    envelope_df = feature_with_label[feature_with_label['envelope'] == 1]
    not_envelope_df = feature_with_label[feature_with_label['envelope'] != 1]
    list_envelope = [0] * len(envelope_df) + [1] * len(not_envelope_df)
    envelope_P_N_df = pd.concat([envelope_df, not_envelope_df], axis=0)
    list_envelope = pd.DataFrame(list_envelope)
    sm = BorderlineSMOTE(sampling_strategy={0: len(envelope_df) * multiple + len(envelope_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(envelope_P_N_df.iloc[:, :feature_num], list_envelope)
    generate_envelope = X_res.iloc[len(features_pd):, :]

    lumen_df = feature_with_label[feature_with_label['lumen'] == 1]
    not_lumen_df = feature_with_label[feature_with_label['lumen'] != 1]
    list_lumen = [0] * len(lumen_df) + [1] * len(not_lumen_df)
    lumen_P_N_df = pd.concat([lumen_df, not_lumen_df], axis=0)
    list_lumen = pd.DataFrame(list_lumen)
    sm = BorderlineSMOTE(sampling_strategy={0: len(lumen_df) * multiple * 5 + len(lumen_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(lumen_P_N_df.iloc[:, :feature_num], list_lumen)
    generate_lumen = X_res.iloc[len(features_pd):, :]

    stroma_df = feature_with_label[feature_with_label['stroma'] == 1]
    not_stroma_df = feature_with_label[feature_with_label['stroma'] != 1]
    list_stroma = [0] * len(stroma_df) + [1] * len(not_stroma_df)
    stroma_P_N_df = pd.concat([stroma_df, not_stroma_df], axis=0)
    list_stroma = pd.DataFrame(list_stroma)
    sm = BorderlineSMOTE(sampling_strategy={0: len(stroma_df) * multiple * 2 + len(stroma_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(stroma_P_N_df.iloc[:, :feature_num], list_stroma)
    generate_stroma = X_res.iloc[len(features_pd):, :]

    thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] == 1]
    not_thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] != 1]
    list_thylakoid_membrane = [0] * len(thylakoid_membrane_df) + [1] * len(not_thylakoid_membrane_df)
    thylakoid_membrane_P_N_df = pd.concat([thylakoid_membrane_df, not_thylakoid_membrane_df], axis=0)
    list_thylakoid_membrane = pd.DataFrame(list_thylakoid_membrane)
    sm = BorderlineSMOTE(sampling_strategy={0: len(thylakoid_membrane_df) * multiple * 2 + len(thylakoid_membrane_df)},
               random_state=42)
    X_res, y_res = sm.fit_resample(thylakoid_membrane_P_N_df.iloc[:, :feature_num], list_thylakoid_membrane)
    generate_thylakoid_membrane = X_res.iloc[len(features_pd):, :]

    data_1_test_label = {'envelope': [1] * len(generate_envelope),
                         'lumen': [0] * len(generate_envelope),
                         'stroma': [0] * len(generate_envelope),
                         'thylakoid_membrane': [0] * len(generate_envelope)}
    df1 = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(generate_lumen),
                         'lumen': [1] * len(generate_lumen),
                         'stroma': [0] * len(generate_lumen),
                         'thylakoid_membrane': [0] * len(generate_lumen)}
    df2 = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(generate_stroma),
                         'lumen': [0] * len(generate_stroma),
                         'stroma': [1] * len(generate_stroma),
                         'thylakoid_membrane': [0] * len(generate_stroma)}
    df3 = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(generate_thylakoid_membrane),
                         'lumen': [0] * len(generate_thylakoid_membrane),
                         'stroma': [0] * len(generate_thylakoid_membrane),
                         'thylakoid_membrane': [1] * len(generate_thylakoid_membrane)}
    df4 = pd.DataFrame(data_4_test_label)

    all_G_feature = pd.concat([generate_envelope, generate_lumen, generate_stroma, generate_thylakoid_membrane], axis=0)
    all_G_label = pd.concat([df1, df2, df3, df4], axis=0)
    return all_G_feature, all_G_label


def dataAugForTestRandomOverSampler(features_pd, label_pd, multiple, feature_num):
    feature_with_label = pd.concat([features_pd, label_pd], axis=1)

    envelope_df = feature_with_label[feature_with_label['envelope'] == 1]
    not_envelope_df = feature_with_label[feature_with_label['envelope'] != 1]
    list_envelope = [0] * len(envelope_df) + [1] * len(not_envelope_df)
    envelope_P_N_df = pd.concat([envelope_df, not_envelope_df], axis=0)
    list_envelope = pd.DataFrame(list_envelope)
    sm = RandomOverSampler(sampling_strategy={0: len(envelope_df) * multiple + len(envelope_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(envelope_P_N_df.iloc[:, :feature_num], list_envelope)
    generate_envelope = X_res.iloc[len(features_pd):, :]

    lumen_df = feature_with_label[feature_with_label['lumen'] == 1]
    not_lumen_df = feature_with_label[feature_with_label['lumen'] != 1]
    list_lumen = [0] * len(lumen_df) + [1] * len(not_lumen_df)
    lumen_P_N_df = pd.concat([lumen_df, not_lumen_df], axis=0)
    list_lumen = pd.DataFrame(list_lumen)
    sm = RandomOverSampler(sampling_strategy={0: len(lumen_df) * multiple * 5 + len(lumen_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(lumen_P_N_df.iloc[:, :feature_num], list_lumen)
    generate_lumen = X_res.iloc[len(features_pd):, :]

    stroma_df = feature_with_label[feature_with_label['stroma'] == 1]
    not_stroma_df = feature_with_label[feature_with_label['stroma'] != 1]
    list_stroma = [0] * len(stroma_df) + [1] * len(not_stroma_df)
    stroma_P_N_df = pd.concat([stroma_df, not_stroma_df], axis=0)
    list_stroma = pd.DataFrame(list_stroma)
    sm = RandomOverSampler(sampling_strategy={0: len(stroma_df) * multiple * 2 + len(stroma_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(stroma_P_N_df.iloc[:, :feature_num], list_stroma)
    generate_stroma = X_res.iloc[len(features_pd):, :]

    thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] == 1]
    not_thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] != 1]
    list_thylakoid_membrane = [0] * len(thylakoid_membrane_df) + [1] * len(not_thylakoid_membrane_df)
    thylakoid_membrane_P_N_df = pd.concat([thylakoid_membrane_df, not_thylakoid_membrane_df], axis=0)
    list_thylakoid_membrane = pd.DataFrame(list_thylakoid_membrane)
    sm = RandomOverSampler(sampling_strategy={0: len(thylakoid_membrane_df) * multiple * 2 + len(thylakoid_membrane_df)},
               random_state=42)
    X_res, y_res = sm.fit_resample(thylakoid_membrane_P_N_df.iloc[:, :feature_num], list_thylakoid_membrane)
    generate_thylakoid_membrane = X_res.iloc[len(features_pd):, :]

    data_1_test_label = {'envelope': [1] * len(generate_envelope),
                         'lumen': [0] * len(generate_envelope),
                         'stroma': [0] * len(generate_envelope),
                         'thylakoid_membrane': [0] * len(generate_envelope)}
    df1 = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(generate_lumen),
                         'lumen': [1] * len(generate_lumen),
                         'stroma': [0] * len(generate_lumen),
                         'thylakoid_membrane': [0] * len(generate_lumen)}
    df2 = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(generate_stroma),
                         'lumen': [0] * len(generate_stroma),
                         'stroma': [1] * len(generate_stroma),
                         'thylakoid_membrane': [0] * len(generate_stroma)}
    df3 = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(generate_thylakoid_membrane),
                         'lumen': [0] * len(generate_thylakoid_membrane),
                         'stroma': [0] * len(generate_thylakoid_membrane),
                         'thylakoid_membrane': [1] * len(generate_thylakoid_membrane)}
    df4 = pd.DataFrame(data_4_test_label)

    all_G_feature = pd.concat([generate_envelope, generate_lumen, generate_stroma, generate_thylakoid_membrane], axis=0)
    all_G_label = pd.concat([df1, df2, df3, df4], axis=0)
    return all_G_feature, all_G_label


def dataAugForTestSMOTE(features_pd, label_pd, multiple, feature_num):
    # 添加加强多标签代码

    feature_with_label = pd.concat([features_pd, label_pd], axis=1)

    envelope_df = feature_with_label[feature_with_label['envelope'] == 1]
    not_envelope_df = feature_with_label[feature_with_label['envelope'] != 1]
    list_envelope = [0] * len(envelope_df) + [1] * len(not_envelope_df)
    envelope_P_N_df = pd.concat([envelope_df, not_envelope_df], axis=0)
    list_envelope = pd.DataFrame(list_envelope)
    sm = SMOTE(sampling_strategy={0: len(envelope_df) * multiple+len(envelope_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(envelope_P_N_df.iloc[:, :feature_num], list_envelope)
    generate_envelope = X_res.iloc[len(features_pd):, :]

    lumen_df = feature_with_label[feature_with_label['lumen'] == 1]
    not_lumen_df = feature_with_label[feature_with_label['lumen'] != 1]
    list_lumen = [0] * len(lumen_df) + [1] * len(not_lumen_df)
    lumen_P_N_df = pd.concat([lumen_df, not_lumen_df], axis=0)
    list_lumen = pd.DataFrame(list_lumen)
    sm = SMOTE(sampling_strategy={0: len(lumen_df) * multiple +len(lumen_df)}, random_state=42)
    # sm = SMOTE(sampling_strategy={0: len(lumen_df) * multiple * 5 + len(lumen_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(lumen_P_N_df.iloc[:, :feature_num], list_lumen)
    generate_lumen = X_res.iloc[len(features_pd):, :]

    stroma_df = feature_with_label[feature_with_label['stroma'] == 1]
    not_stroma_df = feature_with_label[feature_with_label['stroma'] != 1]
    list_stroma = [0] * len(stroma_df) + [1] * len(not_stroma_df)
    stroma_P_N_df = pd.concat([stroma_df, not_stroma_df], axis=0)
    list_stroma = pd.DataFrame(list_stroma)
    sm = SMOTE(sampling_strategy={0: len(stroma_df) * multiple * 2 + len(stroma_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(stroma_P_N_df.iloc[:, :feature_num], list_stroma)
    generate_stroma = X_res.iloc[len(features_pd):, :]

    thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] == 1]
    not_thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] != 1]
    list_thylakoid_membrane = [0] * len(thylakoid_membrane_df) + [1] * len(not_thylakoid_membrane_df)
    thylakoid_membrane_P_N_df = pd.concat([thylakoid_membrane_df, not_thylakoid_membrane_df], axis=0)
    list_thylakoid_membrane = pd.DataFrame(list_thylakoid_membrane)
    # sm = SMOTE(sampling_strategy={0: len(thylakoid_membrane_df) * multiple * 2 + len(thylakoid_membrane_df)},random_state=42)
    sm = SMOTE(sampling_strategy={0: len(thylakoid_membrane_df) * multiple  + len(thylakoid_membrane_df)},random_state=42)
    X_res, y_res = sm.fit_resample(thylakoid_membrane_P_N_df.iloc[:, :feature_num], list_thylakoid_membrane)
    generate_thylakoid_membrane = X_res.iloc[len(features_pd):, :]

    data_1_test_label = {'envelope': [1] * len(generate_envelope),
               'lumen': [0] * len(generate_envelope),
               'stroma': [0] * len(generate_envelope),
               'thylakoid_membrane': [0] * len(generate_envelope)}
    df1 = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(generate_lumen),
               'lumen': [1] * len(generate_lumen),
               'stroma': [0] * len(generate_lumen),
               'thylakoid_membrane': [0] * len(generate_lumen)}
    df2 = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(generate_stroma),
               'lumen': [0] * len(generate_stroma),
               'stroma': [1] * len(generate_stroma),
               'thylakoid_membrane': [0] * len(generate_stroma)}
    df3 = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(generate_thylakoid_membrane),
               'lumen': [0] * len(generate_thylakoid_membrane),
               'stroma': [0] * len(generate_thylakoid_membrane),
               'thylakoid_membrane': [1] * len(generate_thylakoid_membrane)}
    df4 = pd.DataFrame(data_4_test_label)

    all_G_feature = pd.concat([generate_envelope, generate_lumen, generate_stroma, generate_thylakoid_membrane], axis=0)
    all_G_label = pd.concat([df1, df2, df3, df4], axis=0)
    return all_G_feature, all_G_label


def dataAugForTestADASYN(features_pd, label_pd, multiple, feature_num):

    feature_with_label = pd.concat([features_pd, label_pd], axis=1)

    envelope_df = feature_with_label[feature_with_label['envelope'] == 1]
    not_envelope_df = feature_with_label[feature_with_label['envelope'] != 1]
    list_envelope = [0] * len(envelope_df) + [1] * len(not_envelope_df)
    envelope_P_N_df = pd.concat([envelope_df, not_envelope_df], axis=0)
    list_envelope = pd.DataFrame(list_envelope)
    sm = ADASYN(sampling_strategy={0: len(envelope_df) * multiple+len(envelope_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(envelope_P_N_df.iloc[:, :feature_num], list_envelope)
    generate_envelope = X_res.iloc[len(features_pd):, :]

    lumen_df = feature_with_label[feature_with_label['lumen'] == 1]
    not_lumen_df = feature_with_label[feature_with_label['lumen'] != 1]
    list_lumen = [0] * len(lumen_df) + [1] * len(not_lumen_df)
    lumen_P_N_df = pd.concat([lumen_df, not_lumen_df], axis=0)
    list_lumen = pd.DataFrame(list_lumen)
    sm = ADASYN(sampling_strategy={0: len(lumen_df) * multiple * 5+len(lumen_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(lumen_P_N_df.iloc[:, :feature_num], list_lumen)
    generate_lumen = X_res.iloc[len(features_pd):, :]

    stroma_df = feature_with_label[feature_with_label['stroma'] == 1]
    not_stroma_df = feature_with_label[feature_with_label['stroma'] != 1]
    list_stroma = [0] * len(stroma_df) + [1] * len(not_stroma_df)
    stroma_P_N_df = pd.concat([stroma_df, not_stroma_df], axis=0)
    list_stroma = pd.DataFrame(list_stroma)
    sm = ADASYN(sampling_strategy={0: len(stroma_df) * multiple * 2 + len(stroma_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(stroma_P_N_df.iloc[:, :feature_num], list_stroma)
    generate_stroma = X_res.iloc[len(features_pd):, :]

    thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] == 1]
    not_thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] != 1]
    list_thylakoid_membrane = [0] * len(thylakoid_membrane_df) + [1] * len(not_thylakoid_membrane_df)
    thylakoid_membrane_P_N_df = pd.concat([thylakoid_membrane_df, not_thylakoid_membrane_df], axis=0)
    list_thylakoid_membrane = pd.DataFrame(list_thylakoid_membrane)
    sm = ADASYN(sampling_strategy={0: len(thylakoid_membrane_df) * multiple * 2 + len(thylakoid_membrane_df)},random_state=42)
    X_res, y_res = sm.fit_resample(thylakoid_membrane_P_N_df.iloc[:, :feature_num], list_thylakoid_membrane)
    generate_thylakoid_membrane = X_res.iloc[len(features_pd):, :]

    data_1_test_label = {'envelope': [1] * len(generate_envelope),
               'lumen': [0] * len(generate_envelope),
               'stroma': [0] * len(generate_envelope),
               'thylakoid_membrane': [0] * len(generate_envelope)}
    df1 = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(generate_lumen),
               'lumen': [1] * len(generate_lumen),
               'stroma': [0] * len(generate_lumen),
               'thylakoid_membrane': [0] * len(generate_lumen)}
    df2 = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(generate_stroma),
               'lumen': [0] * len(generate_stroma),
               'stroma': [1] * len(generate_stroma),
               'thylakoid_membrane': [0] * len(generate_stroma)}
    df3 = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(generate_thylakoid_membrane),
               'lumen': [0] * len(generate_thylakoid_membrane),
               'stroma': [0] * len(generate_thylakoid_membrane),
               'thylakoid_membrane': [1] * len(generate_thylakoid_membrane)}
    df4 = pd.DataFrame(data_4_test_label)

    all_G_feature = pd.concat([generate_envelope, generate_lumen, generate_stroma, generate_thylakoid_membrane], axis=0)
    all_G_label = pd.concat([df1, df2, df3, df4], axis=0)
    return all_G_feature, all_G_label


def test_aug(feature_label, multiple):
    feature = feature_label.iloc[:, :1424]
    label = feature_label.iloc[:, -4:]

    multi_label_samples = label[(label.sum(axis=1) >= 2)]
    multi_label_indices = multi_label_samples.index
    multi_features_samples = feature.loc[multi_label_indices]

    true_feature = feature.iloc[:15, :]
    true_label = label.iloc[:15, :]
    true_feature = true_feature.drop(12)
    true_label = true_label.drop(12)

    multi_features_samples = pd.concat([multi_features_samples, true_feature], axis=0)
    multi_label_samples = pd.concat([multi_label_samples, true_label], axis=0)

    feature = multi_features_samples
    label = multi_label_samples
    label_sames = selectSameLabel(label)

    new_X = np.zeros((len(feature) * multiple, feature.shape[1]))
    new_y = np.zeros((len(feature) * multiple, label.shape[1]))
    i = 0

    # 筛选多标签
    for labels, samples in label_sames.items():
        for x in samples:
            true_feature = feature.loc[x]
            for _ in range(multiple):
                ratio = random.uniform(0, 0.5)
                new_X[i] = np.array(true_feature + true_feature * ratio)
                new_y[i] = np.array(labels)
                i = i + 1
    new_y = new_y.astype(int)
    new_X = pd.DataFrame(new_X)
    new_X_column_names = [str(i) for i in range(feature.shape[1])]
    new_X.columns = new_X_column_names
    new_y = pd.DataFrame(new_y)
    new_y.columns = ['envelope', 'lumen', 'stroma', 'thylakoid_membrane']
    return new_X, new_y