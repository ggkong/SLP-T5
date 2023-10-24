import pandas as pd
from imblearn.over_sampling import SMOTE

# 数据增强方法，改！


def dataAugSMOTE(features_pd, label_pd, multiple, feature_num):
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

    plastoglobule_df = feature_with_label[feature_with_label['plastoglobule'] == 1]
    not_plastoglobule_df = feature_with_label[feature_with_label['plastoglobule'] != 1]
    list_plastoglobule = [0] * len(plastoglobule_df) + [1] * len(not_plastoglobule_df)
    plastoglobule_P_N_df = pd.concat([plastoglobule_df, not_plastoglobule_df], axis=0)
    list_plastoglobule = pd.DataFrame(list_plastoglobule)
    sm = SMOTE(sampling_strategy={0: len(plastoglobule_df) * multiple + len(plastoglobule_df)}, random_state=42)
    # sm = SMOTE(sampling_strategy={0: len(lumen_df) * multiple * 5 + len(lumen_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(plastoglobule_P_N_df.iloc[:, :feature_num], list_plastoglobule)
    generate_plastoglobule = X_res.iloc[len(features_pd):, :]


    stroma_df = feature_with_label[feature_with_label['stroma'] == 1]
    not_stroma_df = feature_with_label[feature_with_label['stroma'] != 1]
    list_stroma = [0] * len(stroma_df) + [1] * len(not_stroma_df)
    stroma_P_N_df = pd.concat([stroma_df, not_stroma_df], axis=0)
    list_stroma = pd.DataFrame(list_stroma)
    sm = SMOTE(sampling_strategy={0: len(stroma_df) * multiple + len(stroma_df)}, random_state=42)
    X_res, y_res = sm.fit_resample(stroma_P_N_df.iloc[:, :feature_num], list_stroma)
    generate_stroma = X_res.iloc[len(features_pd):, :]

    thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] == 1]
    not_thylakoid_membrane_df = feature_with_label[feature_with_label['thylakoid_membrane'] != 1]
    list_thylakoid_membrane = [0] * len(thylakoid_membrane_df) + [1] * len(not_thylakoid_membrane_df)
    thylakoid_membrane_P_N_df = pd.concat([thylakoid_membrane_df, not_thylakoid_membrane_df], axis=0)
    list_thylakoid_membrane = pd.DataFrame(list_thylakoid_membrane)
    # sm = SMOTE(sampling_strategy={0: len(thylakoid_membrane_df) * multiple * 2 + len(thylakoid_membrane_df)},random_state=42)
    sm = SMOTE(sampling_strategy={0: len(thylakoid_membrane_df) * multiple + len(thylakoid_membrane_df)},random_state=42)
    X_res, y_res = sm.fit_resample(thylakoid_membrane_P_N_df.iloc[:, :feature_num], list_thylakoid_membrane)
    generate_thylakoid_membrane = X_res.iloc[len(features_pd):, :]

    data_1_test_label = {'envelope': [1] * len(generate_envelope),
               'lumen': [0] * len(generate_envelope),
                'plastoglobule': [0] * len(generate_envelope),
               'stroma': [0] * len(generate_envelope),
               'thylakoid_membrane': [0] * len(generate_envelope)}
    df1 = pd.DataFrame(data_1_test_label)

    data_2_test_label = {'envelope': [0] * len(generate_lumen),
               'lumen': [1] * len(generate_lumen),
                'plastoglobule': [0] * len(generate_lumen),
               'stroma': [0] * len(generate_lumen),
               'thylakoid_membrane': [0] * len(generate_lumen)}
    df2 = pd.DataFrame(data_2_test_label)

    data_3_test_label = {'envelope': [0] * len(generate_plastoglobule),
                         'lumen': [0] * len(generate_plastoglobule),
                         'plastoglobule': [1] * len(generate_plastoglobule),
                         'stroma': [0] * len(generate_plastoglobule),
                         'thylakoid_membrane': [0] * len(generate_plastoglobule)}
    df3 = pd.DataFrame(data_3_test_label)

    data_4_test_label = {'envelope': [0] * len(generate_stroma),
               'lumen': [0] * len(generate_stroma),
                'plastoglobule': [0] * len(generate_stroma),
               'stroma': [1] * len(generate_stroma),
               'thylakoid_membrane': [0] * len(generate_stroma)}
    df4 = pd.DataFrame(data_4_test_label)

    data_5_test_label = {'envelope': [0] * len(generate_thylakoid_membrane),
               'lumen': [0] * len(generate_thylakoid_membrane),
                'plastoglobule': [0] * len(generate_thylakoid_membrane),
               'stroma': [0] * len(generate_thylakoid_membrane),
               'thylakoid_membrane': [1] * len(generate_thylakoid_membrane)}
    df5 = pd.DataFrame(data_5_test_label)

    all_G_feature = pd.concat([generate_envelope, generate_lumen, generate_plastoglobule, generate_stroma, generate_thylakoid_membrane], axis=0)
    all_G_label = pd.concat([df1, df2, df3, df4, df5], axis=0)
    return all_G_feature, all_G_label