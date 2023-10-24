import pandas as pd


def processRealData(real_feature_pd, real_label_pd, feature_num):
    feature_with_label = pd.concat([real_feature_pd, real_label_pd], axis=1)
    not_plastoglobule_df = feature_with_label[feature_with_label['plastoglobule'] != 1]
    not_plastoglobule_feature_pd = not_plastoglobule_df.iloc[:, :feature_num]
    not_plastoglobule_label_pd = not_plastoglobule_df.iloc[:, feature_num:]
    not_plastoglobule_label_pd = not_plastoglobule_label_pd.drop(columns=['plastoglobule'])
    return not_plastoglobule_feature_pd, not_plastoglobule_label_pd