import torch


class ModelClassify(torch.nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.fc1 = torch.nn.Linear(feature_num, 20)
        self.fc2 = torch.nn.Linear(20, 20)
        self.fc3 = torch.nn.Linear(20, 20)
        self.fc4 = torch.nn.Linear(20, 20)
        self.fc5 = torch.nn.Linear(20, 4)

        self.bn1 = torch.nn.BatchNorm1d(20)
        self.bn2 = torch.nn.BatchNorm1d(20)
        self.bn3 = torch.nn.BatchNorm1d(20)
        self.bn4 = torch.nn.BatchNorm1d(20)
        self.sigmoid = torch.nn.Sigmoid()
        self.drop_rate = 0.4

    def forward(self, x):

        out = self.fc1(x)
        out = torch.nn.Dropout(self.drop_rate)(out)
        out = torch.nn.ReLU()(out)
        out = self.bn1(out)

        # out = self.fc2(out)
        # out = torch.nn.Dropout(self.drop_rate)(out)
        # out = torch.nn.ReLU()(out)
        # out = self.bn2(out)

        # out = self.fc3(out)
        # out = torch.nn.Dropout(self.drop_rate)(out)
        # out = torch.nn.ReLU()(out)
        # out = self.bn3(out)

        out = self.fc4(out)
        out = torch.nn.Dropout(self.drop_rate)(out)
        out = torch.nn.ReLU()(out)
        out = self.bn4(out)

        out = self.fc5(out)
        out = self.sigmoid(out)

        return out