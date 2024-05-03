# %%
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class CNNModel(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.model.classifier[3] = nn.Linear(1024, embedding_size)
        
    def forward(self, x):
        return self.model(x)

class VideoSequenceModel(nn.Module):
    def __init__(self, sequence_length: int, embedding_size, lstm_hidden, lstm_layers, num_classes):
        super().__init__()
        self.sequence_length = sequence_length
        self.cnn_models_1 = CNNModel(embedding_size)
        self.cnn_models_2 = CNNModel(embedding_size)
        self.cnn_models_3 = CNNModel(embedding_size)
        self.cnn_models_4 = CNNModel(embedding_size)
        self.cnn_models_5 = CNNModel(embedding_size)
        self.cnn_models_6 = CNNModel(embedding_size)
        self.cnn_models_7 = CNNModel(embedding_size)
        self.cnn_models_8 = CNNModel(embedding_size)
        self.cnn_models_9 = CNNModel(embedding_size)
        self.cnn_models_10 = CNNModel(embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_hidden, lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        self.cnn_1 = self.cnn_models_1(x[:, 0, ...])
        self.cnn_2 = self.cnn_models_2(x[:, 1, ...])
        self.cnn_3 = self.cnn_models_3(x[:, 2, ...])
        self.cnn_4 = self.cnn_models_4(x[:, 3, ...])
        self.cnn_5 = self.cnn_models_5(x[:, 4, ...])
        self.cnn_6 = self.cnn_models_6(x[:, 5, ...])
        self.cnn_7 = self.cnn_models_7(x[:, 6, ...])
        self.cnn_8 = self.cnn_models_8(x[:, 7, ...])
        self.cnn_9 = self.cnn_models_9(x[:, 8, ...])
        self.cnn_10 = self.cnn_models_10(x[:, 9, ...])
        x = torch.stack([
            self.cnn_1, self.cnn_2, self.cnn_3, self.cnn_4, self.cnn_5, self.cnn_6, self.cnn_7, self.cnn_8, self.cnn_9, self.cnn_10
        ], dim=1)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# %%
if __name__ == "__main__":
    from torchsummary import summary
    
    X = torch.randn(4, 5, 3, 64, 64)
    
    model = VideoSequenceModel(
        sequence_length=5,
        embedding_size=128,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=10
    )
    
    
    y = model(X)
    print(y.shape)