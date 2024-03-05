### Sección 1: Conceptos base
# %%  1)

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split

import numpy as np

# from tqdm.notebook import tqdm
from tqdm import tqdm

class ModelExample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softamx = nn.LogSoftmax(dim=1)
        
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(13872, 100)
        self.f2 = nn.Linear(100, 64)
        self.f3 = nn.Linear(64, 5)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.f1(x))
        x = self.sigmoid(self.f2(x))
        x = self.f3(x)
        # softmax not implemented to 
        # implement cross entropy loss
        return x
        



# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# a) crea un tensor con las siguientes dimensiones: (25, 3, 68, 68) e inicializalo con valores aleatorios entre 0.0 y 255.0  . 

X: torch.Tensor = torch.rand(25, 3, 68, 68) * 255.0
X: torch.Tensor = X.to(device)

# b) crea una instancia de tu red neuronal.
model_example_1 = ModelExample1()

# c) invoca la red neuronal con el tensor de entrada e imprime las dimensiones del tensor de salida.
out = model_example_1(X)
print(out.shape)
# %% 2)

def fit_t(
        M, 
        X: torch.Tensor, 
        Y: torch.Tensor,
        epochs: int, 
        batch_size: int, 
        loss, 
        optimizer, 
        checkpoint_path: str = None
    ):
    
    loss_history = []
    
    M.to(device) # move model to device
    
    # batch size it and shuffle
    dataset = TensorDataset(X, Y)
    dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        losses = []
        for x, y in tqdm(dataset):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = M(x)
            loss_ = loss(y_pred, y)
            loss_.backward()
            optimizer.step()
            
            losses.append(loss_.item())
        
        loss_history.append(np.mean(losses))
        print(f"Epoch {epoch} loss: {np.mean(losses)}")
        
        if checkpoint_path is not None:
        
            torch.save(
                {
                    "model_state_dict": M.state_dict(),
                    "loss_history": loss_history
                },
                checkpoint_path
            )

    return M, loss_history

# %%

# a) Definamos un tensor de entrada con las siguientes dimensiones: (777, 3, 68, 68). Con valores entre 0.0 y 1.0.
X = torch.rand(777, 3, 68, 68)


# b) Definamos un tensor de salida con las siguientes dimensiones: (777, 5)  y con valores entre 0 y 4 en una representación one hot vector.
Y = torch.randint(0, 5, (777,))
Y = nn.functional.one_hot(Y, num_classes=5).float()

# c) Implementa una función u objeto iterable que te permita acceder a cada uno de los batch durante el ciclo de entrenamiento.

# d) Crea una instancia del modelo que implementaste al iniciar la actividad (red neuronal para clasificación).
model_example_2 = ModelExample1()

# e) Utiliza la función fit_t() para entrenar la instancia del modelo con los datos sintéticos. Para inicializar el resto de los parámetros considera las características de los datos de entrada y salida, y que la tarea que se espera realice la red neuronal es una clasificación multiclase.
optimizer = optim.Adam(model_example_2.parameters(), lr=0.001)
loss = nn.CrossEntropyLoss()

model_example_2, history = fit_t(
    model_example_2,
    X,
    Y,
    epochs=10,
    batch_size=32,
    loss=loss,
    optimizer=optimizer,
    checkpoint_path="model_example_2.pckl"
)

# f) Al concluir el entrenamiento, no olvides verificar que los pesos y el loss total esten almacenados correctamente
loaded_model = ModelExample1()
state = torch.load("model_example_2.pckl")
loaded_model.load_state_dict(state["model_state_dict"])

loss_history = state["loss_history"]

# %% 3)

def fit_tv(
        M, 
        dataset_train: torch.utils.data.TensorDataset,
        dataset_val: torch.utils.data.TensorDataset,
        epochs: int, 
        batch_size: int,
        loss, 
        optimizer, 
        checkpoint_path: str = None
    ):
    
    
    dataset_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataset_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
    
    loss_history_train, loss_history_eval = [], []
    
    M.to(device) # move model to device

    for epoch in range(epochs):
        losses = []
        for x, y in tqdm(dataset_train):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            y_pred = M(x)
            loss_ = loss(y_pred, y)
            loss_.backward()
            optimizer.step()
            
            losses.append(loss_.item())
        
        loss_history_train.append(np.mean(losses))
        print(f"Epoch {epoch} loss: {np.mean(losses)}")
        
        with torch.no_grad():
            losses_eval = []
            for x, y in tqdm(dataset_val):
                x, y = x.to(device), y.to(device)
                y_pred = M(x)
                loss_ = loss(y_pred, y)
                losses_eval.append(loss_.item())
                
            loss_history_eval.append(np.mean(losses_eval))
            
        
        if checkpoint_path is not None:
        
            torch.save(
                {
                    "model_state_dict": M.state_dict(),
                    "train_loss_history": loss_history_train,
                    "validaation_loss_history": loss_history_eval
                },
                checkpoint_path
            )

    return M, loss_history





# %%

# a) Definamos, 

# un tensor de entrada con las siguientes dimensiones: (1000, 3, 68, 68). Con valores entre 0.0 y 1.0.
# un tensor de salida con las siguientes dimensiones: (1000, 5)  y con valores entre 0 y 4 en una representación one hot vector.
# b) De los tensores anteriores utiliza el 80% para entrenamiento (X_train, Y_train) y el resto para validación (X_val, Y_val)
X = torch.rand(1000, 3, 68, 68)
Y = torch.randint(0, 5, (1000,))
Y = nn.functional.one_hot(Y, num_classes=5).float()
dataset = TensorDataset(X, Y)

dataset_size = X.shape[0]
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size


dataset_train, dataset_val = random_split(dataset, [train_size, val_size])

# c) Implementa una función u objeto iterable que te permita acceder a cada uno de los batch (datos de entrenamiento) durante el ciclo de entrenamiento.

# d) Implementa una función u objeto iterable que te permita acceder a cada uno de los batch (datos de validación) durante el ciclo de validación.

# e) Crea una instancia del modelo que implementaste al iniciar la actividad (red neuronal para clasificación).
red_neuronal_para_clasificacion = ModelExample1()
# f) Utiliza la función fit_tv() para entrenar y validar la instancia del modelo con los datos sintéticos. Para inicializar el resto de los parámetros considera las características de los datos de entrada y salida, y que la tarea que se espera realice la red neuronal es una clasificación multiclase.
fit_tv(
    red_neuronal_para_clasificacion,
    dataset_train,
    dataset_val,
    epochs=10,
    batch_size=32,
    loss=loss,
    optimizer=optimizer,
    checkpoint_path="red_neuronal_para_clasificacion.pckl"

)
# g) Al concluir el entrenamiento, no olvides verificar que los pesos, el loss total de entrenamiento y el loss total de validación esten almacenados correctamente

# %%
