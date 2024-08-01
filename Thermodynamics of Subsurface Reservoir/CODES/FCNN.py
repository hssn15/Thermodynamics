import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score

# Architecture

class NN_model(torch.nn.Module):

    def __init__(self,layers):
        super().__init__()

        self.layers = layers
        self.activation = torch.nn.Tanh()
        self.activation_sig = torch.nn.Sigmoid()
        self.loss_function = torch.nn.MSELoss(reduction ='mean')
        self.linears = torch.nn.ModuleList([torch.nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])

        for i in range(len(layers)-1):
            torch.nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            torch.nn.init.zeros_(self.linears[i].bias.data)

    def forward(self,x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.layers)-2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        a = self.activation_sig(a)
        return a
    
    def loss(self, x, y_target ):
        L = self.loss_function(self.forward(x), y_target)
        return L
    
    def predict(self, x_test):
        y_pred = self.forward(x_test)
        y_pred = y_pred.cpu().detach().numpy()
        return y_pred
    
### Data Preparation ###

feature_matrices = np.genfromtxt("X_samples.dat").reshape((22832, 9, 6))
edge_index = torch.tensor(np.genfromtxt("edge_index.dat").T, dtype=torch.int).to(device="cuda")
edge_weight = torch.tensor(np.genfromtxt("edge_weight.dat").reshape(-1, 1), dtype=torch.float).to(device="cuda")

scalerX = StandardScaler()
scalerX.fit(feature_matrices.reshape((22832*9, 6)))
scaled_feature_matrices = torch.tensor((scalerX.transform(feature_matrices.reshape((22832*9, 6)))).reshape((22832, 9, 6)), dtype=torch.float).to(device="cuda")
sample_results_dict = np.load("sample_results.npy",allow_pickle='TRUE').item()

XY = []
phi_LV = []
V = []
K = []
sample_results_list = list(sample_results_dict.items())

for i in range(len(sample_results_list)):
    xy = np.zeros((9, 2))
    phi_lv = np.zeros((9, 2))
    k = np.zeros((9, 1))

    xy[:, 0] = sample_results_list[i][1]["X"].cpu().numpy()
    xy[:, 1] = sample_results_list[i][1]["Y"].cpu().numpy()

    phi_lv[:, 0] = sample_results_list[i][1]["phi_L"].cpu().numpy()
    phi_lv[:, 1] = sample_results_list[i][1]["phi_V"].cpu().numpy()

    k[:, 0] = sample_results_list[i][1]["K"].cpu().numpy()

    XY.append(xy)
    phi_LV.append(phi_lv)
    K.append(k)
    V.append(sample_results_list[i][1]["V"].cpu().numpy())

XY = np.array(XY)

Y = XY.reshape(22832, 9*2)
X = scaled_feature_matrices.reshape(22832, 9*6)

indices = np.linspace(0, 22831, 22832, dtype=np.int32)

X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(X, Y, indices, test_size=0.33, random_state=123)

X_train = torch.tensor(X_train, dtype=torch.float).to(device = "cuda")
X_test = torch.tensor(X_test, dtype=torch.float).to(device = "cuda")
Y_train = torch.tensor(Y_train, dtype=torch.float).to(device = "cuda")
Y_test = torch.tensor(Y_test, dtype=torch.float).to(device = "cuda")

##### Optimizer ########
input_dimension = X.shape[1]
output_dimension = Y.shape[1]
layer_architectures = [[input_dimension, 20, 10, 20, output_dimension]]
# Adam Optimizer Parameters
learing_rate_Adam = [0.5e-3, 1e-3, 1.5e-3]
N_epochs_lst_Adam = [1000, 2000,  4000]
eps= 1e-08
weight_decay=0
amsgrad_lst = [False, True]

# LBFGS Oprimizer Parameters
learing_rate_LBFGS = [1.0, 1.5, 2.0]
N_epochs_lst_LBFGS = [1, 3, 5]
max_iter_lst=[100, 1000, 10000]
history_size_lst = [5, 10, 15]
line_search_fn = 'strong_wolfe'
tolerance_grad = 1e-07
tolerance_change = 1e-09

def training_w_Adam(model, x_test, x_train, y_test_target, y_train_target, lr, N_epochs, eps, weight_decay, amsgrad, n, save_model = False):
    training_loss_list = []
    test_loss_list = []
    # Setting Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    # Training Loop
    for i in range(N_epochs):
        # initialize optimizer
        optimizer.zero_grad()
        # forward pass
        train_loss = model.loss(x_train, y_train_target)
        test_loss = model.loss(x_test, y_test_target)
         # collecting loss
        test_loss_list.append(test_loss.item())
        training_loss_list.append(train_loss.item())
        # print loss values
        print("# Epoch  = {} ".format(i), " | Train Loss = {}".format(train_loss.item()), " | Test Loss = {}".format(test_loss.item())) 
        # backward pass
        train_loss.backward()
        optimizer.step()
    if save_model == True:
        torch.save(model.state_dict(), r"C:\Users\YUSIFOH\ERPE 394A\Huseyn_Yusifov_Homework_4\adam saved nn models\adam_model_{}.pth".format(n))
    return training_loss_list, test_loss_list

def training_w_LBFGS(model, x_test, x_train, y_test_target, y_train_target, lr, N_epochs, max_iter, history_size, tolerance_grad, tolerance_change, line_search_fn, n, save_model = False):
    training_loss_list = []
    test_loss_list = []
    # Setting Optimizer
    optimizer = torch.optim.LBFGS(model.parameters(),lr=lr, max_iter=max_iter, history_size = history_size, tolerance_grad=tolerance_grad, tolerance_change = tolerance_change, line_search_fn = line_search_fn)
    # Training Loop
    for i in range(N_epochs):
        def closure():
            # initialize optimizer
            optimizer.zero_grad()
            # forward pass
            train_loss = model.loss(x_train, y_train_target)
            test_loss = model.loss(x_test, y_test_target)
            # collecting loss
            training_loss_list.append(train_loss.item()) 
            test_loss_list.append(test_loss.item()) 
            print(" | Train Loss = {}".format(train_loss.item()), " | Test Loss = {}".format(test_loss.item()))
            # backward pass
            train_loss.backward()
            return train_loss
        optimizer.step(closure)
    if save_model == True:
        torch.save(model.state_dict(), r"C:\Users\YUSIFOH\ERPE 394A\Huseyn_Yusifov_Homework_4\lbfgs saved nn models\lbfgs_model_{}.pth".format(n))
    return training_loss_list, test_loss_list

######### Training #############
# Training and Testing By choosing random parameters: Adam
layers = [input_dimension, 25, 20, 15, 20, 25, output_dimension]
model = NN_model(layers).to(device="cuda")
lr = 1e-3
N_epochs = 15000
amsgrad = False
n = 0
train_loss, test_loss = training_w_Adam(model, X_test, X_train, Y_test, Y_train, lr, N_epochs, eps, weight_decay, amsgrad, n, save_model= False)

plt.plot(train_loss, "^b", label = " Train Loss")
plt.plot(test_loss, "r", label = " Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()

def R2_plot(L_train_true, L_train_pred, L_test_true, L_test_pred, V_train_true, V_train_pred, V_test_true, V_test_pred):
    """R2 plot calculation"""
    # Get prediciton score in terms of R2
    r2_test_L  = r2_score(L_test_true ,  L_test_pred)
    r2_train_L = r2_score(L_train_true,  L_train_pred)

    r2_test_V  = r2_score(V_test_true ,  V_test_pred)
    r2_train_V = r2_score(V_train_true,  V_train_pred)
    # Plot parity plots 
    fs = 9
    plt.figure(figsize=(6, 6), dpi = 200)
    plt.suptitle('Parity Plot', fontsize=fs)

    plt.subplot(2, 2, 1)
    plt.plot(L_test_true, L_test_pred, 'ro', label="Test: R2 = {}".format(round(r2_test_L, 3)))
    plt.plot([L_test_true.min(), L_test_true.max()], [L_test_true.min(), L_test_true.max()], 'k-.')
    plt.ylabel('Liquid Phase (X) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Liquid Phase (X) True', fontsize=fs)

    plt.subplot(2, 2, 2)
    plt.plot(L_train_true, L_train_pred, 'bo', label="Train: R2 = {}".format(round(r2_train_L, 3)))
    plt.plot([L_train_true.min(), L_train_true.max()], [L_train_true.min(), L_train_true.max()], 'k-.')
    plt.ylabel('Liquid Phase (X) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Liquid Phase (X) True', fontsize=fs)

    plt.subplot(2, 2, 3)
    plt.plot(V_test_true, V_test_pred, 'ro', label="Test: R2 = {}".format(round(r2_test_V, 3)))
    plt.plot([V_test_true.min(), V_test_true.max()], [V_test_true.min(), V_test_true.max()], 'k-.')
    plt.ylabel('Vapor Phase (Y) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Vapor Phase (Y) True', fontsize=fs)

    plt.subplot(2, 2, 4)
    plt.plot(V_train_true, V_train_pred, 'bo', label="Train: R2 = {}".format(round(r2_train_V, 3)))
    plt.plot([V_train_true.min(), V_train_true.max()], [V_train_true.min(), V_train_true.max()], 'k-.')
    plt.ylabel('Vapor Phase (Y) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Vapor Phase (Y) True', fontsize=fs)
    plt.tight_layout()
    plt.show()

def R2_VL_test_plot(L_test_true, L_test_pred, V_test_true, V_test_pred, sample_no):
    components = ["C1","C2","C3","nC4","nC5","C6","C7+","CO2","N2"]
    colormap = np.array(["r", "g", "b", "c", "m", "y", "k", "grey", "brown"])
    """R2 plot calculation"""
    # Get prediciton score in terms of R2
    r2_test_L  = r2_score(L_test_true ,  L_test_pred)

    r2_test_V  = r2_score(V_test_true ,  V_test_pred)
    # Plot parity plots 
    fs = 9
    plt.figure(figsize=(5, 5), dpi=200)
    plt.suptitle('Parity Plot: Sample Number # {} P = {} MPa  T = {} K'.format(sample_no, round(feature_matrices[sample_no][0, 0]), round(feature_matrices[sample_no][0, 1])), fontsize=fs)
    plt.subplot(2, 1, 1)
    #plt.scatter(L_test_true, L_test_pred, c = colormap, label="Test: R2 = {}".format(round(r2_test_L, 3)))
    for i in range(9):
        plt.scatter(L_test_true[i], L_test_pred[i], color = "{}".format(colormap[i]), label = "{}".format(components[i]))
    plt.plot([L_test_true.min(), L_test_true.max()], [L_test_true.min(), L_test_true.max()], 'k-.')
    plt.text(0, 0.6, "Test: R2 = {}".format(round(r2_test_L, 3)))
    plt.ylabel('Liquid Phase (X) Prediction', fontsize=fs)
    plt.legend(loc = (1,0))
    plt.xlabel('Liquid Phase (X) True', fontsize=fs)


    plt.subplot(2, 1, 2)
    #plt.scatter(V_test_true, V_test_pred, c = colormap, label="Test: R2 = {}".format(round(r2_test_V, 3)))
    for i in range(9):
        plt.scatter(V_test_true[i], V_test_pred[i], color = "{}".format(colormap[i]), label = "{}".format(components[i]))
    plt.plot([V_test_true.min(), V_test_true.max()], [V_test_true.min(), V_test_true.max()], 'k-.')
    plt.text(0, 0.6, "Test: R2 = {}".format(round(r2_test_V, 3)))
    plt.ylabel('Vapor Phase (Y) Prediction', fontsize=fs)
    plt.legend(loc = (1,0))
    plt.xlabel('Vapor Phase (Y) True', fontsize=fs)
    
    plt.tight_layout()
    #plt.savefig(r'C:\Users\YUSIFOH\NNs\thermo_project\results\sample_{}.jpeg'.format(sample_no))
    plt.show()

Y_pred_test = model.predict(X_test).reshape(len(X_test), 9, 2)
Y_pred_train = model.predict(X_train).reshape(len(X_train), 9, 2)

common_idx_test = np.genfromtxt("common_idx_dat.dat", dtype=np.integer)

for i in common_idx_test[-4:]:
    sample_no = idx_test[i]
    L_test_true_1 = ((Y_test.cpu().numpy()).reshape(len(X_test), 9, 2))[i][:, 0]
    V_test_true_1 = ((Y_test.cpu().numpy()).reshape(len(X_test), 9, 2))[i][:, 1]
    L_test_pred_1 = Y_pred_test[i][:, 0]
    V_test_pred_1 = Y_pred_test[i][:, 1]

    R2_VL_test_plot(L_test_true_1, L_test_pred_1, V_test_true_1, V_test_pred_1, sample_no)

    L_train_true = ((Y_train.cpu().numpy()).reshape(len(X_train)*9, 2))[:, 0]
V_train_true = ((Y_train.cpu().numpy()).reshape(len(X_train)*9, 2))[:, 1]
L_train_pred = (model.predict(X_train).reshape(len(X_train)*9, 2))[:, 0]
V_train_pred = (model.predict(X_train).reshape(len(X_train)*9, 2))[:, 1]

L_test_true= ((Y_test.cpu().numpy()).reshape(len(X_test)*9, 2))[:, 0]
V_test_true= ((Y_test.cpu().numpy()).reshape(len(X_test)*9, 2))[:, 1]
L_test_pred= (model.predict(X_test).reshape(len(X_test)*9, 2))[:, 0]
V_test_pred= (model.predict(X_test).reshape(len(X_test)*9, 2))[:, 1]

R2_plot(L_train_true, L_train_pred, L_test_true, L_test_pred, V_train_true, V_train_pred, V_test_true, V_test_pred)