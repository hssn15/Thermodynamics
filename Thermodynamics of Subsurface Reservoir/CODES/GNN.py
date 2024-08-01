import numpy as np
import torch_geometric
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

## Architecture ##
import torch
from torch.nn import Linear, Tanh
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import Sequential, GCNConv

class ENC_DEC(torch.nn.Module):
    def __init__(self,in_features, embedding_dim, hidden_channels, n_nodes):
        super().__init__()
        # encoding
        self.enc_block1_1 = Sequential('x, edge_index, edge_weight', [(GCNConv(in_channels = in_features, out_channels = hidden_channels[0]),          'x, edge_index, edge_weight -> x'),   Tanh()])
        self.enc_block2_1 = Sequential('x, edge_index, edge_weight', [(GCNConv(in_channels = in_features, out_channels = hidden_channels[0]),          'x, edge_index, edge_weight -> x'),   Tanh()])
        self.enc_block3_1 = Sequential('x, edge_index, edge_weight', [(GCNConv(in_channels = in_features, out_channels = hidden_channels[0]),          'x, edge_index, edge_weight -> x'),   Tanh()])

        self.enc_block4_2 = Sequential('x, edge_index, edge_weight', [(GCNConv(in_channels = hidden_channels[0], out_channels = hidden_channels[1]),   'x, edge_index, edge_weight -> x'),   Tanh()])
        self.enc_block5_2 = Sequential('x, edge_index, edge_weight', [(GCNConv(in_channels = hidden_channels[0], out_channels = hidden_channels[1]),   'x, edge_index, edge_weight -> x'),   Tanh()])

        self.enc_block6_3 = Sequential('x, edge_index, edge_weight', [(GCNConv(in_channels = hidden_channels[1], out_channels = embedding_dim),        'x, edge_index, edge_weight -> x'),   Tanh()])

    def forward(self,h0, edge_index, edge_weight):
        # encode
        h1 = self.enc_block1_1(h0, edge_index, edge_weight)
        h2 = self.enc_block2_1(h0, edge_index, edge_weight)
        h3 = self.enc_block3_1(h0, edge_index, edge_weight)

        h4 = self.enc_block4_2(h1*h2*h3, edge_index, edge_weight)
        h5 = self.enc_block5_2(h1*h2*h3, edge_index, edge_weight)
 
        h6 = self.enc_block6_3(h4*h5, edge_index, edge_weight)
        
        return h6

class GCN(torch.nn.Module):
    def __init__(self, in_features, embedding_dim, hidden_channels, n_nodes):
        super().__init__()
        
        # Deploying Pressure and Saturation Model
        self.model = ENC_DEC(in_features, embedding_dim, hidden_channels, n_nodes)
        self.loss_fn = torch.nn.MSELoss()
        
    def predict(self, h0, edge_index, edge_weight):
        prediction = self.model.forward(h0, edge_index, edge_weight)
        return prediction
    
    def Loss(self, prediction, target):
        loss = self.loss_fn(prediction, target)
        return loss
    

####### Data Prep ##########
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


XY = torch.tensor(XY, dtype=torch.float).to(device="cuda")
phi_LV = torch.tensor(phi_LV, dtype=torch.float).to(device="cuda")
V = torch.tensor(np.array(V), dtype=torch.float).to(device="cuda")
K = torch.tensor(K, dtype=torch.float).to(device="cuda")

scaled_Data = [] # Graph Data Holder
# Converting data to Graph  
for i in range(len(scaled_feature_matrices)):
    scaled_Data.append(torch_geometric.data.Data(x=scaled_feature_matrices[i], edge_index=edge_index, edge_attr=edge_weight, y=XY[i]))

def split(data_list, train_size):
    import random
    test_list = []
    train_list = []
    selected_idx = []
    selected_idx_test = []
    num0ftrain = int(round(len(data_list)*train_size, 0))
    for i in range(num0ftrain):
        while True:
            get_index = random.randrange(len(data_list))
            if not get_index in selected_idx:
                selected_idx.append(get_index)
                train_list.append(data_list[get_index])
                break
    for i in range(int(len(data_list))):
        if not i in selected_idx:
            selected_idx_test.append(i)
            test_list.append(data_list[i])
    
    return train_list, test_list,selected_idx, selected_idx_test
train_list, test_list, train_idx, test_idx = split(scaled_Data, 0.67)

train_list, test_list, train_idx, test_idx = split(scaled_Data, 0.67)

train_loader = DataLoader(train_list, batch_size=len(train_list), shuffle=False)
test_loader = DataLoader(test_list, batch_size=len(test_list), shuffle=False)

batch_test = next(iter(test_loader))

### Optimizer ###
def training_w_LBFGS(model, train_data, test_data, lr, max_iter, history_size, tolerance_grad, tolerance_change, line_search_fn, n, save_model = False):
    training_loss_list = []
    test_loss_list = []
    # Setting Optimizer
    optimizer = torch.optim.LBFGS(model.parameters(),lr=lr, max_iter=max_iter, history_size = history_size, tolerance_grad=tolerance_grad, tolerance_change = tolerance_change, line_search_fn = line_search_fn)
    # Training Loop
    for batch_train in train_data:
        for batch_test in test_data:
            def closure():
                # initialize optimizer
                optimizer.zero_grad()
                
                # forward pass
                h0, edge_index, edge_weight, train_true = batch_train.x, batch_train.edge_index, batch_train.edge_attr, batch_train.y
                train_prediction = model.predict(h0, edge_index, edge_weight)
                train_loss = model.Loss(train_prediction, train_true)

                h0, edge_index, edge_weight, test_true = batch_test.x, batch_test.edge_index, batch_test.edge_attr, batch_test.y
                test_prediction = model.predict(h0, edge_index, edge_weight)
                test_loss = model.Loss(test_prediction, test_true)

                # collecting loss
                print("#Epoch = {}".format(len(training_loss_list))," | Train Loss = {}".format(train_loss.item()), " | Test Loss = {}".format(test_loss.item()))
                training_loss_list.append(train_loss.item()) 
                test_loss_list.append(test_loss.item()) 

                # backward pass
                train_loss.backward()
                return train_loss
            optimizer.step(closure)
    if save_model == True:
        torch.save(model.state_dict(), "lbfgs_model_{}.pth".format(n))
    return training_loss_list, test_loss_list

### Training #####
in_features, embedding_dim, hidden_channels, n_nodes = 6, 2, [36, 24, 12], 9
model = GCN(in_features, embedding_dim, hidden_channels, n_nodes).to(device="cuda")
lr = 1.0
N_epochs = 1
max_iter = 2000
history_size = 300
line_search_fn = 'strong_wolfe'
tolerance_grad = 1e-07
tolerance_change = 1e-09
n = 0
train_loss, test_loss = training_w_LBFGS(model, train_loader, test_loader, lr, max_iter, history_size, tolerance_grad, tolerance_change, line_search_fn, n, save_model = False)


plt.plot(train_loss, "^b", label = "Train Loss")
plt.plot(test_loss, "r", label = "Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()
in_features, embedding_dim, hidden_channels, n_nodes = 6, 2, [24, 12], 9
model = GCN(in_features, embedding_dim, hidden_channels, n_nodes).to(device="cuda")
model.load_state_dict(torch.load("lbfgs_gnn_model.pth"))

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
    plt.plot([0, 1], [0, 1], 'k-.')
    plt.ylabel('Liquid Phase (X) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Liquid Phase (X) True', fontsize=fs)

    plt.subplot(2, 2, 2)
    plt.plot(L_train_true, L_train_pred, 'bo', label="Train: R2 = {}".format(round(r2_train_L, 3)))
    plt.plot([0, 1], [0, 1], 'k-.')
    plt.ylabel('Liquid Phase (X) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Liquid Phase (X) True', fontsize=fs)

    plt.subplot(2, 2, 3)
    plt.plot(V_test_true, V_test_pred, 'ro', label="Test: R2 = {}".format(round(r2_test_V, 3)))
    plt.plot([0, 1], [0, 1], 'k-.')
    plt.ylabel('Vapor Phase (Y) Prediction', fontsize=fs)
    plt.legend()
    plt.xlabel('Vapor Phase (Y) True', fontsize=fs)

    plt.subplot(2, 2, 4)
    plt.plot(V_train_true, V_train_pred, 'bo', label="Train: R2 = {}".format(round(r2_train_V, 3)))
    plt.plot([0, 1], [0, 1], 'k-.')
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


### Prediction ##

batch_train = next(iter(train_loader))
h0, edge_index, edge_weight, train_true = batch_train.x, batch_train.edge_index, batch_train.edge_attr, batch_train.y
train_prediction = model.predict(h0, edge_index, edge_weight).view(-1, 9, 2)

batch_test = next(iter(test_loader))
h0, edge_index, edge_weight, test_true = batch_test.x, batch_test.edge_index, batch_test.edge_attr, batch_test.y
test_prediction = model.predict(h0, edge_index, edge_weight).view(-1, 9, 2)

test_idx = np.array(test_idx)
test_idx_dat = np.genfromtxt("idx_test.dat", dtype=np.integer)

common_idx = []
common_idx_dat = []
for i in range(len(test_idx)):
    if test_idx[i] in test_idx_dat:
        common_idx.append(i)
        common_idx_dat.append(np.where(test_idx_dat == test_idx[i])[0])

common_idx_dat = np.array(common_idx_dat)
np.savetxt("common_idx_dat.dat", common_idx_dat)

for i in common_idx[-4:]:
    sample_no = test_idx[i]
    L_test_true_1 = (test_true.cpu().numpy().reshape(-1, 9, 2))[i][:, 0]
    V_test_true_1 = (test_true.cpu().numpy().reshape(-1, 9, 2))[i][:, 1]
    L_test_pred_1 = test_prediction.cpu().detach().numpy()[i][:, 0]
    V_test_pred_1 = test_prediction.cpu().detach().numpy()[i][:, 1]

    R2_VL_test_plot(L_test_true_1, L_test_pred_1, V_test_true_1, V_test_pred_1, sample_no)

L_train_true = ((train_true.cpu().numpy()))[:, 0]
V_train_true = ((train_true.cpu().numpy()))[:, 1]
L_train_pred = (train_prediction.cpu().detach().numpy().reshape(len(train_list)*9, 2))[:, 0]
V_train_pred = (train_prediction.cpu().detach().numpy().reshape(len(train_list)*9, 2))[:, 1]

L_test_true= ((test_true.cpu().numpy()))[:, 0]
V_test_true= ((test_true.cpu().numpy()))[:, 1]
L_test_pred= (test_prediction.cpu().detach().numpy().reshape(len(test_list)*9, 2))[:, 0]
V_test_pred= (test_prediction.cpu().detach().numpy().reshape(len(test_list)*9, 2))[:, 1]

mask1_L = L_train_true>0 
mask2_L = L_train_true<1
mask1_V = V_train_true>0 
mask2_V = V_train_true<1

R2_plot(L_train_true[mask1_L & mask2_L], L_train_pred[mask1_L & mask2_L], L_test_true, L_test_pred, V_train_true[mask1_V & mask2_V], V_train_pred[mask1_V & mask2_V], V_test_true, V_test_pred)