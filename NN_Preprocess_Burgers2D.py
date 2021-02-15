import scipy.io as sio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch.autograd import Variable
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from torchvision import datasets
from pytorchtools import EarlyStopping
import time
start_time = time.time()
np.random.seed(seed=1)
torch.manual_seed(0)



Data = sio.loadmat('dataBurgers2DN40.mat')
x = Data["x"] 
y = Data["y"] 
t = Data["t"] 
u = Data["u"] 
xTrn = Data["xTrn"] 
xVal = Data["xVal"]
yTrn = Data["yTrn"] 
yVal = Data["yVal"]
tTrn = Data["tTrn"] 
tVal = Data["tVal"]  
uTrn = Data["uTrn"] 
uVal = Data["uVal"] 
# coefficients of PDE
# Data = sio.loadmat('c_pred.mat')
# c_pred = Data["c_pred"] 
# Convert numpy arrays to torch Variables
X = Variable(torch.from_numpy(x).float(),requires_grad=True)
Y = Variable(torch.from_numpy(y).float(),requires_grad=True)
T = Variable(torch.from_numpy(t).float(),requires_grad=True)
U = Variable(torch.from_numpy(u).float(),requires_grad=True)
XTrn = Variable(torch.from_numpy(xTrn).float(),requires_grad=True)
YTrn = Variable(torch.from_numpy(yTrn).float(),requires_grad=True)
TTrn = Variable(torch.from_numpy(tTrn).float(),requires_grad=True)
UTrn = Variable(torch.from_numpy(uTrn).float(),requires_grad=True)
XVal = Variable(torch.from_numpy(xVal).float(),requires_grad=True)
YVal = Variable(torch.from_numpy(yVal).float(),requires_grad=True)
TVal = Variable(torch.from_numpy(tVal).float(),requires_grad=True)
UVal = Variable(torch.from_numpy(uVal).float(),requires_grad=True)
# c_pred = Variable(torch.from_numpy(c_pred).float(),requires_grad=True)

# Hyperparameters
# gamma: regularizer of data loss
# lmbd1: regularizer of l1 norm of model weights
# lmbd2: regularizer of l2 norm of model weights
# alpha1: regularizer of physical loss 1: PDE
# alpha2: regularizer of physical loss 2: initial condition
# alpha3: regularizer of physical loss 3: boundary condition
# alpha4: regularizer of physical loss 4: boundary condition
gamma = 1E1; 
lmbd1 = 0;
lmbd2 = 1E-10;
alpha1 = 0# 1E1
alpha2 = 0# 1E-1;
alpha3 = 0# 1E-1;
alpha4 = 0# 1E-1;
sumAlpha = alpha1+alpha2+alpha3+alpha4
batch_size = 512
class MyNet1(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MyNet1, self).__init__()
        self.fc1 = nn.Linear(D_in, H[0])
        self.fc2 = nn.Linear(H[0], H[1])
        self.fc3 = nn.Linear(H[1], H[2])
        self.fc4 = nn.Linear(H[2], D_out)
    def forward(self, x, y, t):
        xyt = torch.cat([x, y, t], dim=1)
        f = nn.Tanh()(self.fc1(xyt))
        f = nn.Tanh()(self.fc2(f))
        f = nn.Tanh()(self.fc3(f))
        f = self.fc4(f)
        return f

def lossCal(output,target):
    ## calculate the loss
    # data loss on training data
    data_loss = loss_fn(output,target)
    phy_loss1 = 0;
    phy_loss2 = 0;
    phy_loss3 = 0;
    phy_loss4 = 0;

    # l1 norm of weights
    reg_loss1 = None
    for param in model.parameters():
        if reg_loss1 is None:
            reg_loss1 = param.norm(1)
        else:
            reg_loss1 = reg_loss1+param.norm(1)
    # l2 norm of weights
    reg_loss2 = None
    for param in model.parameters():
        if reg_loss2 is None:
            reg_loss2 = param.norm(2)
        else:
            reg_loss2 = reg_loss2+param.norm(2)
    # total loss
    loss =  gamma*data_loss\
    +lmbd1*reg_loss1+lmbd2*reg_loss2\
    +alpha1*phy_loss1+alpha2*phy_loss2+alpha3*phy_loss3+alpha4*phy_loss4

    return loss,data_loss,phy_loss1,phy_loss2,phy_loss3,phy_loss4

def train_model(model, patience, n_epochs, ifEarlyS):

    if ifEarlyS==1:
        early_stopping = EarlyStopping(patience=patience, verbose=False) # , delta=1E-6
        print('Yes')

    for epoch in range(1, n_epochs + 1):
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        # batch processing
        permutation = torch.randperm(XTrn.size()[0])
        for i in range(0,XTrn.size()[0], batch_size):
            indI = permutation[i:i+batch_size]
            xi, yi, ti, ui = XTrn[indI], YTrn[indI], TTrn[indI], UTrn[indI]
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            U_pred = model(xi,yi,ti)
            # calculate the loss
            loss,_,_,_,_,_ = lossCal(U_pred,ui)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())
        # print loss value
        if epoch%1000 == 0:
            print('loss:',epoch,loss.item())

        if ifEarlyS==1:
            ###################
            # validate the model #
            ###################
            model.eval() # prep model for evaluation
            permutation = torch.randperm(XVal.size()[0])
            for i in range(0,XVal.size()[0], batch_size):
                indI = permutation[i:i+batch_size]
                xi, yi, ti, ui = XVal[indI], YVal[indI], TVal[indI], UVal[indI]
                U_pred_Val = model(xi,yi,ti)
                # calculate the loss
                _,data_loss,_,_,_,_ = lossCal(U_pred_Val,ui)
                # record validation loss
                valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        epoch_len = len(str(n_epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')
        print(print_msg)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        if ifEarlyS==1:
            early_stopping(valid_loss, model)
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
    
    if ifEarlyS==1:
        # load the last checkpoint with the best model
        model.load_state_dict(torch.load('checkpoint.pt'))


    return model

# D_in is input dimension;
# H is the dimension of the hidden layers
# D_out is output dimension.
D_in, H, D_out =  3, [50, 100, 500, 100, 50], 1

# Construct our model by instantiating the class defined above
model = MyNet1(D_in, H, D_out)
# Define loss function and optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

patience = 500
n_epochs = 20000
ifEarlyS = 1    # set it to zero if with only physics loss
model = train_model(model,patience,n_epochs,ifEarlyS)


# =============================================================================
# # generate extended dataset
# xExt, tExt = np.meshgrid(
#     np.linspace(-1, 1, 2560), 
#     np.linspace(0, 1, 1000)
# )
# xExt = np.reshape(xExt,(-1,1))
# tExt = np.reshape(tExt,(-1,1))
# XExt = Variable(torch.from_numpy(xExt).float(),requires_grad=True)
# TExt = Variable(torch.from_numpy(tExt).float(),requires_grad=True)
# =============================================================================

model.eval()
U_pred = model(X,Y,T)
u_pred = U_pred.detach().numpy()

sio.savemat('Burgers2DNN_N40.mat',{'u_pred':u_pred})

fig = plt.figure(figsize=(5,4))
plt.plot(u,'g-', linewidth=1)
plt.plot(u_pred,'r--', linewidth=2)
plt.show() 