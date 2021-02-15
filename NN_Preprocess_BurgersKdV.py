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



# Data = sio.loadmat('dataBurgersN100.mat')
Data = sio.loadmat('dataKdVN20.mat')
x = Data["x"] 
t = Data["t"] 
u = Data["u"] 
xTrn = Data["xTrn"] 
xVal = Data["xVal"]
tTrn = Data["tTrn"] 
tVal = Data["tVal"]  
uTrn = Data["uTrn"] 
uVal = Data["uVal"] 
# coefficients of PDE
# Data = sio.loadmat('c_pred.mat')
# c_pred = Data["c_pred"] 
# Convert numpy arrays to torch Variables
X = Variable(torch.from_numpy(x).float(),requires_grad=True)
T = Variable(torch.from_numpy(t).float(),requires_grad=True)
U = Variable(torch.from_numpy(u).float(),requires_grad=True)
XTrn = Variable(torch.from_numpy(xTrn).float(),requires_grad=True)
TTrn = Variable(torch.from_numpy(tTrn).float(),requires_grad=True)
UTrn = Variable(torch.from_numpy(uTrn).float(),requires_grad=True)
XVal = Variable(torch.from_numpy(xVal).float(),requires_grad=True)
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
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        f = nn.Tanh()(self.fc1(xt))
        f = nn.Tanh()(self.fc2(f))
        f = nn.Tanh()(self.fc3(f))
        f = self.fc4(f)
        return f

def lossCal(datax,datat,output,target):
    ## calculate the loss
    # data loss on training data
    data_loss = loss_fn(output,target)
    phy_loss1 = 0;
    phy_loss2 = 0;
    phy_loss3 = 0;
    phy_loss4 = 0;

    if sumAlpha>0:
        # phy_loss1: PDE loss
        Y_x = torch.autograd.grad(outputs=output.sum(), inputs=datax, create_graph=True)[0]
        Y_t = torch.autograd.grad(outputs=output.sum(), inputs=datat, create_graph=True)[0]
        Y_xx = torch.autograd.grad(outputs=Y_x.sum(), inputs=datax, create_graph=True)[0]
        Y_xxx = torch.autograd.grad(outputs=Y_xx.sum(), inputs=datax, create_graph=True)[0]
        Y = output; Y2 = Y*Y; Y3 = Y*Y2; 
        YY_x = Y*Y_x; Y2Y_x = Y2*Y_x; Y3Y_x = Y3*Y_x;
        YY_xx = Y*Y_xx; Y2Y_xx = Y2*Y_xx; Y3Y_xx = Y3*Y_xx;
        YY_xxx = Y*Y_xxx; Y2Y_xxx = Y2*Y_xxx; Y3Y_xxx = Y3*Y_xxx; 

        PDE = Y_t-c_pred[0,0]*torch.ones_like(Y_t)-c_pred[0,1]*Y-c_pred[0,2]*Y2-c_pred[0,3]*Y3   \
                    -c_pred[0,4]*Y_x-c_pred[0,5]*YY_x-c_pred[0,6]*Y2Y_x-c_pred[0,7]*Y3Y_x        \
                    -c_pred[0,8]*Y_xx-c_pred[0,9]*YY_xx-c_pred[0,10]*Y2Y_xx-c_pred[0,11]*Y3Y_xx        \
                    -c_pred[0,12]*Y_xxx-c_pred[0,13]*YY_xxx-c_pred[0,14]*Y2Y_xxx-c_pred[0,15]*Y3Y_xxx   
        val_PDE = torch.zeros_like(PDE)
        phy_loss1 =  loss_fn(PDE,val_PDE)
        # phy_loss2: initial condition
        Y0 = model(datax,torch.zeros_like(datat))
        phy_loss2 =  loss_fn(Y0+torch.sin(np.pi*datax),torch.zeros_like(Y0))
        # phy_loss3: boundary condition
        Y_1 = model(-torch.ones_like(datax),datat)
        phy_loss3 =  loss_fn(Y_1,torch.zeros_like(Y_1))
        # phy_loss4: boundary condition
        Y1 = model(torch.ones_like(datax),datat)
        phy_loss4 =  loss_fn(Y1,torch.zeros_like(Y1))
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
            xi, ti, ui = XTrn[indI], TTrn[indI], UTrn[indI]
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            U_pred = model(xi,ti)
            # calculate the loss
            loss,_,_,_,_,_ = lossCal(xi,ti,U_pred,ui)
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
                xi, ti, ui = XVal[indI], TVal[indI], UVal[indI]
                U_pred_Val = model(xi,ti)
                # calculate the loss
                _,data_loss,_,_,_,_ = lossCal(xi,ti,U_pred_Val,ui)
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

def difCal(datax,datat,output):
    Y_x = torch.autograd.grad(outputs=output.sum(), inputs=datax, create_graph=True)[0]
    Y_t = torch.autograd.grad(outputs=output.sum(), inputs=datat, create_graph=True)[0]
    Y_xx = torch.autograd.grad(outputs=Y_x.sum(), inputs=datax, create_graph=True)[0]
    Y_xxx = torch.autograd.grad(outputs=Y_xx.sum(), inputs=datax, create_graph=True)[0]
    Y = output; Y2 = Y*Y; Y3 = Y*Y2; 
    YY_x = Y*Y_x; Y2Y_x = Y2*Y_x; Y3Y_x = Y3*Y_x;
    YY_xx = Y*Y_xx; Y2Y_xx = Y2*Y_xx; Y3Y_xx = Y3*Y_xx;
    YY_xxx = Y*Y_xxx; Y2Y_xxx = Y2*Y_xxx; Y3Y_xxx = Y3*Y_xxx; 
    THETA = torch.cat((torch.ones_like(Y), Y, Y2, Y3, Y_x, YY_x, Y2Y_x, Y3Y_x, Y_xx, YY_xx, Y2Y_xx, Y3Y_xx, Y_xxx, YY_xxx, Y2Y_xxx, Y3Y_xxx), 1)
    return Y_t, THETA

# D_in is input dimension;
# H is the dimension of the hidden layers
# D_out is output dimension.
D_in, H, D_out =  2, [50, 100, 500, 100, 50], 1

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
U_pred = model(X,T)
u_pred = U_pred.detach().numpy()
# Y_t, THETA = difCal(X,T,U_pred)
# theta = THETA.detach().numpy()
# Y_t = Y_t.detach().numpy()

# sio.savemat('BurgersNN_N100.mat',{'u_pred':u_pred})
sio.savemat('KdVNN_N20.mat',{'u_pred':u_pred})

fig = plt.figure(figsize=(5,4))
plt.plot(u,'g-', linewidth=1)
plt.plot(u_pred,'r--', linewidth=2)
plt.show() 