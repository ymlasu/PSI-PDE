%% Burgers' Equation: u_t + u*u_x - 0.01/pi*u_xx = 0
clear;close all;clc
load('./Datasets/BurgersMATLAB_N0.mat')
rng(1)
noiseL = 1.00;
m = length(x);
n = length(t);
x = repmat(x,n,1);
t = repmat(t,1,m);
t = reshape(t',[],1);
u = reshape(u,[],1);
u = u+noiseL*std(u,0,'all')*randn(size(u));
valR = 0.2; % ratio of validation data
valInd =  randperm(length(x),round(valR*length(x)));
trnInd = setdiff(1:length(x),valInd);
xTrn = x(trnInd);
xVal = x(valInd);
tTrn = t(trnInd);
tVal = t(valInd);
uTrn = u(trnInd);
uVal = u(valInd);
% save('dataBurgersN100.mat','x','t','u','xTrn','xVal','tTrn','tVal','uTrn','uVal')
%% KdV equation: u_t + u*u_x + 0.0025u_xxx = 0
clear;close all;clc
load('./Datasets/KdV.mat')
t = tt';
x = x';
u = uu;
clear tt uu
rng(1)
noiseL = 1.00;
m = length(x);
n = length(t);
x = repmat(x,n,1);
t = repmat(t,1,m);
t = reshape(t',[],1);
u = reshape(u,[],1);
u = u+noiseL*std(u,0,'all')*randn(size(u));
valR = 0.2; % ratio of validation data
valInd =  randperm(length(x),round(valR*length(x)));
trnInd = setdiff(1:length(x),valInd);
xTrn = x(trnInd);
xVal = x(valInd);
tTrn = t(trnInd);
tVal = t(valInd);
uTrn = u(trnInd);
uVal = u(valInd);
save('dataKdVN100.mat','x','t','u','xTrn','xVal','tTrn','tVal','uTrn','uVal')
%% Burgers' 2D Equation: ut = -uux+0.01uxx -uuy+0.01uyy
clear;close all;clc
load('./Datasets/Burgers2D.mat')
t = t';
x = x';
y = y';
rng(1)
noiseL = 0.40;
m1 = length(x);
m2 = length(y);
m3 = length(t);
tt = zeros(m1*m2*m3,1);
xx = tt; yy = tt; uu = tt;
ind = 0;
for i = 1:m3
    for j = 1:m2
        for k = 1:m1
            ind = ind+1;
            tt(ind) = t(i);
            yy(ind) = y(j);
            xx(ind) = x(k);
            uu(ind) = u(i,j,k);
        end
    end
end
t = tt; x = xx; y = yy; u = uu;            
clear tt xx yy uu
u = u+noiseL*std(u,0,'all')*randn(size(u));
valR = 0.2; % ratio of validation data
valInd =  randperm(length(x),round(valR*length(x)));
trnInd = setdiff(1:length(x),valInd);
xTrn = x(trnInd);
xVal = x(valInd);
yTrn = y(trnInd);
yVal = y(valInd);
tTrn = t(trnInd);
tVal = t(valInd);
uTrn = u(trnInd);
uVal = u(valInd);
% save('dataBurgers2DN40.mat','x','y','t','u','xTrn','xVal','yTrn','yVal','tTrn','tVal','uTrn','uVal')
%% NS Equation: lid-driven cavity
clear;close all;clc
load('./Datasets/NS_N0.mat')
u = (u(1:100,1:end-1,2:end-1)+u(1:100,2:end,2:end-1))/2;
v = (v(1:100,2:end-1,1:end-1)+v(1:100,2:end-1,2:end))/2;
p = p(1:100,2:end-1,2:end-1);
t = t(1:100)';
x = x';
x = (x(2:end-2)+x(3:end-1))/2;
y = y';
y = (y(2:end-2)+y(3:end-1))/2;
rng(1)
noiseL = 0.00;
m1 = length(x);
m2 = length(y);
m3 = length(t);
tt = zeros(m1*m2*m3,1);
xx = tt; yy = tt; 
uu = tt; vv = tt; pp = tt;
ind = 0;
for i = 1:m3
    for j = 1:m1
        for k = 1:m2
            ind = ind+1;
            tt(ind) = t(i);
            xx(ind) = x(j);
            yy(ind) = y(k);
            uu(ind) = u(i,j,k);
            vv(ind) = v(i,j,k);
            pp(ind) = p(i,j,k);
        end
    end
end
t = tt; x = xx; y = yy; 
u = uu; v = vv; p = pp;           
clear tt xx yy uu vv pp
u = u+noiseL*std(u,0,'all')*randn(size(u));
v = v+noiseL*std(v,0,'all')*randn(size(v));
p = p+noiseL*std(p,0,'all')*randn(size(p));
valR = 0.2; % ratio of validation data
valInd =  randperm(length(x),round(valR*length(x)));
trnInd = setdiff(1:length(x),valInd);
xTrn = x(trnInd);
xVal = x(valInd);
yTrn = y(trnInd);
yVal = y(valInd);
tTrn = t(trnInd);
tVal = t(valInd);
uTrn = u(trnInd);
uVal = u(valInd);
vTrn = v(trnInd);
vVal = v(valInd);
pTrn = p(trnInd);
pVal = p(valInd);
% save('dataNSN50.mat','x','y','t','u','v','p','xTrn','xVal','yTrn','yVal','tTrn','tVal','uTrn','uVal','vTrn','vVal','pTrn','pVal')