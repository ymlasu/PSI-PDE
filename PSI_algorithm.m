%% Data preparation: Burgers
clear;close all;clc
load('./Datasets/BurgersNN_N10.mat')
load('./Datasets/BurgersMATLAB_N0.mat')
un = reshape(u_pred,[],length(t));
un = un';
clear u u_pred
dt = t(2)-t(1);
dx = x(2)-x(1);
U_dxn=differfitx(un,[-15:1:15],dx,8,4);U_dtn=differfitx(un.',[-15:1:15],dt,8,2);
U=U_dxn(:,:,1);U2=U.^2;U3=U.^3;Ut1=U_dtn(:,:,2).';Ut2=U_dtn(:,:,3).';
Ux1=U_dxn(:,:,2);Ux2=U_dxn(:,:,3);Ux3=U_dxn(:,:,4);Ux4=U_dxn(:,:,5);
UUx1=U.*Ux1;U2Ux1=U.*U.*Ux1;U3Ux1=U.^3.*Ux1;
UUx2=U.*Ux2;U2Ux2=U.*U.*Ux2;U3Ux2=U.^3.*Ux2;
UUx3=U.*Ux3;U2Ux3=U.*U.*Ux3;U3Ux3=U.^3.*Ux3;
UUx4=U.*Ux4;U2Ux4=U.*U.*Ux4;U3Ux4=U.^3.*Ux4;
A=[f(U),f(U2),f(U3),f(Ux1),f(UUx1),f(U2Ux1),f(U3Ux1),f(Ux2),f(UUx2),f(U2Ux2),...
    f(U3Ux2),f(Ux3),f(UUx3),f(U2Ux3),f(U3Ux3)];
A=[ones(size(A,1),1),A];
b=f(Ut1);
A=[real(A);imag(A)];b=[real(b);imag(b)];
% save('./Datasets/BurgersMATLAB_N10_FFT.mat','A','b')
%% Data preparation: KdV 
clear;close all;clc
load('./Datasets/KdVNN_N20.mat')
load('./Datasets/KdV.mat')
t = tt;
un = reshape(u_pred,[],length(t));
un = un';
clear uu tt u_pred
dt = t(2)-t(1);
dx = x(2)-x(1);
U_dxn=differfitx(un,[-15:1:15],dx,6,4);U_dtn=differfitx(un.',[-15:1:15],dt,6,2);
U=U_dxn(:,:,1);U2=U.^2;U3=U.^3;Ut1=U_dtn(:,:,2).';Ut2=U_dtn(:,:,3).';
Ux1=U_dxn(:,:,2);Ux2=U_dxn(:,:,3);Ux3=U_dxn(:,:,4);Ux4=U_dxn(:,:,5);
UUx1=U.*Ux1;U2Ux1=U.*U.*Ux1;U3Ux1=U.^3.*Ux1;
UUx2=U.*Ux2;U2Ux2=U.*U.*Ux2;U3Ux2=U.^3.*Ux2;
UUx3=U.*Ux3;U2Ux3=U.*U.*Ux3;U3Ux3=U.^3.*Ux3;
UUx4=U.*Ux4;U2Ux4=U.*U.*Ux4;U3Ux4=U.^3.*Ux4;
A=[f(U),f(U2),f(U3),f(Ux1),f(UUx1),f(U2Ux1),f(U3Ux1),f(Ux2),f(UUx2),f(U2Ux2),...
    f(U3Ux2),f(Ux3),f(UUx3),f(U2Ux3),f(U3Ux3),f(Ux4),f(UUx4),f(U2Ux4),f(U3Ux4)];
A=[ones(size(A,1),1),A];
b=f(Ut1);
A=[real(A);imag(A)];b=[real(b);imag(b)];
% save('./Datasets/KdVMATLAB_N20_FFT.mat','A','b')
%% Data preparation: Burgers2D 
clear;close all;clc
load('./Datasets/Burgers2D.mat','t','x','y')
load('./Datasets/Burgers2DNN_N40.mat')

m1 = length(x);
m2 = length(y);
m3 = length(t);
u = zeros(m3,m2,m1);
ind = 0;
for i = 1:m3
    for j = 1:m2
        for k = 1:m1
            ind = ind+1;
            u(i,j,k) = u_pred(ind);
        end
    end
end

clearvars -except u t x y
dt = t(2)-t(1);
dx = x(2)-x(1);
dy = y(2)-y(1);
form1=-2:2;form2=-3:3;
un = u;
U=un;U2=U.*U;U3=U2.*U;
Ut1=differ3(un,form1,1,1)/dt;Ut2=differ3(un,form1,1,2)/dt^2;
Ux1=differ3(un,form1,3,1)/dx;Ux2=differ3(un,form1,3,2)/dx^2;Ux3=differ3(un,form2,3,3)/dx^3;
Uy1=differ3(un,form1,2,1)/dy;Uy2=differ3(un,form1,2,2)/dy^2;Uy3=differ3(un,form2,2,3)/dy^3;
UUx1=U.*Ux1;U2Ux1=U.*U.*Ux1;U3Ux1=U3.*Ux1;
UUx2=U.*Ux2;U2Ux2=U.*U.*Ux2;U3Ux2=U3.*Ux2;
UUx3=U.*Ux3;U2Ux3=U.*U.*Ux3;U3Ux3=U3.*Ux3;
UUy1=U.*Uy1;U2Uy1=U.*U.*Uy1;U3Uy1=U3.*Uy1;
UUy2=U.*Uy2;U2Uy2=U.*U.*Uy2;U3Uy2=U3.*Uy2;
UUy3=U.*Uy3;U2Uy3=U.*U.*Uy3;U3Uy3=U3.*Uy3;
A=[fn(U),fn(U2),fn(U3),fn(Ux1+Uy1),fn(UUx1+UUy1),fn(U2Ux1+U2Uy1),fn(U3Ux1+U3Uy1),...
    fn(Ux2+Uy2),fn(UUx2+UUy2),fn(U2Ux2+U2Uy2),fn(U3Ux2+U3Uy2),...
    fn(Ux3+Uy3),fn(UUx3+UUy3),fn(U2Ux3+U2Uy3),fn(U3Ux3+U3Uy3)];
A=[ones(size(A,1),1),A];
b=fn(Ut1);
A=[real(A);imag(A)];b=[real(b);imag(b)];
% save('./Datasets/Burgers2D_N40_FFT.mat','A','b')
%% Data preparation: Navier Stokes: lid-driven cavity
clear;close all;clc
load('./Datasets/NS_N0.mat','t','x','y')
load('./Datasets/v_NS_NN_N50.mat')
v_pred = double(u_pred);
clear u_pred;
load('./Datasets/p_NS_NN_N50.mat')
p_pred = double(u_pred);
clear u_pred;
load('./Datasets/u_NS_NN_N50.mat')
u_pred = double(u_pred);
t = t(1:100)';
x = x';
x = (x(2:end-2)+x(3:end-1))/2;
y = y';
y = (y(2:end-2)+y(3:end-1))/2;

m1 = length(x);
m2 = length(y);
m3 = length(t);
u = zeros(m3,m1,m2);
ind = 0;
for i = 1:m3
    for j = 1:m1
        for k = 1:m2
            ind = ind+1;
            u(i,j,k) = u_pred(ind);
            v(i,j,k) = v_pred(ind);
            p(i,j,k) = p_pred(ind);
        end
    end
end
clearvars -except u v p t x y
dt = t(2)-t(1);
dx = x(2)-x(1);
dy = y(2)-y(1);
u = permute(u,[1 3 2]);
v = permute(v,[1 3 2]);
p = permute(p,[1 3 2]);
form1=-2:2;form2=-3:3;
U=u;U2=U.*U;U3=U2.*U;
Ut1=differ3(u,form1,1,1)/dt;Ut2=differ3(u,form1,1,2)/dt^2;
V=v;V2=V.*V;V3=V2.*V;
Vt1=differ3(v,form1,1,1)/dt;Vt2=differ3(v,form1,1,2)/dt^2;
Px1=differ3(p,form1,3,1)/dx;Px2=differ3(p,form1,3,2)/dx^2;
Py1=differ3(p,form1,2,1)/dy;Py2=differ3(p,form1,2,2)/dy^2;
Ux1=differ3(u,form1,3,1)/dx;Ux2=differ3(u,form1,3,2)/dx^2;Ux3=differ3(u,form2,3,3)/dx^3;
Uy1=differ3(u,form1,2,1)/dy;Uy2=differ3(u,form1,2,2)/dy^2;Uy3=differ3(u,form2,2,3)/dy^3;
Vx1=differ3(v,form1,3,1)/dx;Vx2=differ3(v,form1,3,2)/dx^2;Vx3=differ3(v,form2,3,3)/dx^3;
Vy1=differ3(v,form1,2,1)/dy;Vy2=differ3(v,form1,2,2)/dy^2;Vy3=differ3(v,form2,2,3)/dy^3;
A=[fn(Px1),fn(Py1),fn(Ux1+Uy1),fn(Vx1+Vy1),fn(U.*Ux1+V.*Uy1),fn(U.*Vx1+V.*Vy1),fn(U2.*Ux1+V2.*Uy1),fn(U2.*Vx1+V2.*Vy1)...
    fn(Ux2+Uy2),fn(Vx2+Vy2),fn(U.*Ux2+V.*Uy2),fn(U.*Vx2+V.*Vy2),fn(U2.*Ux2+V2.*Uy2),fn(U2.*Vx2+V2.*Vy2)...
    fn(Ux3+Uy3),fn(Vx3+Vy3),fn(U.*Ux3+V.*Uy3),fn(U.*Vx3+V.*Vy3),fn(U2.*Ux3+V2.*Uy3),fn(U2.*Vx3+V2.*Vy3)];
b1=fn(Ut1);
b2=fn(Vt1);
A=[real(A);imag(A)];
b1=[real(b1);imag(b1)];
b2=[real(b2);imag(b2)];
% save('./Datasets/NS_N50_FFT.mat','A','b1','b2')
%% Aanalysis via train-validation
clear;close all;clc
load('./Datasets/BurgersMATLAB_N0_FFT.mat')
% load('./Datasets/KdVMATLAB_N0_FFT.mat')
% load('./Datasets/Burgers2D_N0_FFT.mat')
% load('./Datasets/NS_N0_FFT.mat');b = b1;

% normlize the data
normA = vecnorm(A,2);
A = A./normA;
normb = vecnorm(b,2);
b = b./normb;
% find the most contributive terms by cross validation
valR = 0.2;             % ratio of validation data
numVals = 10000;          % number of cross validations  
indL = 1:size(A,2);  	% initial indices
errRec = zeros(numVals,size(A,2)+1);
errBIC = zeros(numVals,size(A,2)+1);
for i = 1:numVals
    rng(i); % assign rand seeds for this validation
    % separate data into training and validation sets
    valInd =  randperm(length(b),round(valR*length(b)));
    valInd = sort(valInd);
    trnInd = setdiff(1:length(b),valInd);
    ATrn = A(trnInd,:);
    AVal = A(valInd,:);
    bTrn = b(trnInd,:);
    bVal = b(valInd,:);
    % find the one most contributive term
    for j = 1:size(A,2)
        % exclude index j from the list
        indL1 = setdiff(indL,j);
        % add index j to the list
%         indL1 = unique([ j]);
        % least square solution of Ax = b
        A1 = ATrn(:,indL1);
        x1 = A1\bTrn;
        A2 = AVal(:,indL1);
        b2 = A2*x1;
        errRec(i,j) = rms(b2-bVal);
        mseIJ = immse(b2,bVal);
        errBIC(i,j) = length(bTrn)*log(mseIJ)+(sum(indL1.^2)+length(indL1)^2)/length(indL1)*log(length(bTrn));   % modified BIC
    end
    % reference regression error
    errRec(i,end) = rms(bVal);
    bTemp = bVal;
    bTemp(:) = 0;
    indTemp = (1:size(A,2));
    % reference BIC
    errBIC(i,end) = length(bTrn)*log(immse(bTemp,bVal))+(sum(indTemp.^2)+length(indTemp)^2)/length(indTemp)*log(length(bTrn));   % modified BIC
end
%% plot error distributions: Burgers
txts = {'const 1','$u$','$u^2$','$u^3$','$u_x$','$uu_x$','$u^2u_x$','$u^3u_x$',...
    '$u_{xx}$','$uu_{xx}$','$u^2u_{xx}$','$u^3u_{xx}$',...
    '$u_{xxx}$','$uu_{xxx}$','$u^2u_{xxx}$','$u^3u_{xxx}$'};
clc;close all
ff = fig('units','inches','width',5,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(4,4,[.04 .04],[.08 .015],[.02 .03]);
for i=1:16
axes(h(i));box on;
histogram(errRec(:,i)*1000,10)
set(gca,'yticklabel',[])
xl = xlim; yl = ylim;
tx = 0.5*(xl(2)-xl(1))+xl(1);
ty = 0.8*(yl(2)-yl(1))+yl(1);
if ismember(i,[ ])
    text(tx,ty, txts(i),'Interpreter','latex','Color','red')
else
    text(tx,ty, txts(i),'Interpreter','latex','Color','black')
end
end
tx = -1.4*(xl(2)-xl(1))+xl(1);
ty = -0.3*(yl(2)-yl(1))+yl(1);
text(tx,ty,'$\varepsilon_\mathrm{Reg}, 10^{-3}$','Interpreter','latex')
% saveas(ff,'Figures\errReg_burgersN0_3.png')
% saveas(ff,'Figures\errReg_burgersN0_3.eps','epsc') 
% savefig(ff,'Figures\errReg_burgersN0_3.fig')
%% plot error BIC distributions: Burgers
txts = {'const 1','$u$','$u^2$','$u^3$','$u_x$','$uu_x$','$u^2u_x$','$u^3u_x$',...
    '$u_{xx}$','$uu_{xx}$','$u^2u_{xx}$','$u^3u_{xx}$',...
    '$u_{xxx}$','$uu_{xxx}$','$u^2u_{xxx}$','$u^3u_{xxx}$'};
clc;close all
ff = fig('units','inches','width',5,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(4,4,[.04 .04],[.08 .015],[.02 .03]);
for i=1:16
axes(h(i));box on;
histogram(errBIC(:,i)/1000,10)
set(gca,'yticklabel',[])
xl = xlim; yl = ylim;
tx = 0.5*(xl(2)-xl(1))+xl(1);
ty = 0.8*(yl(2)-yl(1))+yl(1);
if ismember(i,[])
    text(tx,ty, txts(i),'Interpreter','latex','Color','red')
else
    text(tx,ty, txts(i),'Interpreter','latex','Color','black')
end
end
tx = -1.4*(xl(2)-xl(1))+xl(1);
ty = -0.3*(yl(2)-yl(1))+yl(1);
text(tx,ty,'$\varepsilon_\mathrm{BIC}, 10^{3}$','Interpreter','latex')
% saveas(ff,'Figures\errBIC_burgersN0_3.png')
% saveas(ff,'Figures\errBIC_burgersN0_3.eps','epsc') 
% savefig(ff,'Figures\errBIC_burgersN0_3.fig')
%% Plot mean error: Burgers
clc;close all;
errAve = sum(errRec)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.18 .04],[.12 .03]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12,0.06, '$\varepsilon_\mathrm{Reg}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylabel('$\bar{\varepsilon}_\mathrm{Reg}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,-0.002, txts(i),'Interpreter','latex');
set(tt,'Rotation',-45);
if ismember(i,[])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errRegAve_burgersN0_3.png')
% saveas(ff,'Figures\errRegAve_burgersN0_3.eps','epsc') 
% savefig(ff,'Figures\errRegAve_burgersN0_3.fig')
%% Plot mean error BIC: Burgers
clc;close all;
errAve = sum(errBIC)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.04 .18],[.125 .03]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12,errAve(end)*1.2, '$\varepsilon_\mathrm{BIC}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylabel('$\bar{\varepsilon}_\mathrm{BIC}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,50, txts(i),'Interpreter','latex');
set(tt,'Rotation',45);
if ismember(i,[ ])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errBICAve_burgersN0_3.png')
% saveas(ff,'Figures\errBICAve_burgersN0_3.eps','epsc') 
% savefig(ff,'Figures\errBICAve_burgersN0_3.fig')
%% plot error distributions: KdV
txts = {'const 1','$u$','$u^2$','$u^3$','$u_x$','$uu_x$','$u^2u_x$','$u^3u_x$',...
    '$u_{xx}$','$uu_{xx}$','$u^2u_{xx}$','$u^3u_{xx}$',...
    '$u_{xxx}$','$uu_{xxx}$','$u^2u_{xxx}$','$u^3u_{xxx}$',...
    '$u_{xxxx}$','$uu_{xxxx}$','$u^2u_{xxxx}$','$u^3u_{xxxx}$'};
clc;close all
ff = fig('units','inches','width',5,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(4,5,[.04 .04],[.08 .015],[.02 .03]);
for i=1:size(errRec,2)-1
axes(h(i));box on;
histogram(errRec(:,i)*1000,10)
set(gca,'yticklabel',[])
% title(strcat('(',char(96+i),')',{''}),'FontWeight','Normal')
xl = xlim; yl = ylim;
tx = 0.55*(xl(2)-xl(1))+xl(1);
ty = 0.8*(yl(2)-yl(1))+yl(1);
if ismember(i,[ ])
    text(tx,ty, txts(i),'Interpreter','latex','Color','red')
else
    text(tx,ty, txts(i),'Interpreter','latex','Color','black')
end
end
tx = -2.4*(xl(2)-xl(1))+xl(1);
ty = -0.3*(yl(2)-yl(1))+yl(1);
text(tx,ty,'$\varepsilon_\mathrm{Reg}, 10^{-3}$','Interpreter','latex')
% saveas(ff,'Figures\errReg_KdVN0_3.png')
% saveas(ff,'Figures\errReg_KdVN0_3.eps','epsc') 
% savefig(ff,'Figures\errReg_KdVN0_3.fig')
%% plot error BIC distributions: KdV
txts = {'const 1','$u$','$u^2$','$u^3$','$u_x$','$uu_x$','$u^2u_x$','$u^3u_x$',...
    '$u_{xx}$','$uu_{xx}$','$u^2u_{xx}$','$u^3u_{xx}$',...
    '$u_{xxx}$','$uu_{xxx}$','$u^2u_{xxx}$','$u^3u_{xxx}$',...
    '$u_{xxxx}$','$uu_{xxxx}$','$u^2u_{xxxx}$','$u^3u_{xxxx}$'};
clc;close all
ff = fig('units','inches','width',5,'height',5,'font','Times New Roman','fontsize',11);
h = tight_subplot(4,5,[.04 .04],[.08 .015],[.02 .03]);
for i=1:size(errBIC,2)-1
axes(h(i));box on;
histogram(errBIC(:,i)*1E-3,10)
set(gca,'yticklabel',[])
% title(strcat('(',char(96+i),')',{''}),'FontWeight','Normal')
xl = xlim; yl = ylim;
tx = 0.02*(xl(2)-xl(1))+xl(1);
ty = 0.8*(yl(2)-yl(1))+yl(1);
if ismember(i,[ ])
    text(tx,ty, txts(i),'Interpreter','latex','Color','red')
else
    text(tx,ty, txts(i),'Interpreter','latex','Color','black')
end
end
tx = -2.4*(xl(2)-xl(1))+xl(1);
ty = -0.3*(yl(2)-yl(1))+yl(1);
text(tx,ty,'$\varepsilon_\mathrm{BIC}, 10^{3}$','Interpreter','latex')
% saveas(ff,'Figures\errBIC_KdVN0_3.png')
% saveas(ff,'Figures\errBIC_KdVN0_3.eps','epsc') 
% savefig(ff,'Figures\errBIC_KdVN0_3.fig')
%% Plot mean error: KdV
clc;close all;
errAve = sum(errRec)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.18 .04],[.12 .03]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(15,0.075, '$\varepsilon_\mathrm{Reg}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylabel('$\bar{\varepsilon}_\mathrm{Reg}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,-0.002, txts(i),'Interpreter','latex');
set(tt,'Rotation',-45);
if ismember(i,[])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errRegAve_kdvN0_3.png')
% saveas(ff,'Figures\errRegAve_kdvN0_3.eps','epsc') 
% savefig(ff,'Figures\errRegAve_kdvN0_3.fig')
%% Plot mean error BIC: KdV
clc;close all;
errAve = sum(errBIC)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.04 .18],[.125 .03]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12.5,errAve(end)*3, '$\varepsilon_\mathrm{BIC}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylabel('$\bar{\varepsilon}_\mathrm{BIC}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,50, txts(i),'Interpreter','latex');
set(tt,'Rotation',45);
if ismember(i,[])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errBICAve_kdvN0_3.png')
% saveas(ff,'Figures\errBICAve_kdvN0_3.eps','epsc') 
% savefig(ff,'Figures\errBICAve_kdvN0_3.fig')
%% Plot mean error: Burgers2D
clc;close all;
txts = {'const 1','$u$','$u^2$','$u^3$',...
    '$u_x+u_y$','$u(u_x+u_y)$','$u^2(u_x+u_y)$','$u^3(u_x+u_y)$',...
    '$u_{xx}+u_{yy}$','$u(u_{xx}+u_{yy})$','$u^2(u_{xx}+u_{yy})$','$u^3(u_{xx}+u_{yy})$',...
    '$u_{xxx}+u_{yyy}$','$u(u_{xxx}+u_{yyy})$','$u^2(u_{xxx}+u_{yyy})$','$u^3(u_{xxx}+u_{yyy})$'};
errAve = sum(errRec)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.30 .04],[.12 .10]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12,0.055, '$\varepsilon_\mathrm{Reg}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylim([0 0.08])
ylabel('$\bar{\varepsilon}_\mathrm{Reg}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,-0.002, txts(i),'Interpreter','latex');
set(tt,'Rotation',-45);
if ismember(i,[])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errRegAve_bugers2DN0_3.png')
% saveas(ff,'Figures\errRegAve_bugers2DN0_3.eps','epsc') 
% savefig(ff,'Figures\errRegAve_bugers2DN0_3.fig')
%% Plot mean error BIC: Burgers2D
clc;close all;
errAve = sum(errBIC)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.04 .28],[.125 .12]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12.5,errAve(end)*1.2, '$\varepsilon_\mathrm{BIC}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylabel('$\bar{\varepsilon}_\mathrm{BIC}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,50, txts(i),'Interpreter','latex');
set(tt,'Rotation',45);
if ismember(i,[])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errBICAve_bugers2DN0_3.png')
% saveas(ff,'Figures\errBICAve_bugers2DN0_3.eps','epsc') 
% savefig(ff,'Figures\errBICAve_bugers2DN0_3.fig')
%% Plot mean error: NS
clc;close all;
txts = {'$p_x$','$p_y$','$u_x+u_y$','$v_x+v_y$','$uu_x+vu_y$','$uv_x+vv_y$','$u^2u_x+v^2u_y$','$u^2v_x+v^2v_y$',...
    '$u_{xx}+u_{yy}$','$v_{xx}+v_{yy}$','$uu_{xx}+vu_{yy}$','$uv_{xx}+vv_{yy}$','$u^2u_{xx}+v^2u_{yy}$','$u^2v_{xx}+v^2v_{yy}$',...
    '$u_{xxx}+u_{yyy}$','$v_{xxx}+v_{yyy}$','$uu_{xxx}+vu_{yyy}$','$uv_{xxx}+vv_{yyy}$','$u^2u_{xxx}+v^2u_{yyy}$','$u^2v_{xxx}+v^2v_{yyy}$'};
errAve = sum(errRec)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.30 .04],[.12 .10]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12,0.067, '$\varepsilon_\mathrm{Reg}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylim([0 0.08])
ylabel('$\bar{\varepsilon}_\mathrm{Reg}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,-0.002, txts(i),'Interpreter','latex');
set(tt,'Rotation',-45);
if ismember(i,[2 6 10])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errRegAve_NS_N0_4.png')
% saveas(ff,'Figures\errRegAve_NS_N0_4.eps','epsc') 
% savefig(ff,'Figures\errRegAve_NS_N0_4.fig')
%% Plot mean error BIC: NS
clc;close all;
errAve = sum(errBIC)/numVals;
ff = fig('units','inches','width',5,'height',3.0,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,1,[.04 .04],[.04 .28],[.125 .12]);
axes(h(1));box on;
bar(errAve(1:end-1))
yline(errAve(end),'-.r','LineWidth',1.5);
text(12.5,errAve(end)*1.6, '$\varepsilon_\mathrm{BIC}$ reference','Interpreter','latex');
set(gca,'xticklabel',[])
set(gca,'xtick',1:16)
ylabel('$\bar{\varepsilon}_\mathrm{BIC}$','Interpreter','latex')
for i=1:length(txts)
tt = text(i,50, txts(i),'Interpreter','latex');
set(tt,'Rotation',45);
if ismember(i,[])
    set(tt,'Color','red');
end
end
% saveas(ff,'Figures\errBICAve_NS_N0_4.png')
% saveas(ff,'Figures\errBICAve_NS_N0_4.eps','epsc') 
% savefig(ff,'Figures\errBICAve_NS_N0_4.fig')

%% functions FFT
function r=f(u)
    u_tem=u(20:end-20,20:end-20);
    u_f=fft2(u_tem);
    indy=1:10;indx=1:10;NA=length(indx)*length(indy);
    r=reshape(u_f(indx,indy),NA,1);
end

function r=fn(u)
    u_tem=u(20:end-20,20:end-20,20:end-20);
    u_f=fftn(u_tem);
    indt=1:5;indx=1:5;indy=indx;NA=length(indt)*length(indx)*length(indy);
    r=reshape(u_f(indt,indx,indy),NA,1);   
end