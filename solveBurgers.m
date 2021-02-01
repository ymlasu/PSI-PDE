%% Optimize coefficients of Burgers uux & uxx
clear;close all;clc
clearvars -global
global a1 a2 t x u0
load('Datasets\BurgersMATLAB_N0.mat')
rng(1)
noiseL = 0.50;
un = u+noiseL*std(u,0,'all')*randn(size(u));
% u0 = u';clear u
u0 = un';clear un
rng default
% case N0
% a10 = 0.0075;
% a20 = -0.9886;
% case N10
% a10 = 0.0075;
% a20 = -1.0293;
% case N20
% a10 = 0.0057;
% a20 = -0.9853;
% case N50
a10 = 0.0076;
a20 = -0.9758;
a0 = [a10 a20];

da0 = [1E-4 1E-5];
da = da0;
errTol = 1E-6;
dTol = 1E-10;
badTol = 20;
maxIters = 1E5;
err = 1;
a = a0;
g = [0 0];
numIters = 0;
numBad = 0;
errMin = err;
while (numIters<maxIters && min(abs(da./da0)>dTol*abs(da0)) && err>errTol)
    f0 = errBurgers(a);
    g(1) = (errBurgers([a(1)+da0(1) a(2)]) - f0)/da0(1);
    g(2) = (errBurgers([a(1) a(2)+da0(2)]) - f0)/da0(2);
    da = -da0.*g;
    a = a+da;
    numIters = numIters+1;
    err = errBurgers(a)
    if (err < errMin)
        aOpt = a;
        errMin = err;
    else
        numBad = numBad+1;
    end
    if numBad>badTol
        break;
    end
end
%% Plot solutions before & after optimization
m = 0;
a1 = aOpt(1);
a2 = aOpt(2);
err1 = errBurgers(aOpt);
uOpt = pdepe(m,@pdeBurgersOPT,@pdex1ic,@pdex1bc,x,t);
close all;ff = figure;subplot(121)
surf(x,t,u0, 'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','g')
hold on
surf(x,t,uOpt, 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
zlabel('$u$','Interpreter','latex')
set(gca, 'FontName', 'Times New Roman')
legend({'$u_t = -uu_x+\frac{0.01}{\pi}u_{xx}$+50\%noise',...
    '$u_t = -0.9758uu_x + \frac{0.0098}{\pi}u_{xx}$'},'Interpreter','latex')
zlim([-1.3 1.3])
v = [-10.6109 4.1246];view(v)
subplot(122)
surf(x,t,uOpt-u0, 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
zlabel('$\delta_u$','Interpreter','latex')
set(gca, 'FontName', 'Times New Roman')
v = [-10.6109 4.1246];view(v)
% saveas(ff,'Figures\solBurgersN50_1.png')
% saveas(ff,'Figures\solBurgersN50_1.eps','epsc') 
% savefig(ff,'Figures\solBurgersN50_1.fig')
%% Optimize coefficients of Burgers uux, & u2uxx
clear;close all;clc
clearvars -global
global b2 b3 t x u0
load('Datasets\BurgersMATLAB_N0.mat')
u0 = u';clear u
rng default
b20 = 0.0075;
b30 = -0.9886;
b0 = [b20 b30];

db0 = [1E-4 1E-2];
db = db0;
errTol = 1E-6;
dTol = 1E-10;
badTol = 20;
maxIters = 1E5;
err = 1;
b = b0;
g = [0 0];
numIters = 0;
numBad = 0;
errMin = err;
while (numIters<maxIters && min(abs(db./db0)>dTol*abs(db0)) && err>errTol)
    f0 = errBurgers2(b);
    g(1) = (errBurgers2([b(1)+db0(1) b(2)]) - f0)/db0(1);
    g(2) = (errBurgers2([b(1) b(2)+db0(2)]) - f0)/db0(2);
    db = -db0.*g;
    b = b+db;
    numIters = numIters+1;
    db(2)
    err = errBurgers2(b)
    if (err < errMin)
        bOpt = b;
        errMin = err;
    else
        numBad = numBad+1;
    end
    if numBad>badTol
        break;
    end
end
%% Plot solutions before & after optimization
m = 0;
b2 = bOpt(1);
b3 = bOpt(2);
err2 = errBurgers2(bOpt);
uOpt = pdepe(m,@pdeBurgersOPT2,@pdex1ic,@pdex1bc,x,t);
close all;ff = figure;
subplot(121)
surf(x,t,u0, 'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','g')
hold on
surf(x,t,uOpt, 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
zlabel('$u$','Interpreter','latex')
set(gca, 'FontName', 'Times New Roman')
legend({'$u_t = -uu_x+\frac{0.01}{\pi}u_{xx}$',...
    '$u_t = -1.0029uu_x + 0.0092u^2u_{xx}$'},'Interpreter','latex')
zlim([-1 1])
v = [-10.6109 4.1246];view(v)
subplot(122)
surf(x,t,uOpt-u0, 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
zlabel('$\delta_u$','Interpreter','latex')
set(gca, 'FontName', 'Times New Roman')
v = [-10.6109 4.1246];view(v)
% saveas(ff,'Figures\solBurgersN0_2.png')
% saveas(ff,'Figures\solBurgersN0_2.eps','epsc') 
% savefig(ff,'Figures\solBurgersN0_2.fig')
%% Optimize coefficients of Burgers uux, uxx, & u2uxx
clear;close all;clc
clearvars -global
global b1 b2 b3 t x u0
load('Datasets\BurgersMATLAB_N0.mat')
u0 = u';clear u
rng default
b10 = 0.0086/pi;
b20 = 0.0047;
b30 = -1.0101;
b0 = [b10 b20 b30];

db0 = [1E-4 1E-4 1E-2];
db = db0;
errTol = 1E-6;
dTol = 1E-10;
badTol = 20;
maxIters = 1E5;
err = 1;
b = b0;
g = [0 0 0];
numIters = 0;
numBad = 0;
errMin = err;
while (numIters<maxIters && min(abs(db./db0)>dTol*abs(db0)) && err>errTol)
    f0 = errBurgers3(b);
    g(1) = (errBurgers3([b(1)+db0(1) b(2) b(3)]) - f0)/db0(1);
    g(2) = (errBurgers3([b(1) b(2)+db0(2) b(3)]) - f0)/db0(2);
    g(3) = (errBurgers3([b(1) b(2) b(3)+db0(3)]) - f0)/db0(3);
    db = -db0.*g;
    b = b+db;
    numIters = numIters+1;
    db(2)
    err = errBurgers3(b)
    if (err < errMin)
        bOpt = b;
        errMin = err;
    else
        numBad = numBad+1;
    end
    if numBad>badTol
        break;
    end
end
%% Plot solutions before & after optimization
m = 0;
b1 = bOpt(1);
b2 = bOpt(2);
b3 = bOpt(3);
uOpt = pdepe(m,@pdeBurgersOPT3,@pdex1ic,@pdex1bc,x,t);
close all;ff = figure;
subplot(121)
surf(x,t,u0, 'FaceColor','g', 'FaceAlpha',0.5, 'EdgeColor','g')
hold on
surf(x,t,uOpt, 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
zlabel('$u$','Interpreter','latex')
set(gca, 'FontName', 'Times New Roman')
legend({'$u_t = -uu_x+\frac{0.01}{\pi}u_{xx}$',...
    '$u_t = -1.0053uu_x + \frac{0.009}{\pi}u_{xx} + 0.00004u^2u_{xx}$'},'Interpreter','latex')
zlim([-1 1])
v = [-10.6109 4.1246];view(v)
subplot(122)
surf(x,t,uOpt-u0, 'FaceColor','b', 'FaceAlpha',0.5, 'EdgeColor','b')
xlabel('$x$','Interpreter','latex')
ylabel('$t$','Interpreter','latex')
zlabel('$\delta_u$','Interpreter','latex')
set(gca, 'FontName', 'Times New Roman')
v = [-10.6109 4.1246];view(v)
% saveas(ff,'Figures\solBurgersN0_4.png')
% saveas(ff,'Figures\solBurgersN0_4.eps','epsc') 
% savefig(ff,'Figures\solBurgersN0_4.fig')
%% Functions
function err = errBurgers(a)
m = 0;
global a1 a2 t x u0
a1 = a(1);
a2 = a(2);
u1 = pdepe(m,@pdeBurgersOPT,@pdex1ic,@pdex1bc,x,t);
u1 = u1(:,:,1);
err = rms(reshape(u1-u0,[],1));
end
%----------------------------------------------
function err = errBurgers2(b)
m = 0;
global b2 b3 t x u0
b2 = b(1);
b3 = b(2);
u1 = pdepe(m,@pdeBurgersOPT2,@pdex1ic,@pdex1bc,x,t);
u1 = u1(:,:,1);
err = rms(reshape(u1-u0,[],1));
end
%----------------------------------------------
function err = errBurgers3(b)
m = 0;
global b1 b2 b3 t x u0
b1 = b(1);
b2 = b(2);
b3 = b(3);
u1 = pdepe(m,@pdeBurgersOPT3,@pdex1ic,@pdex1bc,x,t);
u1 = u1(:,:,1);
err = rms(reshape(u1-u0,[],1));
end
%--------------------------------------------------------------------
function [c,f,s] = pdeBurgersOPT(x,t,u,dudx) % Equation to solve
global a1 a2
c = 1;
f = a1*dudx;
s = a2*u*dudx;
end
%--------------------------------------------------------------------
function [c,f,s] = pdeBurgersOPT2(x,t,u,dudx) % Equation to solve
global b2 b3
c = 1;
f = b2*u*u*dudx;
s = -2*b2*u*(dudx)*(dudx)+b3*u*dudx;
end
%--------------------------------------------------------------------
function [c,f,s] = pdeBurgersOPT3(x,t,u,dudx) % Equation to solve
global b1 b2 b3
c = 1;
f = b1*dudx+b2*u*u*dudx;
s = -2*b2*u*(dudx)*(dudx)+b3*u*dudx;
end
%----------------------------------------------
function u0 = pdex1ic(x) % Initial conditions
u0 = -sin(pi*x);
end
%----------------------------------------------
function [pl,ql,pr,qr] = pdex1bc(xl,ul,xr,ur,t) % Boundary conditions
pl = ul;
ql = 0;
pr = ur;
qr = 0;
end
