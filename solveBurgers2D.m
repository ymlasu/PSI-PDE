clear;close all;clc
clearvars -global
global a1 a2 u0 uData t x y dt
% prepare data
L=2;Nx=101;Ny=51;Nt=100;
x=linspace(-L/2,L/2,Nx);
y=linspace(-L/2,L/2,Ny);
dt=0.02;
t = 0:dt:(Nt-1)*dt;
[X,Y]=meshgrid(x,y);
u0=0.1*sech(20*X.^2+25*Y.^2);
a1 = 0.01;
a2 = -1;
u = solBurgers2D();
rng(1)
noiseL = 0.30;
un = u+noiseL*std(u,0,'all')*randn(size(u));
uData = un;

% Optimization using steepest descent method
rng default
% case N0
% a10 = 0.01;
% a20 = -1.0028;
% case N10
% a10 = 0.0102;
% a20 = -1.0417;
% case N20
% a10 = 0.0103;
% a20 = -1.0077; 
% case N30
% a10 = 0.0109;
% a20 = -1.0723; 
% case N40
a10 = 0.0107;
a20 = -1.0168; 
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
    f0 = errBurgers2D(a);
    g(1) = (errBurgers2D([a(1)+da0(1) a(2)]) - f0)/da0(1);
    g(2) = (errBurgers2D([a(1) a(2)+da0(2)]) - f0)/da0(2);
    da = -da0.*g;
    a = a+da;
    numIters = numIters+1;
    err = errBurgers2D(a)
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
%% plot solutions
clear;close all;clc
clearvars -global
global a1 a2 u0 uData t x y dt
% prepare data
L=2;Nx=101;Ny=51;Nt=100;
x=linspace(-L/2,L/2,Nx);
y=linspace(-L/2,L/2,Ny);
dt=0.02;
t = 0:dt:(Nt-1)*dt;
[X,Y]=meshgrid(x,y);
u0=0.1*sech(20*X.^2+25*Y.^2);
a1 = 0.01;
a2 = -1;
u = solBurgers2D();
rng(1)
noiseL = 0.40;
un = u+noiseL*std(u,0,'all')*randn(size(u));
a1 = 0.0101;
a2 = -1.0168; 
uSol = solBurgers2D();
ff = fig('units','inches','width',8,'height',2.5,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,3,[.02 .04],[.15 .08],[.06 .0]);
axes(h(1));box on;
imagesc(x,y,squeeze(un(1,:,:))');colorbar
xlabel('$y$','Interpreter','latex')
ylabel('$x$','Interpreter','latex')
title('(a)','FontWeight','normal')
axes(h(2));box on;
imagesc(x,y,squeeze(u(1,:,:)));colorbar
xlabel('$y$','Interpreter','latex')
title('(b)','FontWeight','normal')
axes(h(3));box on;
imagesc(x,y,squeeze(uSol(1,:,:)));colorbar
xlabel('$y$','Interpreter','latex')
title('(c)','FontWeight','normal')
% saveas(ff,'Figures\solBurgers2dN40.png')
% saveas(ff,'Figures\solBurgers2dN40.eps','epsc') 
% savefig(ff,'Figures\solBurgers2dN40.fig')
%% Functions
function err = errBurgers2D(a)
global a1 a2 uData
a1 = a(1);
a2 = a(2);
u1 = solBurgers2D();
err = sqrt(immse(u1,uData));
end

function u = solBurgers2D()
global a1 a2 u0 t x y dt
dx=x(2)-x(1);
dy=y(2)-y(1);
Nt = length(t);
Nx = length(x);
Ny = length(y);
u = zeros(Nt,Ny,Nx);
u(1,:,:)=u0;
for i=2:Nt
    u_1=reshape(u(i-1,:,:),Ny,Nx);
    formwork1=[-1:1];formwork2=[-1:1];
    du_t1=(a1*(differx(u_1,formwork1,2)/dx^2+differy(u_1,formwork1,2)/dy^2)...
        +a2*u_1.*((differx(u_1,formwork2,1)/dx+differy(u_1,formwork2,1)/dy)));
    u1=u_1+dt*du_t1;
    du_t2=(0.01*(differx(u1,formwork1,2)/dx^2+differy(u1,formwork1,2)/dy^2)...
        -u1.*((differx(u1,formwork2,1)/dx+differy(u1,formwork2,1)/dy)));
    u2=0.75*u_1+0.25*(u1+dt*du_t1);
    du_t3=(0.01*(differx(u2,formwork1,2)/dx^2+differy(u2,formwork1,2)/dy^2)...
        -u2.*((differx(u2,formwork2,1)/dx+differy(u2,formwork2,1)/dy)));
    unew=1/3*u_1+2/3*(u2+dt*du_t3);
    u(i,:,:)=unew;
end
end