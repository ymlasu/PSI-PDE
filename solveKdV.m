clear;close all;clc
clearvars -global
global a1 a2 t k u0t uData
Lx = 2; Lt = 1;
Nx = 512; Nt = 201;
x = Lx/Nx*(-Nx/2:Nx/2-1);
k = 2*pi/Lx*[0:Nx/2-1 -Nx/2:-1].';
u0 = cos(pi*x);   % initial condition
u0t = fft(u0);
t = Lt/Nt*(0:Nt-1);
% load('Datasets\KdV.mat','uu')
a1 = -1;
a2 = -0.0025;
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
uu=real(usol);
rng(1)
noiseL = 0.50;
un = uu+noiseL*std(uu,0,'all')*randn(size(uu));
uData = un;
clear Lx Lt Nx Nt un
rng default
% case N0
% a10 = -0.9994;
% a20 = -0.0025;
% case N10
% a10 = -0.9756;
% a20 = -0.0025;  
% case N20
% a10 = -0.9618;
% a20 = -0.0024;  
% case N50
a10 = -1.0222;
a20 = -0.0025;  
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
    f0 = errKdV(a);
    g(1) = (errKdV([a(1)+da0(1) a(2)]) - f0)/da0(1);
    g(2) = (errKdV([a(1) a(2)+da0(2)]) - f0)/da0(2);
    da = -da0.*g;
    a = a+da;
    numIters = numIters+1;
    err = errKdV(a)
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
global a1 a2 t k u0t uData
Lx = 2; Lt = 1;
Nx = 512; Nt = 201;
x = Lx/Nx*(-Nx/2:Nx/2-1);
k = 2*pi/Lx*[0:Nx/2-1 -Nx/2:-1].';
u0 = cos(pi*x);   % initial condition
u0t = fft(u0);
t = Lt/Nt*(0:Nt-1);
% load('Datasets\KdV.mat','uu')
a1 = -1;
a2 = -0.0025;
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
uu=real(usol);
rng(1)
noiseL = 0.50;
u116 = uu(116,:);
un = uu+noiseL*std(uu,0,'all')*randn(size(uu));
uData = un;

ff = fig('units','inches','width',7,'height',3,'font','Times New Roman','fontsize',11);
h = tight_subplot(1,2,[.04 .06],[.15 .08],[.08 .02]);
axes(h(1));box on;
imagesc(t,x,uData');colorbar
xline(0.8,'w-','LineWidth',2.0)
xlabel('$t$','Interpreter','latex')
ylabel('$x$','Interpreter','latex')
title('(a) $u(x,t)$','FontWeight','normal','Interpreter','latex')
axes(h(2));box on;hold on 
plot(x,uData(116,:),'b-','LineWidth',1.5)
plot(x,u116,'g-','LineWidth',2.0)
a1 = -1.0221;
a2 = -0.0029;
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
uu=real(usol);
plot(x,uu(116,:),'r--','LineWidth',2.0)
set(gca,'xtick',-1:0.5:1)
set(gca,'xticklabel',-1:0.5:1)
set(gca,'ytick',-1:3)
set(gca,'yticklabel',-1:3)
xlabel('$x$','Interpreter','latex')
ylabel('$u$','Interpreter','latex')
legend({'measurement','ground truth','solution'},'location','northwest')
legend boxoff
title('(b) $u(x,0.80)$','FontWeight','normal','Interpreter','latex')
% saveas(ff,'Figures\solKdvN50.png')
% saveas(ff,'Figures\solKdvN50.eps','epsc') 
% savefig(ff,'Figures\solKdvN50.fig')
%% functions
function err = errKdV(a)
global a1 a2 t k u0t uData
a1 = a(1);
a2 = a(2);
[t,utso1]=ode23tb('KdV',t,u0t,[],k);
usol=ifft(utso1,[],2);
u1=real(usol);
err = rms(reshape(u1-uData,[],1));
end