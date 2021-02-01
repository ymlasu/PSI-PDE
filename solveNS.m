clear;close all;clc
clearvars -global
global un vn pn
[~,~,~,u,v,p] = mit18086_navierstokes(0,-1,-1,0.01);
rng(1)
noiseL = 0.50;
un = u+noiseL*std(u,0,'all')*randn(size(u));
vn = v+noiseL*std(v,0,'all')*randn(size(v));
pn = p+noiseL*std(p,0,'all')*randn(size(p));

rng default
% case N0
% a10 = -0.9917;
% a20 = -0.9633;
% a30 = 0.0104;
% case N10
% a10 = -0.9769;
% a20 = -0.9678;
% a30 = 0.0101;
% case N20
% a10 = -1.0056;
% a20 = -1.0060;
% a30 = 0.0107;
% case N30
% a10 = -0.9727;
% a20 = -0.9758;
% a30 = 0.0108;
% case N40
% a10 = -0.9791;
% a20 = -0.9095;
% a30 = 0.0103;
% case N50
a10 = -1.0069;
a20 = -0.8362;
a30 = 0.0107;
a0 = [a10 a20 a30];

da0 = [1E-2 1E-2 1E-3];
da = da0;
errTol = 1E-6;
dTol = 1E-10;
badTol = 20;
maxIters = 1E5;
err = 1;
a = a0;
g = [0 0 0];
numIters = 0;
numBad = 0;
errMin = err;
while (numIters<maxIters && min(abs(da./da0)>dTol*abs(da0)) && err>errTol)
    f0 = errNS(a);
    g(1) = (errNS([a(1)+da0(1) a(2) a(3)]) - f0)/da0(1);
    g(2) = (errNS([a(1) a(2)+da0(2) a(3)]) - f0)/da0(2);
    g(3) = (errNS([a(1) a(2) a(3)+da0(3)]) - f0)/da0(3);
    da = -da0.*g;
    a = a+da;
    numIters = numIters+1;
    da(2)
    err = errNS(a)
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
%% matlab global optimization
a0 = [a10 a20 a30];
gs = GlobalSearch;
sixmin = @errNS;
options = optimoptions('fmincon','Display','iter','MaxIterations',100);
problem = createOptimProblem('fmincon','x0',[a10,a20,a30],...
    'objective',sixmin,'lb',[-2,-2,1E-4],'ub',[1,1,1],'options',options);
x = run(gs,problem)
%% error function
function err = errNS(a)
global un vn pn
[~,~,~,u,v,p] = mit18086_navierstokes(0,a(1),a(2),a(3));
err1 = sqrt(immse(u,un));
err2 = sqrt(immse(v,vn));
err3 = sqrt(immse(p,pn));
err = rms([err1 err2 err3]);
end