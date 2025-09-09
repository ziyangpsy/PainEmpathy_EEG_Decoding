function [bestacc,bestc] = SVMcgForClass_NoDisplay_linear(train_label,train,cmin,cmax,v,cstep)

%% about the parameters of SVMcg 
if nargin < 6
    cstep = 0.8;
end
if nargin < 5
    v = 5;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end
%% X:c Y:g cg:CVaccuracy
X = meshgrid(cmin:cstep:cmax,1);
[m,n] = size(X);
cg = zeros(m,n);
eps = 1e-1;
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) )];
        cg(i,j) = svmtrain(train_label, train, cmd);
        
        if cg(i,j) <= 55
            continue;
        end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j) 
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
        end        
        
    end
end