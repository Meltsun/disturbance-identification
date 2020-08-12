function [SA,SB] = F_m(X,Y,s)
%X为第一类数据，Y为第二类数据
%SA为第一类数据所有样本的隶属度，SB为第一类数据所有样本的隶属度。
[m,~] = size(X);
[n,~] = size(Y);
CA = mean(X,1);
CB = mean(Y,1);
for i = 1:m
    diA2(i,1) = norm(X(i,:)-CA).^2;        
end
rA2 = max(diA2);
for i = 1:m
    %SA(i,1) = 1-1/(1+rA2-diA2(i,1)+s);
    SA(i,1) = 1-diA2(i,1)/(rA2+s);
end
for i = 1:n
    diB2(i,1) = norm(Y(i,:)-CB).^2;
end
rB2 = max(diB2);
for i = 1:n
    %SB(i,1) = 1-1/(1+rB2-diB2(i,1)+s);   
    SB(i,1) = 1-diB2(i,1)/(rB2+s);
end