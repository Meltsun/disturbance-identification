function S = lsd(X,s)
[m,~] = size(X);
CA = mean(X,1);
for i = 1:m
    diA2(i,1) = norm(X(i,:)-CA).^2;        
end
rA2 = max(diA2);
for i = 1:m
    %SA(i,1) = 1-1/(1+rA2-diA2(i,1)+s);
    S(i,1) = 1-diA2(i,1)/(rA2+s);
end
end