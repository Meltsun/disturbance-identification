function data = Z_mean(Data)
[m,n] = size(Data);
a = mean(Data,1);
b = sum((Data - repmat(a,m,1)).^2,1)/m;
data = (Data-repmat(b,m,1))./repmat(a,m,1);
end