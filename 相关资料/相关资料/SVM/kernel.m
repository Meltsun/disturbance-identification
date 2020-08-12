%---------------核函数---------------
function K = kernel(X,Y,type)
%X 维数*个数
switch type
case 'linear'   %此时代表线性核
    K = X'*Y;
case 'rbf'      %此时代表高斯核
    delta = 7;
    XX = sum(X'.*X',2);%2表示将矩阵中的按行为单位进行求和
    YY = sum(Y'.*Y',2);
    XY = X'*Y;
    K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
    K = exp(-K./delta);
  % K=(norm(X'-Y))^2;
  % K=exp(-K./(delta*2));
case 'dxs'      %多项式核函数
    d = 3;
    K = (X'*Y+1).^d;
end
end


