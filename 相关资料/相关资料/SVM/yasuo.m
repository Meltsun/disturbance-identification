function data = yasuo(A,B,C,D,E,s)
%数据压缩，去除最无关数据
%s为压缩个数
SA = lsd(A,0);
SB = lsd(B,0);
SC = lsd(C,0);
SD = lsd(D,0);
SE = lsd(E,0);
[~,pA] = sort(SA,'descend');%从小到大
[~,pB] = sort(SB,'descend');
[~,pC] = sort(SC,'descend');
[~,pD] = sort(SD,'descend');
[~,pE] = sort(SE,'descend');
A_ = A(pA(1:end-s),:);
B_ = B(pB(1:end-s),:);
C_ = C(pC(1:end-s),:);
D_ = D(pD(1:end-s),:);
E_ = E(pE(1:end-s),:);
data = [A_;B_;C_;D_;E_];
%data = [A_;B_;C_;E_];
end