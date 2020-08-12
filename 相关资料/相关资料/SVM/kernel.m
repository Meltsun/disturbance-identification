%---------------�˺���---------------
function K = kernel(X,Y,type)
%X ά��*����
switch type
case 'linear'   %��ʱ�������Ժ�
    K = X'*Y;
case 'rbf'      %��ʱ�����˹��
    delta = 7;
    XX = sum(X'.*X',2);%2��ʾ�������еİ���Ϊ��λ�������
    YY = sum(Y'.*Y',2);
    XY = X'*Y;
    K = abs(repmat(XX,[1 size(YY,1)]) + repmat(YY',[size(XX,1) 1]) - 2*XY);
    K = exp(-K./delta);
  % K=(norm(X'-Y))^2;
  % K=exp(-K./(delta*2));
case 'dxs'      %����ʽ�˺���
    d = 3;
    K = (X'*Y+1).^d;
end
end


