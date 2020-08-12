clear all
close all
clc

data = xlsread('iris.xlsx');
m = 3;                     %�����
n = 4;                     %������
T = 7;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = n1_*T/10;             %��һ���ѵ��������
n2 = n2_*T/10;             %�ڶ����ѵ��������
s_ = n1_*(10-T)/10;        %ĳһ��Ĳ���������
k = 0;                     %��ֹ������������ֵ�ĳ���
s = 1;                     %��˹�˺����еķ���
c1 = 0.25;                 %��������
c2 = 1;                    %��������
A1 = data(1:50,1:4);                  %��ȡ��һ��
B2 = data(51:100,1:4);                %��ȡ�ڶ���
C3 = data(101:150,1:4);               %��ȡ������
%% ѵ�����Ͳ��Լ�
[trDA,teDA,tr1] = F_CV(A1,T);            %A��ѵ�����Ͳ��Լ����±�
[trDB,teDB,tr2] = F_CV(B2,T);            %B��ѵ�����Ͳ��Լ����±�
[trDC,teDC,tr3] = F_CV(C3,T);            %C��ѵ�����Ͳ��Լ����±�
trD =[trDA;trDB;trDC];                   %ѵ����ѹ������
c= 0;
for i = 1:m-1
    for j = i+1:m
        trD1 = trD((i-1)*n1+1:i*n1,:); %��һ�� 
        trD2 = trD((j-1)*n1+1:j*n1,:); %�ڶ���
        e1 = ones(n1,1);
        e2 = ones(n2,1);
        H = [trD1,e1];
        G = [trD2,e2];
        P = [trD1,e1];
        Q = [trD2,e2];
        [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);
%             A = trD1;
%             B = trD2;
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             C = [A',B']';
%             S = [rbf(A,C',s),e1];
%             R = [rbf(B,C',s),e2];
%             L = [rbf(A,C',s),e1];
%             N = [rbf(B,C',s),e2];
%             [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,c1,c2,50,50,k,T);
                %�������ĸ���
        c = c+1;
        %Ȩֵ����
        W1(c,:) = w1';
        W2(c,:) = w2';
        B1(c)   = b1;
        B2(c)   = b2;
    end
end
 %% ����������
flq = [1,2;1,3;2,3];
%% ���Լ�����ǩ
teD_z = [teDA;teDB;teDC];
biaoqian = [ones(s_,1)*1;
            ones(s_,1)*2;
            ones(s_,1)*3];
 %% ����
tic;
[row,col] = size(teD_z);
% ��һ��
%���������
trA_mean = mean(trDA,1);
trB_mean = mean(trDB,1);
trC_mean = mean(trDC,1);
 for i = 1:row
     S1 = zeros(n,n);
     S2 = zeros(n,n);
     S3 = zeros(n,n);
     for k = 1:n1
         S1 = S1+(trDA(k,:)-trA_mean)'*(trDA(k,:)-trA_mean);
     end
     for k = 1:n1
         S2 = S2+(trDB(k,:)-trB_mean)'*(trDB(k,:)-trB_mean);
     end
     for k = 1:n1
         S3 = S3+(trDC(k,:)-trC_mean)'*(trDC(k,:)-trC_mean);
     end
     d1 = sqrt((teD_z(i,:)-trA_mean)*inv(S1)*(teD_z(i,:)-trA_mean)');
     d2 = sqrt((teD_z(i,:)-trB_mean)*inv(S2)*(teD_z(i,:)-trB_mean)');
     d3 = sqrt((teD_z(i,:)-trC_mean)*inv(S3)*(teD_z(i,:)-trC_mean)');
     [~,px] = sort([d1,d2,d3]);
     pxjg(i,:) = sort(px(1:2));
     pxjg_(i,:) = px(1:2);
 end
for i = 1:row
     for j = 1:c
         if((flq(j,1) == pxjg(i,1))&&(flq(j,2) == pxjg(i,2)))
            jg_(i,1) = j;
         end
     end
end
 for i = 1:row
     for k = 1:1
        %�޺�
         result1 = (W1(jg_(i,k),:)*teD_z(i,:)'+B1(jg_(i,k)))/(W1(jg_(i,k),:)*W1(jg_(i,k),:)');
         result2 = (W2(jg_(i,k),:)*teD_z(i,:)'+B2(jg_(i,k)))/(W2(jg_(i,k),:)*W2(jg_(i,k),:)');
%       %�к�%         for p = 1:n
%             for q = 1:n
%                 H(p,q) = exp(-s*norm(C(:,p)-C(:,q))^2);
%             end
%         end
%         
%          result1 = (W1(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B1(jg_(i,k)))/sqrt(W1(jg_(i,k),:)*H*W1(jg_(i,k),:)');
%          result2 = (W2(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B2(jg_(i,k)))/sqrt(W2(jg_(i,k),:)*H*W2(jg_(i,k),:)');

         if(abs(result1) > abs(result2))
               te_(i,k) = flq(jg_(i,k),2);
         else
               te_(i,k) = flq(jg_(i,k),1);
         end 
            result(i,1) = abs(result1);
            result(i,2) = abs(result2);
       
     end
 end
     %ͶƱ
 for i =1:row
     table = tabulate(te_(i,:));
     [~,idx] = max(table(:,2));
     vote_y(i,1) = idx;
 end
 Accuracy(1,1) = 1-length(find((vote_y(:,1)-biaoqian)~=0))/row;
 toc;