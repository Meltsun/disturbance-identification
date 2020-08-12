clear all
close all
clc

data = xlsread('iris.xlsx');
m = 3;                     %类别数
n = 4;                     %特征数
T = 7;                     %交叉验证的比例。例如6：4
n1_ = 50;                  %第一类的样本数
n2_ = 50;                  %第一类的样本数
n1 = n1_*T/10;             %第一类的训练样本数
n2 = n2_*T/10;             %第二类的训练样本数
s_ = n1_*(10-T)/10;        %某一类的测试样本数
k = 0;                     %防止矩阵求逆奇异值的出现
s = 1;                     %高斯核函数中的方差
c1 = 0.25;                 %调整参数
c2 = 1;                    %调整参数
A1 = data(1:50,1:4);                  %截取第一类
B2 = data(51:100,1:4);                %截取第二类
C3 = data(101:150,1:4);               %截取第三类
%% 训练集和测试集
[trDA,teDA,tr1] = F_CV(A1,T);            %A的训练集和测试集和下标
[trDB,teDB,tr2] = F_CV(B2,T);            %B的训练集和测试集和下标
[trDC,teDC,tr3] = F_CV(C3,T);            %C的训练集和测试集和下标
trD =[trDA;trDB;trDC];                   %训练集压缩汇总
c= 0;
for i = 1:m-1
    for j = i+1:m
        trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
        trD2 = trD((j-1)*n1+1:j*n1,:); %第二类
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
                %分类器的个数
        c = c+1;
        %权值汇总
        W1(c,:) = w1';
        W2(c,:) = w2';
        B1(c)   = b1;
        B2(c)   = b2;
    end
end
 %% 分类器汇总
flq = [1,2;1,3;2,3];
%% 测试集及标签
teD_z = [teDA;teDB;teDC];
biaoqian = [ones(s_,1)*1;
            ones(s_,1)*2;
            ones(s_,1)*3];
 %% 测试
tic;
[row,col] = size(teD_z);
% 第一步
%缩减类别数
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
        %无核
         result1 = (W1(jg_(i,k),:)*teD_z(i,:)'+B1(jg_(i,k)))/(W1(jg_(i,k),:)*W1(jg_(i,k),:)');
         result2 = (W2(jg_(i,k),:)*teD_z(i,:)'+B2(jg_(i,k)))/(W2(jg_(i,k),:)*W2(jg_(i,k),:)');
%       %有核%         for p = 1:n
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
     %投票
 for i =1:row
     table = tabulate(te_(i,:));
     [~,idx] = max(table(:,2));
     vote_y(i,1) = idx;
 end
 Accuracy(1,1) = 1-length(find((vote_y(:,1)-biaoqian)~=0))/row;
 toc;