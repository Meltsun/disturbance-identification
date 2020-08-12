clc
clear all
close all

%% 初始化参数
m = 5;                     %类别数
n = 11;
T = 6;                     %交叉验证的比例。例如6：4
n1_ = 50;                  %第一类的样本数
n2_ = 50;                  %第一类的样本数
n1 = n1_*T/10;           %第一类的训练样本数
n2 = n2_*T/10;           %第二类的训练样本数
s_ = n1_*(10-T)/10;        %某一类的测试样本数
k = 1;                     %防止矩阵求逆奇异值的出现
s = 1;                     %高斯核函数中的方差
c1 = 8.9944;               %调整参数
c2 = 7.5212;               %调整参数
c3 = 0;                    %调整参数
c4 = 0;                    %调整参数
lambda1 = 4;               %调整参数                     
lambda2 = 1;               %调整参数
DD = 1;                    %训练次数
%% 获取数据
data = xlsread('D:\PyCharm Edu 2018.3\ML\selfCodes\new\p1z.xlsx');

A1 = data(1:50,1:end-1);                  %截取第一类
B2 = data(51:100,1:end-1);                %截取第二类
C3 = data(101:150,1:end-1);               %截取第三类
D4 = data(151:200,1:end-1);               %截取第四类
E5 = data(201:250,1:end-1);               %截取第五类

%% 训练集和测试集
for g = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A的训练集和测试集和下标
[trDB,teDB,tr2] = F_CV(B2,T);            %B的训练集和测试集和下标
[trDC,teDC,tr3] = F_CV(C3,T);            %C的训练集和测试集和下标
[trDD,teDD,tr4] = F_CV(D4,T);            %D的训练集和测试集和下标
[trDE,teDE,tr5] = F_CV(E5,T);            %E的训练集和测试集和下标

trD = [trDA;trDB;trDC;trDD;trDE];

%% 训练

    c = 0;                                 %计数
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
            trD2 = trD((j-1)*n1+1:j*n1,:); %第二类

            %基于TLDM方法
%             X  = [trD1;trD2];
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             e  = ones(n1+n2,1);
%             Y = [ones(n1,1);-ones(n2,1)];
%             M  = [X,e];
%             D  = M'*(ones(n1+n2,n1+n2)-Y*Y');
%             H  = [trD1,e1];
%             G  = [trD2,e2];
%             Q  = lambda2/(n1+n2)*M'*Y;
%             P  = lambda1/(n1+n2).^2*D*M+H'*H; 
%             S  = lambda1/(n1+n2).^2*D*M+G'*G; 
%             [w1,b1,w2,b2] = TLDMtrain(H,Q,P,G,S,e1,e2,n1,n2,c1,c2);

            %基于twsvm线性核函数方法
            e1 = ones(n1,1);
            e2 = ones(n2,1);
            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];
            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);

            %基于twsvm高斯核函数方法
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

            %基于LSTWSVM
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             H = [trD1,e1];
%             G = [trD2,e2];
%             I = ones(size(H'*H,1),size(H'*H,1));
%             u1 = -c1*inv(H'*H+c1*(G'*G)+c3*I)*G'*e2;
%             u2 = -c2*inv(G'*G+c2*(H'*H)+c4*I)*H'*e1;
%             w1 = u1(1:end-1);
%             b1 = u1(end);
%             w2 = u2(1:end-1);
%             b2 = u2(end);

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
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    %% 测试集及标签
    teD_z = [teDA;teDB;teDC;teDD;teDE];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;
                ones(s_,1)*5];
    %% 测试
    tic;
    [row,col] = size(teD_z);
    % 第一步
    %缩减类别数
    trA_mean = mean(trDA,1);
    trB_mean = mean(trDB,1);
    trC_mean = mean(trDC,1);
    trD_mean = mean(trDD,1);
    trE_mean = mean(trDE,1);
    for i = 1:row
         S1 = zeros(n,n);
         S2 = zeros(n,n);
         S3 = zeros(n,n);
         S4 = zeros(n,n);
         S5 = zeros(n,n);
         for k = 1:n1
             S1 = S1+(trDA(k,:)-trA_mean)'*(trDA(k,:)-trA_mean);
         end
         for k = 1:n1
             S2 = S2+(trDB(k,:)-trB_mean)'*(trDB(k,:)-trB_mean);
         end
         for k = 1:n1
             S3 = S3+(trDC(k,:)-trC_mean)'*(trDC(k,:)-trC_mean);
         end

         for k = 1:n1
             S4 = S4+(trDD(k,:)-trD_mean)'*(trDD(k,:)-trD_mean);
         end

         for k = 1:n1
             S5 = S5+(trDE(k,:)-trE_mean)'*(trDE(k,:)-trE_mean);
         end
         d1 = sqrt((teD_z(i,:)-trA_mean)*inv(S1)*(teD_z(i,:)-trA_mean)');
         d2 = sqrt((teD_z(i,:)-trB_mean)*inv(S2)*(teD_z(i,:)-trB_mean)');
         d3 = sqrt((teD_z(i,:)-trC_mean)*inv(S3)*(teD_z(i,:)-trC_mean)');
         d4 = sqrt((teD_z(i,:)-trD_mean)*inv(S4)*(teD_z(i,:)-trD_mean)');
         d5 = sqrt((teD_z(i,:)-trE_mean)*inv(S5)*(teD_z(i,:)-trE_mean)');
         [~,px] = sort([d1,d2,d3,d4,d5]);
         pxjg(i,:) = sort(px(1:2));
         pxjg_(i,:) = px;
    end 
     for i = 1:row
         for j = 1:c
             if((flq(j,1) == pxjg(i,1))&&(flq(j,2) == pxjg(i,2)))
                jg_(i,1) = j;
             end
         end
     end
    %第二步分类
    for i = 1:row
        for k = 1:1
            %无核
            result1 = (W1(jg_(i,k),:)*teD_z(i,:)'+B1(jg_(i,k)))/(W1(jg_(i,k),:)*W1(jg_(i,k),:)');
            result2 = (W2(jg_(i,k),:)*teD_z(i,:)'+B2(jg_(i,k)))/(W2(jg_(i,k),:)*W2(jg_(i,k),:)');
            %有核
%             result1 = W1(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B1(jg_(i,k));
%             result2 = W2(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B2(jg_(i,k));
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
         vote_y(i,g) = idx;
     end
     Accuracy(g,1) = 1-length(find((vote_y(:,g)-biaoqian)~=0))/row;
end 
toc;
%% 画图
figure(1)
plot(vote_y,'og')
hold on
plot(biaoqian,'r*');
legend('预测标签','实际标签')
title('twsvm预测分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
%% 分类正确率
A_j = length(find(vote_y(1:s_,:) == 1))/(DD*s_);             %浇的正确率
A_q = length(find(vote_y(s_+1:2*s_,:) == 2))/(DD*s_);        %敲的正确率
A_p = length(find(vote_y(2*s_+1:3*s_,:) == 3))/(DD*s_);      %爬的正确率
A_y = length(find(vote_y(3*s_+1:4*s_,:) == 4))/(DD*s_);      %压的正确率
A_n = length(find(vote_y(4*s_+1:5*s_,:) == 5))/(DD*s_);      %无的正确率
A_z = sum(Accuracy)/DD;                                      %平均正确率


sprintf('浇的测试准确率=%0.2f',A_j)
sprintf('敲的测试准确率=%0.2f',A_q)
sprintf('爬的测试准确率=%0.2f',A_p)
sprintf('压的测试准确率=%0.2f',A_y)
sprintf('无的测试准确率=%0.2f',A_n)
sprintf('总的测试准确率=%0.2f',A_z)
