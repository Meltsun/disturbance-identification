clc
clear all
close all

%% 初始化参数
m = 5;                     %类别数
n = 1;                     %特征数
t = 10;                    %基于模糊隶属度压缩的个数
T = 7;                     %交叉验证的比例。例如6：4
n1_ = 50;                  %第一类的样本数
n2_ = 50;                  %第一类的样本数
n1 = n1_*T/10-t;           %第一类的训练样本数
n2 = n2_*T/10-t;           %第二类的训练样本数
s_ = n1_*(10-T)/10;        %某一类的测试样本数
DD = 1;                    %训练次数
C = 10;
kertype = 'rbf';
d = 6;


load Over_lap.mat
dataT = [Odata{1,2};Odata{2,2};Odata{3,2};Odata{4,2};Odata{5,2}];
dataf = mapnormal(dataT);
for i = 1:250
    x(i,:) =mean(xcorr(dataf(i,:))); %dataf = mapnormal(dataT);
end
%% 获取数据
load alldatapot1_lable.mat
Data = alldatapot1_lable;
Data1 = feature_end(Data);           %后面新加的一个特征它反映离散点间的离散程度
qian = Data(:,end);                  %最后一列
data1 = mapminmax(Data(:,1:44),0,1); %特征归一化
data = [data1,Data1,x,qian];           %重新组合
tz = [3,25,7,17,9,45,46];  %pot2 96%
% tz = [1:44];
 tz = 1;
A1 = data(1:50,tz);                  %截取第一类
B2 = data(51:100,tz);                %截取第二类
C3 = data(101:150,tz);               %截取第三类
D4 = data(151:200,tz);               %截取第四类
E5 = data(201:250,tz);               %截取第五类
%% 训练集和测试集
% for g = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A的训练集和测试集和下标
[trDB,teDB,tr2] = F_CV(B2,T);            %B的训练集和测试集和下标
[trDC,teDC,tr3] = F_CV(C3,T);            %C的训练集和测试集和下标
[trDD,teDD,tr4] = F_CV(D4,T);            %D的训练集和测试集和下标
[trDE,teDE,tr5] = F_CV(E5,T);            %E的训练集和测试集和下标

trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%训练集压缩汇总

%% 训练

c = 0;                                 %计数
xlq = {};
for i = 1:m-1
    for j = i+1:m
        trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
        trD2 = trD((j-1)*n1+1:j*n1,:); %第二类
        X = [trD1',trD2'];
        Y =[ones(1,n1),-ones(1,n2)]; 
        svm = svmTrain(X,Y,kertype,C,d);
        c = c+1;
        xlq{c} = svm;
%         w = zeros(1,n);
%         
%         for k = 1:svm.svnum
%             w = w + svm.a(k)*svm.Ysv(k)*svm.Xsv(:,k)';
%         end
%         for k = 1:svm.svnum
%             b(k) = 1/svm.Ysv(k)-w*svm.Xsv(:,k);
%         end
%         b1 = mean(b);
%         W(c,:) = w;
%         B(c) = b1;
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
result = {};
 for i = 1:row
     for k = 1:1
         result = svmTest(xlq{jg_(i,k)}, teD_z(i,:)', biaoqian,kertype,d);
         if(result.Y == 1)
             te_(i,k) = flq(jg_(i,k),1);
         else
             te_(i,k) = flq(jg_(i,k),2);
         end
     end
 end
 Accuracy = 1-length(find((te_-biaoqian)~=0))/row;
 toc;
 figure(1)
plot(te_,'og')
hold on
plot(biaoqian,'r*');
legend('预测标签','实际标签')
title('twsvm预测分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
%% 分类正确率
A_j = length(find(te_(1:s_,:) == 1))/(DD*s_);             %浇的正确率
A_q = length(find(te_(s_+1:2*s_,:) == 2))/(DD*s_);        %敲的正确率
A_y = length(find(te_(2*s_+1:3*s_,:) == 3))/(DD*s_);      %压的正确率
A_p = length(find(te_(3*s_+1:4*s_,:) == 4))/(DD*s_);      %爬的正确率
A_n = length(find(te_(4*s_+1:5*s_,:) == 5))/(DD*s_);      %无的正确率
A_z = sum(Accuracy)/DD;                                      %平均正确率


sprintf('浇的测试准确率=%0.2f',A_j)
sprintf('敲的测试准确率=%0.2f',A_q)
sprintf('压的测试准确率=%0.2f',A_y)
sprintf('爬的测试准确率=%0.2f',A_p)
sprintf('无的测试准确率=%0.2f',A_n)
sprintf('总的测试准确率=%0.2f',A_z)
