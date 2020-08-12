clc
clear all
close all

%% ��ʼ������
m = 5;                     %�����
n = 1;                     %������
t = 10;                    %����ģ��������ѹ���ĸ���
T = 7;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = n1_*T/10-t;           %��һ���ѵ��������
n2 = n2_*T/10-t;           %�ڶ����ѵ��������
s_ = n1_*(10-T)/10;        %ĳһ��Ĳ���������
DD = 1;                    %ѵ������
C = 10;
kertype = 'rbf';
d = 6;


load Over_lap.mat
dataT = [Odata{1,2};Odata{2,2};Odata{3,2};Odata{4,2};Odata{5,2}];
dataf = mapnormal(dataT);
for i = 1:250
    x(i,:) =mean(xcorr(dataf(i,:))); %dataf = mapnormal(dataT);
end
%% ��ȡ����
load alldatapot1_lable.mat
Data = alldatapot1_lable;
Data1 = feature_end(Data);           %�����¼ӵ�һ����������ӳ��ɢ������ɢ�̶�
qian = Data(:,end);                  %���һ��
data1 = mapminmax(Data(:,1:44),0,1); %������һ��
data = [data1,Data1,x,qian];           %�������
tz = [3,25,7,17,9,45,46];  %pot2 96%
% tz = [1:44];
 tz = 1;
A1 = data(1:50,tz);                  %��ȡ��һ��
B2 = data(51:100,tz);                %��ȡ�ڶ���
C3 = data(101:150,tz);               %��ȡ������
D4 = data(151:200,tz);               %��ȡ������
E5 = data(201:250,tz);               %��ȡ������
%% ѵ�����Ͳ��Լ�
% for g = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A��ѵ�����Ͳ��Լ����±�
[trDB,teDB,tr2] = F_CV(B2,T);            %B��ѵ�����Ͳ��Լ����±�
[trDC,teDC,tr3] = F_CV(C3,T);            %C��ѵ�����Ͳ��Լ����±�
[trDD,teDD,tr4] = F_CV(D4,T);            %D��ѵ�����Ͳ��Լ����±�
[trDE,teDE,tr5] = F_CV(E5,T);            %E��ѵ�����Ͳ��Լ����±�

trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%ѵ����ѹ������

%% ѵ��

c = 0;                                 %����
xlq = {};
for i = 1:m-1
    for j = i+1:m
        trD1 = trD((i-1)*n1+1:i*n1,:); %��һ�� 
        trD2 = trD((j-1)*n1+1:j*n1,:); %�ڶ���
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
    %% ����������
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    %% ���Լ�����ǩ
    teD_z = [teDA;teDB;teDC;teDD;teDE];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;
                ones(s_,1)*5];
    %% ����
    tic;
    [row,col] = size(teD_z);
    % ��һ��
    %���������
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
legend('Ԥ���ǩ','ʵ�ʱ�ǩ')
title('twsvmԤ�������ʵ�����ȶ�','fontsize',12)
ylabel('����ǩ','fontsize',12)
xlabel('������Ŀ','fontsize',12)
%% ������ȷ��
A_j = length(find(te_(1:s_,:) == 1))/(DD*s_);             %������ȷ��
A_q = length(find(te_(s_+1:2*s_,:) == 2))/(DD*s_);        %�õ���ȷ��
A_y = length(find(te_(2*s_+1:3*s_,:) == 3))/(DD*s_);      %ѹ����ȷ��
A_p = length(find(te_(3*s_+1:4*s_,:) == 4))/(DD*s_);      %������ȷ��
A_n = length(find(te_(4*s_+1:5*s_,:) == 5))/(DD*s_);      %�޵���ȷ��
A_z = sum(Accuracy)/DD;                                      %ƽ����ȷ��


sprintf('���Ĳ���׼ȷ��=%0.2f',A_j)
sprintf('�õĲ���׼ȷ��=%0.2f',A_q)
sprintf('ѹ�Ĳ���׼ȷ��=%0.2f',A_y)
sprintf('���Ĳ���׼ȷ��=%0.2f',A_p)
sprintf('�޵Ĳ���׼ȷ��=%0.2f',A_n)
sprintf('�ܵĲ���׼ȷ��=%0.2f',A_z)
