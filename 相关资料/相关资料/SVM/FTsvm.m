clear all
close all
clc
tic;
%% ��ʼ������
m  = 5;                 %�����
n  = 50;
T  = 6;
s  = n*T/10;
                 %ѵ���������������� = A��10-A,�ɵ�
DD  = 1;
k = 0;
%�ͷ�����
%��һ���� %��������   %�ڶ�����
c1 = 9;   %c1 = 8.3;  %c1 = 7.66;      2      3      3     5     6     8     9
c2 = 9;   %c2 = 5.53; %c2 = 8.52;      2      2      4     5     6     7     9
c3 = 0;
c4 = 0;
lambda1 = 4;                        % 4
lambda2 = 4;                        % 1
%Options�����������㷨��ѡ�������������optimset�޲�ʱ������һ��ѡ��ṹ�����ֶ�ΪĬ��ֵ��ѡ��
options = optimset;    
options.LargeScale = 'off';%LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
options.Display = 'off';    %��ʾ�����
%% ��ȡ����
load alldatapot2_lable.mat
data = alldatapot2_lable;
%data = xlsread('iris.xlsx');
%��ȡÿ������
tz = [3,4]; 
%Actz = randi(44,1,6);
%tz = Actz;
%tz = randi(44,1,3);
%tz = [1,2,4,8,22,26];
%tz = [3,4,5,15,24,26];
A1 = data(1:50,tz);
B2 = data(51:100,tz);
C3 = data(101:150,tz);
D4 = data(151:200,tz);
E5 = data(201:250,tz);

%���ÿ���ѵ�����Ͳ��Լ�
%trD��ѵ������teD�ǲ��Լ�
[trDA,teDA] = F_CV(A1,T);
[trDB,teDB] = F_CV(B2,T);
[trDC,teDC] = F_CV(C3,T);
[trDD,teDD] = F_CV(D4,T);
[trDE,teDE] = F_CV(E5,T);
%ѵ��������   
%��һ��������
trD_1 = [trDC;trDA;trDB;trDD;trDE];
fq1  = [3,1245]; 
s1 = 4*s;
%�ڶ���������
trD_2 = [trDD;trDA;trDB;trDE];
fq2  = [4,125];
s2 = 3*s;
%������������
trD_3 = [trDA;trDB;trDE];
fq3  = [1,25];
s3 = 2*s;
%���ĸ�������
trD_4 = [trDB;trDE];
fq4  = [2,5];
s4 = 1*s;
%���Լ�
teD_z = [teDA;teDB;teDC;teDD;teDE];
biaoqian = [ones(n-s,1)*1;
            ones(n-s,1)*2;
            ones(n-s,1)*3;
            ones(n-s,1)*4;
            ones(n-s,1)*5];
%ѵ����һ��������
trD1_1 = trD_1(1:s,:);
trD1_2 = trD_1(s+1:end,:);
[SA,SB] = F_m(trD1_1,trD1_2,10.^(-8));
e1 = ones(s,1);
e2 = ones(s1,1);
H = [trD1_1,e1];
G = [trD1_2,e2];
P = H;
Q = G;
[w1_1,b1_1,w2_1,b2_1] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,s,s1,k,T,SA,SB,options);
%ѵ����2��������
trD2_1 = trD_2(1:s,:);
trD2_2 = trD_2(s+1:end,:);
[SA,SB] = F_m(trD2_1,trD2_2,10.^(-8));
e1 = ones(s,1);
e2 = ones(s2,1);
H = [trD2_1,e1];
G = [trD2_2,e2];
P = H;
Q = G;
[w1_2,b1_2,w2_2,b2_2] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,s,s2,k,T,SA,SB,options);
%ѵ����3��������
trD3_1 = trD_3(1:s,:);
trD3_2 = trD_3(s+1:end,:);
[SA,SB] = F_m(trD3_1,trD3_2,10.^(-8));
e1 = ones(s,1);
e2 = ones(s3,1);
H = [trD3_1,e1];
G = [trD3_2,e2];
P = H;
Q = G;
[w1_3,b1_3,w2_3,b2_3] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,s,s3,k,T,SA,SB,options);
%ѵ����4��������
trD4_1 = trD_4(1:s,:);
trD4_2 = trD_4(s+1:end,:);
[SA,SB] = F_m(trD4_1,trD4_2,10.^(-8));
e1 = ones(s,1);
e2 = ones(s4,1);
H = [trD4_1,e1];
G = [trD4_2,e2];
P = H;
Q = G;
[w1_4,b1_4,w2_4,b2_4] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,s,s4,k,T,SA,SB,options);

[row,col] = size(teD_z);

for i = 40
%���Ե�һ��������
    result1_1 = (w1_1'*teD_z(i,:)'+b1_1)/sqrt(w1_1'*w1_1);
    result1_2 = (w2_1'*teD_z(i,:)'+b2_1)/sqrt(w2_1'*w2_1);
    if(abs(result1_1) < abs(result1_2))
       te_(i,1) = fq1(1);
    else 
       result2_1 = (w1_2'*teD_z(i,:)'+b1_2)/sqrt(w1_2'*w1_2);
       result2_2 = (w2_2'*teD_z(i,:)'+b2_2)/sqrt(w2_2'*w2_2);
       if(abs(result2_1) < abs(result2_2))
           te_(i,1) = fq2(1);
       else
           result3_1 = (w1_3'*teD_z(i,:)'+b1_3)/sqrt(w1_3'*w1_3);
           result3_2 = (w2_3'*teD_z(i,:)'+b2_3)/sqrt(w2_3'*w2_3);
           if(abs(result3_1) < abs(result3_2))
              te_(i,1) = fq3(1);
           else
              result4_1 = (w1_4'*teD_z(i,:)'+b1_4)/sqrt(w1_4'*w1_4);
              result4_2 = (w2_4'*teD_z(i,:)'+b2_4)/sqrt(w2_4'*w2_4);
              if(abs(result4_1) < abs(result4_2))   
                  te_(i,1) = fq4(1);
              else
                  te_(i,1) = fq4(2);
              end
           end
       end
    end
end
Accuracy = 1-length(find((te_-biaoqian)~=0))/row;  
