clear all
close all
clc
tic;
%% ��ʼ������
m  = 5;                 %�����
n1 = 50;                %��һ���������
n2 = 200;                %�ڶ����������
S  = n1+n2;             %��������
T  = 6;                 %ѵ���������������� = A��10-A,�ɵ�
DD  = 1;
s1 = n1*T/10;            %ѵ������1
s2 = n2*T/10;            %ѵ������2
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
%tz = [3,4,5,15,24,26]; 
%Actz = randi(44,1,6);
%tz = Actz;
tz = randi(44,1,3);
%tz = [1,2,4,8,22,26];
tz = [3,4,5,15,24,26];
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
trD = [trDA;trDB;trDC;trDD;trDE];
%ѵ��������   
trD_ = trD;
%����ǩ����
Y = [ones(s1,1);-ones(s2,1)];
c = 0;%����
for i = 1:m
    trD1 = trD((i-1)*s1+1:i*s1,:); %��һ�� 
    trD_((i-1)*s1+1:i*s1,:) = [];
    trD2 =trD_;                     %�ڶ���
    trD_ =trD; 
    [SA,SB] = F_m(trD1,trD2,0.1);
    e1 = ones(s1,1);
    e2 = ones(s2,1);
    
     H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];

            [w1,b1,w2,b2] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,s1,s2,k,T,SA,SB,options);

            c = c+1;

            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
end

  %���Լ�����ǩ
    teD_z = [teDA; teDB;teDC;teDD;teDE];
    biaoqian = [ones(n1*(10-T)/10,1)*1;
                ones(n1*(10-T)/10,1)*2;
                ones(n1*(10-T)/10,1)*3;
                ones(n1*(10-T)/10,1)*4;
                ones(n1*(10-T)/10,1)*5];
    %% ����������
    flq = [1,2345;2,1345;3,1245;4,1235;5,1234];
    [row,col] = size(teD_z);
    %% ͶƱ
    for i = 1:row
        for k = 1:c
            result1 = (W1(k,:)*teD_z(i,:)'+B1(k))/sqrt(W1(k,:)*W1(k,:)');
            result2 = (W2(k,:)*teD_z(i,:)'+B2(k))/sqrt(W2(k,:)*W2(k,:)');
            if(abs(result1) > abs(result2))
               te_(i,k) = flq(k,2);
            else
               te_(i,k) = flq(k,1);
            end 
        end
    end
    for i =1:row
        table = tabulate(te_(i,:));
        [~,idx] = max(table(:,2));
        vote_y(i,1) = idx;
    end
    Accuracy = 1-length(find((vote_y-biaoqian)~=0))/row;  

          
   