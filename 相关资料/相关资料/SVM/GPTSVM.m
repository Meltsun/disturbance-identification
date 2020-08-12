clear all
close all
clc
%[1,15]����������c1 = 7��c2 = 7��lambda1 = 4��lambda2 = 1     82

%ͶӰ����֧��������
tic;
%% ��ʼ������
m  = 5;                 %�����
n = 2;
n1 = 50;                %��һ���������
n2 = 50;                %�ڶ����������
S  = n1+n2;             %��������
T  = 6;                 %ѵ���������������� = A��10-A,�ɵ�
DD  = 1;
s1 = n1*T/10;            %ѵ������1
s2 = n2*T/10;            %ѵ������2
%�ͷ�����
%��һ���� %��������   %�ڶ�����
c1 = 7;   %c1 = 8.3;  %c1 = 7.66;      5      6      7     8     10
c2 = 7;   %c2 = 5.53; %c2 = 8.52;      2      7      7     7     9
%��������
c3 = 0;
c4 = 0;
lambda1 = 4;                        % 4
lambda2 = 1;                        % 1
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
%tz = [3,4,5,16,17,18];
tz = [15,1];
%tz = randi(44,1,2);
A1 = data(1:50,tz);
B2 = data(51:100,tz);
C3 = data(101:150,tz);
D4 = data(151:200,tz);
E5 = data(201:250,tz);

for g = 1:DD
    %���ÿ���ѵ�����Ͳ��Լ�
    %trD��ѵ������teD�ǲ��Լ�
    [trDA,teDA] = F_CV(A1,T);
    [trDB,teDB] = F_CV(B2,T);
    [trDC,teDC] = F_CV(C3,T);
    [trDD,teDD] = F_CV(D4,T);
    [trDE,teDE] = F_CV(E5,T);
    %ѵ��������   
    trD = [trDA;trDB;trDC;trDD;trDE];
    c = 0;%����
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*s1+1:i*s1,:); %��һ�� 
            trD2 = trD((j-1)*s2+1:j*s2,:); %�ڶ���
            A = trD1;
            B = trD2;
            e1 = ones(s1,1);
            e2 = ones(s2,1);
            A_mean = mean(A,1);
            B_mean = mean(B,1);
            S1 = zeros(n,n);
            S2 = zeros(n,n);
            for k = 1:s1
                S1 = S1+(A(k,:)-A_mean)'*(A(k,:)-A_mean);
            end
            for k = 1:s2
                S2 = S2+(B(k,:)-B_mean)'*(B(k,:)-B_mean);
            end
            H1 = (B-1/s1*e2*e1'*A)*inv(S1)*(B'-1/s1*A'*e1*e2');
            H2 = (A-1/s2*e1*e2'*B)*inv(S2)*(A'-1/s2*B'*e2*e1');
            f1 = [];
            f2 = [];
            A1 = [];
            A2 = [];
            b1 = [];
            b2 = [];
            Aeq1 = []; 
            Aeq2 = []; 
            beq1 = [];
            beq2 = [];
            lb1 = zeros(n2*T/10,1); %�൱��Quadprog�����е�LB��UB
            lb2 = zeros(n1*T/10,1);
            ub1 = c1/s2*e2;
            ub2 = c2/s1*e1;
            a01 = zeros(n2*T/10,1);  % a0�ǽ�ĳ�ʼ����ֵ
            a02 = zeros(n1*T/10,1);
            [a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb1,ub1,a01,options);
            [a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb2,ub2,a02,options);
            w1 = inv(S1)*(B'-1/s1*A'*e1*e2')*a1;
            w2 = inv(S2)*(A'-1/s2*B'*e2*e1')*a2;
            c = c+1;

            W1(c,:) = w1';
            W2(c,:) = w2';
            
        end
    end
    %���Լ�����ǩ
    teD_z = [teDA; teDB;teDC;teDD;teDE];
    biaoqian = [ones(n1*(10-T)/10,1)*1;
                ones(n1*(10-T)/10,1)*2;
                ones(n1*(10-T)/10,1)*3;
                ones(n1*(10-T)/10,1)*4;
                ones(n1*(10-T)/10,1)*5];
    %% ����������
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    [row,col] = size(teD_z);
    %% ͶƱ
    for i = 1:row
        for k = 1:c
            result1 = W1(k,:)*(teD_z(i,:)-A_mean)';
            result2 = W2(k,:)*(teD_z(i,:)-B_mean)';
            if(abs(result1) > abs(result2))
               te_(i,k) = flq(k,2);
            else
               te_(i,k) = flq(k,1);
            end 
            result(i,k) = abs(result1)-abs(result2);
            
        end
    end
    for i =1:row
        table = tabulate(te_(i,:));
        [~,idx] = max(table(:,2));
        vote_y(i,1) = idx;
    end
    Accuracy(g,1) = 1-length(find((vote_y-biaoqian)~=0))/row;  
end
A_N = sum(Accuracy)/DD;
