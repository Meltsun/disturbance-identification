clear all
close all
clc
tic;
%% ��ʼ������
m  = 5;                 %�����
n1 = 50;                %��һ���������
n2 = 50;                %�ڶ����������
S  = n1+n2;             %��������
T  = 6;                 %ѵ���������������� = A��10-A,�ɵ�
M  = 100;
%�ͷ�����
%��һ���� %��������   %�ڶ�����
c1 = 7.47;   %c1 = 8.3;  %c1 = 7.66;             
c2 = 7.54;   %c2 = 5.53; %c2 = 8.52;
%��������
c3 = 0;
c4 = 0;
%Options�����������㷨��ѡ�������������optimset�޲�ʱ������һ��ѡ��ṹ�����ֶ�ΪĬ��ֵ��ѡ��
options = optimset;    
options.LargeScale = 'off';%LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
options.Display = 'off';    %��ʾ�����
%% ��ȡ����
load alldatapot2_lable.mat
data = alldatapot2_lable;
%data = xlsread('iris.xlsx');
%��ȡÿ������
A1 = data(1:50,[3,4,5,15,24,26]);
B2 = data(51:100,[3,4,5,15,24,26]);
C3 = data(101:150,[3,4,5,15,24,26]);
D4 = data(151:200,[3,4,5,15,24,26]);
E5 = data(201:250,[3,4,5,15,24,26]);

for dd = 1:M
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
            trD1 = trD((i-1)*n1*T/10+1:i*n1*T/10,:); %��һ�� 
            trD2 = trD((j-1)*n2*T/10+1:j*n2*T/10,:); %�ڶ���

            e1 = ones(n1*T/10,1);
            e2 = ones(n2*T/10,1);

            H = [trD1,e1];
            G = [trD2,e2];
            I = ones(size(H'*H,1),size(H'*H,1));

            u1 = -c1*inv(H'*H+c1*(G'*G)+c3*I)*G'*e2;
            u2 = -c2*inv(G'*G+c2*(H'*H)+c4*I)*H'*e1;

            w1 = u1(1:end-1);
            b1 = u1(end);
            w2 = u2(1:end-1);
            b2 = u2(end);

            c = c+1;

            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
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
    Accuracy(dd,1) = 1-length(find((vote_y-biaoqian)~=0))/row;
end
A_M = sum(Accuracy)/M;
toc;



