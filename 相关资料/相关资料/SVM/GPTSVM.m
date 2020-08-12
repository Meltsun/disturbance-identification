clear all
close all
clc
%[1,15]两类特征，c1 = 7，c2 = 7，lambda1 = 4，lambda2 = 1     82

%投影孪生支持向量机
tic;
%% 初始化参数
m  = 5;                 %类别数
n = 2;
n1 = 50;                %第一类的样本数
n2 = 50;                %第二类的样本数
S  = n1+n2;             %样本总数
T  = 6;                 %训练样本：测试样本 = A：10-A,可调
DD  = 1;
s1 = n1*T/10;            %训练样本1
s2 = n2*T/10;            %训练样本2
%惩罚参数
%第一个点 %第三个点   %第二个点
c1 = 7;   %c1 = 8.3;  %c1 = 7.66;      5      6      7     8     10
c2 = 7;   %c2 = 5.53; %c2 = 8.52;      2      7      7     7     9
%调整参数
c3 = 0;
c4 = 0;
lambda1 = 4;                        % 4
lambda2 = 1;                        % 1
%Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出
%% 获取数据
load alldatapot2_lable.mat
data = alldatapot2_lable;
%data = xlsread('iris.xlsx');
%获取每类数据
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
    %获得每组的训练集和测试集
    %trD是训练集，teD是测试集
    [trDA,teDA] = F_CV(A1,T);
    [trDB,teDB] = F_CV(B2,T);
    [trDC,teDC] = F_CV(C3,T);
    [trDD,teDD] = F_CV(D4,T);
    [trDE,teDE] = F_CV(E5,T);
    %训练集汇总   
    trD = [trDA;trDB;trDC;trDD;trDE];
    c = 0;%计数
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*s1+1:i*s1,:); %第一类 
            trD2 = trD((j-1)*s2+1:j*s2,:); %第二类
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
            lb1 = zeros(n2*T/10,1); %相当于Quadprog函数中的LB，UB
            lb2 = zeros(n1*T/10,1);
            ub1 = c1/s2*e2;
            ub2 = c2/s1*e1;
            a01 = zeros(n2*T/10,1);  % a0是解的初始近似值
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
    %测试集及标签
    teD_z = [teDA; teDB;teDC;teDD;teDE];
    biaoqian = [ones(n1*(10-T)/10,1)*1;
                ones(n1*(10-T)/10,1)*2;
                ones(n1*(10-T)/10,1)*3;
                ones(n1*(10-T)/10,1)*4;
                ones(n1*(10-T)/10,1)*5];
    %% 分类器汇总
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    [row,col] = size(teD_z);
    %% 投票
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
