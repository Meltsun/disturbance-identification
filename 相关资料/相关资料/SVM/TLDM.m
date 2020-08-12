clear all
close all
clc
%[1,15]两类特征，c1 = 7，c2 = 7，lambda1 = 4，lambda2 = 1     82


tic;
%% 初始化参数
m  = 5;                 %类别数
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
tz = [3,4,5,16,17,18];
%tz = [15,1];
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
    %类别标签矩阵
    Y = [ones(s1,1);-ones(s2,1)];
    c = 0;%计数
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*s1+1:i*s1,:); %第一类 
            trD2 = trD((j-1)*s2+1:j*s2,:); %第二类
            X  = [trD1;trD2];
            e1 = ones(s1,1);
            e2 = ones(s2,1);
            e  = ones(s1+s2,1);
            %一系列参数
            M  = [X,e];
            D  = M'*(ones(s1+s2,s1+s2)-Y*Y');
            H  = [trD1,e1];
            G  = [trD2,e2];
            Q  = lambda2/(s1+s2)*M'*Y;
            P  = lambda1/(s1+s2).^2*D*M+H'*H; 
            S  = lambda1/(s1+s2).^2*D*M+G'*G; 
            
            [w1,b1,w2,b2] = TLDMtrain(H,Q,P,G,S,e1,e2,s1,s2,c1,c2,options);
           
            c = c+1;

            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;    
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
    Accuracy(g,1) = 1-length(find((vote_y-biaoqian)~=0))/row;  
end
A_N = sum(Accuracy)/DD;
toc;            