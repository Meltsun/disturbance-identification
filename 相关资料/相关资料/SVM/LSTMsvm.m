clear all
close all
clc
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
c1 = 7.47;   %c1 = 8.3;  %c1 = 7.66;             
c2 = 7.54;   %c2 = 5.53; %c2 = 8.52;
%调整参数
c3 = 0;
c4 = 0;
lambda1 = 1;
lambda2 = 1;
lambda3 = 1;
lambda4 = 1;
%Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出
%% 获取数据
load alldatapot3_lable.mat
data = alldatapot3_lable;
%data = xlsread('iris.xlsx');
%获取每类数据
A1 = data(1:50,[3,4,5,15,24,26]);
B2 = data(51:100,[3,4,5,15,24,26]);
C3 = data(101:150,[3,4,5,15,24,26]);
D4 = data(151:200,[3,4,5,15,24,26]);
E5 = data(201:250,[3,4,5,15,24,26]);

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
            I  = ones(size(H'*H,1),size(H'*H,1));
            
            u1 = inv(lambda1/(s1+s2)*D*M+c3*I+H'*H+c1*G'*G)*(lambda2/(s1+s2)*M'*Y-c1*G'*e2);
            u2 = inv(lambda3/(s1+s2)*D*M+c4*I+G'*G+c2*H'*H)*(lambda4/(s1+s2)*M'*Y-c1*H'*e1);
           
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