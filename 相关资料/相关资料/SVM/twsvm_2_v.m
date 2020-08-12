%二分类投票
clear all
close all
clc
tic;
%% 初始化参数
m  = 5;                %类别
n1 = 50;              
n2 = 50;              %每类的样本数
S = n1+n2;             %样本总数
T = 6;               %训练样本：测试样本 = A：10-A,可调
M = 100;             %迭代次数
W = (1-T/10)*M*100;%m*n;  %测试总数
k = 0.0001;               %用来避免奇异值，一般为0
s = 1;
%可通过果蝇算法得到C1和C2
C1 = 9.9316;             %成本约束参数
C2 = 8.0771;
% Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出
%% 加载数据
load alldatapot2_lable.mat
data = alldatapot2_lable;
c = 0;%计数
W1 = [];
W2 = [];
B1 = [];
B2 = [];
Y = [1,2,3,4,5];
tedte = {};
%迭代100次，再看。。。。
%{
for i = 1:m-1
    for j = i+1:m
        x1 = data((i-1)*n1+1:i*n1,[3,4,5,15,24,26]);   %第一类
        y1 = Y(i)*ones(n1,1);                    %添加标签
        x2 = data((j-1)*n2+1:j*n2,[3,4,5,15,24,26]); %第二类
        y2 = Y(j)*ones(n2,1);                    %添加标签
        
        t1 = 0;
        t2 = 0;

        e1 = ones(n1*T/10,1);
        e2 = ones(n2*T/10,1);
        %trD为训练集，teD为测试集
      
        [trD1,teD1] = F_CV(x1,T);
        [trD2,teD2] = F_CV(x2,T);
            
        H = [trD1,e1];
        G = [trD2,e2];
        P = [trD1,e1];
        Q = [trD2,e2];
            
        [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,C1,C2,n1,n2,k,options);
        
        c = c+1;
        
        W1(c,:) = w1';
        W2(c,:) = w2';
        B1(c)   = b1;
        B2(c)   = b2;
        
        
        
        te1 = Y(i)*ones(size(teD1,1),1);%加标签
        te2 = Y(j)*ones(size(teD2,1),1);%加标签
        
        teDte1 = [teD1,te1];
        teDte2 = [teD2,te2];
        
        tedte{c,1} = teDte1;
        tedte{c,2} = teDte2;
        
         
               
    end
end
%}


    A1 = data(1:50,[3,4,5,15,24,26]);
    B2 = data(51:100,[3,4,5,15,24,26]);
    C3 = data(101:150,[3,4,5,15,24,26]);
    D4 = data(151:200,[3,4,5,15,24,26]);
    E5 = data(201:250,[3,4,5,15,24,26]);
    [trDA,teDA] = F_CV(A1,T);
    [trDB,teDB] = F_CV(B2,T);
    [trDC,teDC] = F_CV(C3,T);
    [trDD,teDD] = F_CV(D4,T);
    [trDE,teDE] = F_CV(E5,T);
    trD = [trDA;trDB;trDC;trDD;trDE];
    
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1*T/10+1:i*n1*T/10,:);   %第一类
            y1 = Y(i)*ones(n1,1);                    %添加标签
            trD2 = trD((j-1)*n2*T/10+1:j*n2*T/10,:); %第二类
            y2 = Y(j)*ones(n2,1);                    %添加标签

            t1 = 0;
            t2 = 0;

            e1 = ones(n1*T/10,1);
            e2 = ones(n2*T/10,1);
            %trD为训练集，teD为测试集

            %[trD1,teD1] = F_CV(x1,T);
            %[trD2,teD2] = F_CV(x2,T);

            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];

            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,C1,C2,n1,n2,k,T,options);

            c = c+1;

            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
        end
    end

            
    teD_z = [teDA,ones(n1*(10-T)/10,1);teDB,ones(n1*(10-T)/10,1)*2;teDC,ones(n1*(10-T)/10,1)*3;teDD,ones(n1*(10-T)/10,1)*4;teDE,ones(n1*(10-T)/10,1)*5];
    %{
    %% 测试集汇总
    teD_z = [];
    for i = 1:c
        for j = 1:2
          teD_z = [teD_z;tedte{i,j}];  
        end
    end
    %}
    %% 分类器汇总
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    [row,col] = size(teD_z);
    %% 投票
    for i = 1:row
        for k = 1:c
            result1 = (W1(k,:)*teD_z(i,1:end-1)'+B1(k))/sqrt(W1(k,:)*W1(k,:)');
            result2 = (W2(k,:)*teD_z(i,1:end-1)'+B2(k))/sqrt(W2(k,:)*W2(k,:)');
            if(abs(result1) > abs(result2))
                    te_(i,k) = flq(k,2);
                else
                    te_(i,k) = flq(k,1);
            end 
            result1_(i,k) = abs(result1);
            result2_(i,k) = abs(result2);
            result(i,k) = (abs(result1)-abs(result2))>=0;

        end
    end
    for i =1:row
        table = tabulate(te_(i,:));
        [~,idx] = max(table(:,2));
        vote_y(i,1) = idx;
    end
Accuracy = 1-length(find((vote_y-teD_z(:,end))~=0))/row;


