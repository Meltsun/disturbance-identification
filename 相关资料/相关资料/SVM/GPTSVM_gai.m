clear all
close all
clc
%[1,15]两类特征，c1 = 7，c2 = 7，lambda1 = 4，lambda2 = 1     82

%投影孪生支持向量机

%% 初始化参数
m  = 5;                 %类别数
n = 6;
n1 = 20;                %第一类的样本数
n2 = 20;                %第二类的样本数
n1_ = 50;
n2_ = 50;
S  = n1+n2;             %样本总数
T  = 6;                 %训练样本：测试样本 = A：10-A,可调
DD  = 1;
s1 = n1*T/10;            %训练样本1
s2 = n2*T/10;            %训练样本2
s_ = n1_*(10-T)/10;
k = 0;
s = 1;
%惩罚参数
%第一个点 %第三个点   %第二个点
c1 = 8.25;   %c1 = 8.3;  %c1 = 7.66;      5      6      7     8     10
c2 = 6.8;   %c2 = 5.53; %c2 = 8.52;      2      7      7     7     9
%调整参数
c3 = 0;
c4 = 0;
lambda1 = 4;                        % 4
lambda2 = 1;                        % 1

%% 获取数据
load alldatapot1_lable.mat
Data = alldatapot1_lable;
Data1 = feature_end(Data);
%data = Z_mean(Data);%归一化处理
qian = Data(:,end);
data1 = mapminmax(Data(:,1:44),0,1);
data = [data1,Data1,qian];
%data = Data;
%data = xlsread('iris.xlsx');
%获取每类数据
%tz = [3,4,5,15,24,26]; 
%tz = [3,4,5,16,17,18];
%tz = [33,45];
tz = randi(44,1,6);
%tz = [10,2,4,11,24,44];%pot3这个很好
%tz = [13,15,32,10,2,9];%pot2这个很好
%tz = [28,8,35,32,24,2];%pot1这个还可以
%tz = [18,2,15,1,5,23];
tz = [35,13,38,2,1];%2
%tz = [14,2,44,32,35,25];%3和1
%tz = [23,36,26,6,31,1];
%tz = [12,9,42,23,7];

tz = [tz,45];
%{
A1 = data(1:40,tz);
B2 = data(41:80,tz);
C3 = data(81:120,tz);
D4 = data(121:160,tz);
E5 = data(161:200,tz);
%}
A1 = data(1:50,tz);
B2 = data(51:100,tz);
C3 = data(101:150,tz);
D4 = data(151:200,tz);
E5 = data(201:250,tz);
%{
gs1 = 0;
gs2 = 0;
gs3 = 0;
gs4 = 0;
gs5 = 0;
%}
for g = 1:DD
    
    %获得每组的训练集和测试集
    %trD是训练集，teD是测试集
    [trDA,teDA] = F_CV(A1,T);
    [trDB,teDB] = F_CV(B2,T);
    [trDC,teDC] = F_CV(C3,T);
    [trDD,teDD] = F_CV(D4,T);
    [trDE,teDE] = F_CV(E5,T);
    %训练集汇总   
    %trD = [trDA;trDB;trDC;trDD;trDE];
    trD = yasuo(trDA,trDB,trDC,trDD,trDE,10);
   %  trD = yasuo(trDA,trDB,trDC,trDE);
    c = 0;%计数
    
    %训练分类器
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
            trD2 = trD((j-1)*n2+1:j*n2,:); %第二类
            %{
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
            %}
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%         
% 
%             H = [trD1,e1];
%             G = [trD2,e2];
%             P = [trD1,e1];
%             Q = [trD2,e2];
% 
%             [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,s1,s2,k,T,options);
         
%            
%             A = trD1;
%             B = trD2;
%             C = [A',B']';
%             %训练集
%             S = [rbf(A,C',s),e1];
%             R = [rbf(B,C',s),e2];
%             L = [rbf(A,C',s),e1];
%             N = [rbf(B,C',s),e2];
%             [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,c1,c2,50,50,k,T,options);
           
            X  = [trD1;trD2];
            s1 = 20;s2 = 20;
            e1 = ones(s1,1);
            e2 = ones(s2,1);
            e  = ones(s1+s2,1);
            Y = [ones(s1,1);-ones(s2,1)];
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
   % teD_z = [teDA; teDB;teDC;teDE];
    biaoqian = [ones(n1_*(10-T)/10,1)*1;
                ones(n1_*(10-T)/10,1)*2;
                ones(n1_*(10-T)/10,1)*3;
                ones(n1_*(10-T)/10,1)*4;
                ones(n1_*(10-T)/10,1)*5];
    %% 分类器汇总
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
   %flq = [1,2;1,3;1,5;2,3;2,5;3,5;];
    [row,col] = size(teD_z);
   tic; 
    %第一步
    trA_mean = mean(trDA,1);
    trB_mean = mean(trDB,1);
    trC_mean = mean(trDC,1);
    trD_mean = mean(trDD,1);
    trE_mean = mean(trDE,1);
    for i = 1:row
        %{
        d1 = norm(teD_z(i,:)-trA_mean);
        d2 = norm(teD_z(i,:)-trB_mean);
        d3 = norm(teD_z(i,:)-trC_mean);
        d4 = norm(teD_z(i,:)-trD_mean);
        d5 = norm(teD_z(i,:)-trE_mean);
        
        D1 = trDA(find(lsd(trDA,0))>0.9,:);
        D2 = trDB(find(lsd(trDB,0))>0.9,:);
        D3 = trDC(find(lsd(trDC,0))>0.9,:);
        D4 = trDD(find(lsd(trDD,0))>0.9,:);
        D5 = trDE(find(lsd(trDE,0))>0.9,:);
        
        d1 = norm(teD_z(i,:)-mean(D1,1));
        d2 = norm(teD_z(i,:)-mean(D2,1));
        d3 = norm(teD_z(i,:)-mean(D3,1));
        d4 = norm(teD_z(i,:)-mean(D4,1));
        d5 = norm(teD_z(i,:)-mean(D5,1));
        %}
         S1 = zeros(n,n);
         S2 = zeros(n,n);
         S3 = zeros(n,n);
         S4 = zeros(n,n);
         S5 = zeros(n,n);
         for k = 1:s1
             S1 = S1+(trDA(k,:)-trA_mean)'*(trDA(k,:)-trA_mean);
         end
         for k = 1:s1
             S2 = S2+(trDB(k,:)-trB_mean)'*(trDB(k,:)-trB_mean);
         end
         for k = 1:s1
             S3 = S3+(trDC(k,:)-trC_mean)'*(trDC(k,:)-trC_mean);
         end
        
         for k = 1:s1
             S4 = S4+(trDD(k,:)-trD_mean)'*(trDD(k,:)-trD_mean);
         end
         
         for k = 1:s1
             S5 = S5+(trDE(k,:)-trE_mean)'*(trDE(k,:)-trE_mean);
         end
         d1 = sqrt((teD_z(i,:)-trA_mean)*inv(S1)*(teD_z(i,:)-trA_mean)');
         d2 = sqrt((teD_z(i,:)-trB_mean)*inv(S2)*(teD_z(i,:)-trB_mean)');
         d3 = sqrt((teD_z(i,:)-trC_mean)*inv(S3)*(teD_z(i,:)-trC_mean)');
         d4 = sqrt((teD_z(i,:)-trD_mean)*inv(S4)*(teD_z(i,:)-trD_mean)');
         d5 = sqrt((teD_z(i,:)-trE_mean)*inv(S5)*(teD_z(i,:)-trE_mean)');
         [~,px] = sort([d1,d2,d3,d4,d5]);
         %[~,px] = sort([d1,d2,d3,d5]);
        %pxjg(i,:) = sort(px(1:3));
          pxjg(i,:) = sort(px(1:2));
          pxjg_(i,:) = px;
         % pxjg(find(pxjg == 4))=5;
    end 
    %{
    gs1 = gs1+length(find(pxjg(1:20,:) == 1));
    gs2 = gs2+length(find(pxjg(21:40,:) == 2));
    gs3 = gs3+length(find(pxjg(41:60,:) == 3));
    gs4 = gs4+length(find(pxjg(61:80,:) == 4));
    gs5 = gs5+length(find(pxjg(81:100,:) == 5));
    %}
    %第二步
    for i = 1:row
        for j = 1:c
            if((flq(j,1) == pxjg(i,1))&&(flq(j,2) == pxjg(i,2)))
                jg_(i,1) = j;
               
            end
            %{
            if((flq(j,1) == pxjg(i,1))&&(flq(j,2) == pxjg(i,3)))
                jg_(i,2) = j;
                
            end
            if((flq(j,1) == pxjg(i,2))&&(flq(j,2) == pxjg(i,3)))
                jg_(i,3) = j;
               
            end
            %}
        end
    end
    for i = 1:row
        for k = 1:1
            result1 = (W1(jg_(i,k),:)*teD_z(i,:)'+B1(jg_(i,k)))/(W1(jg_(i,k),:)*W1(jg_(i,k),:)');
            result2 = (W2(jg_(i,k),:)*teD_z(i,:)'+B2(jg_(i,k)))/(W2(jg_(i,k),:)*W2(jg_(i,k),:)');
%              result1 = W1(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B1(jg_(i,k));
%              result2 = W2(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B2(jg_(i,k));
            if(abs(result1) > abs(result2))
               te_(i,k) = flq(jg_(i,k),2);
            else
               te_(i,k) = flq(jg_(i,k),1);
            end 
            result(i,1) = abs(result1);
            result(i,2) = abs(result2);
            
        end
    end
    %{   
    for i = 1:row
        jg = pxjg(i,:);
        jg1 = [jg(1),jg(2)];
        jg2 = [jg(1),jg(3)];
        jg3 = [jg(2),jg(3)];
        b = [1,1];
        for j = 1:c
            if((flq(j,1) == jg1(1))&&(flq(j,2) == jg1(2)))
                jg1_ = j;
                break;
            end
        end
            
        
        for j = 1:c
            if((flq(j,1) == jg2(1))&&(flq(j,2) == jg2(2)))
                jg2_ = j;
                break;
            end  
        end
        for j = 1:c
            if((flq(j,1) == jg3(1))&&(flq(j,2) == jg3(2)))
                jg3_ = j;
                break;
            end  
            
        end
        
        jg_(i,:) = [jg1_,jg2_,jg3_];  
    end
   
    for i = 1:row
        for k = 1:3
            result1 = W1(jg_(i,k),:)*(teD_z(i,:)-A_mean)';
            result2 = W2(jg_(i,k),:)*(teD_z(i,:)-B_mean)';
            if(abs(result1) > abs(result2))
               te_(i,k) = flq(jg_(i,k),2);
            else
               te_(i,k) = flq(jg_(i,k),1);
            end 
            result(i,k) = abs(result1)-abs(result2);
            
        end
    end
 %}   
    for i =1:row
        table = tabulate(te_(i,:));
        [~,idx] = max(table(:,2));
        vote_y(i,g) = idx;
    end
    Accuracy(g,1) = 1-length(find((vote_y(:,g)-biaoqian)~=0))/row;  
end
A_j = length(find(vote_y(1:s_,:) == 1))/(DD*s_);
A_q = length(find(vote_y(s_+1:2*s_,:) == 2))/(DD*s_);
A_y = length(find(vote_y(2*s_+1:3*s_,:) == 3))/(DD*s_);
A_p = length(find(vote_y(3*s_+1:4*s_,:) == 4))/(DD*s_);
A_n = length(find(vote_y(4*s_+1:5*s_,:) == 5))/(DD*s_);
%A_n = length(find(vote_y(3*s_+1:4*s_,:) == 5))/(DD*s_);
A_N = sum(Accuracy)/DD;

figure(1)
plot(vote_y,'og')
hold on
plot(biaoqian,'r*');
legend('预测标签','实际标签')
title('twsvm预测分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
%{
gs1_ = gs1/(20*DD);
gs2_ = gs2/(20*DD);
gs3_ = gs3/(20*DD);
gs4_ = gs4/(20*DD);
gs5_ = gs5/(20*DD);
%}
toc;

