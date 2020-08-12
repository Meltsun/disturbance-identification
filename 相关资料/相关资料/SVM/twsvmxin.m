%先把爬分出来，剩余的有向无环图
clc
clear all
close all

%% 初始化参数
m = 4;                     %类别数
n = 6;                     %特征数
t = 10;                    %基于模糊隶属度压缩的个数
T = 8;                     %交叉验证的比例。例如6：4
n1_ = 50;                  %第一类的样本数
n2_ = 50;                  %第一类的样本数
n1 = n1_*T/10-t;           %第一类的训练样本数
n2 = n2_*T/10-t;           %第二类的训练样本数
s_ = n1_*(10-T)/10;        %某一类的测试样本数
k = 0.1;                     %防止矩阵求逆奇异值的出现
s = 1;                     %高斯核函数中的方差
c1 = 8.9944;               %调整参数
c2 = 7.5212;               %调整参数
DD = 100;
%% 加载数据
load alldatapot3_lable.mat
Data = alldatapot3_lable;
%把爬的数据放在最下面
Data = Data(:,1:end-1);
Data1 = Data(151:200,:);
Data(151:200,:) = [];
Data = [Data;Data1];
qian = [ones(n1_,1)*1;
        ones(n1_,1)*2;
        ones(n1_,1)*3;
        ones(n1_,1)*4;
        ones(n1_,1)*5];
Data = [Data,qian];   %浇敲压无爬
Data1 = feature_end(Data);           %后面新加的一个特征它反映离散点间的离散程度
qian = Data(:,end);                  %最后一列
data1 = mapminmax(Data(:,1:44),0,1); %特征归一化
data = [data1,Data1,qian];           %重新组合

tz = randi(44,1,5);                  %随机获得5个特征
tz = [tz,45];                        %把最后一个特征填进去
 %tz = [40,8,32,6,21,45]; %pot1 76%  
%tz = [25,43,9,2,20,45]; %pot2 90%
%tz = [1,35,18,28,43,45];%pot2 84% 除了爬其他全对
%tz = [3,25,7,17,9,45];  %pot2 96%
%tz = [35,9,14,28,11,45];%pot2 94% 0.026s
tz = [22,23,42,17,7,45];%pot3 89% 0.025s
%tz = [39,43,11,34,36,45];%pot1 78% 0.024s
%tz = [18,39,10,19,30,45];%pot3
%tz = [15,17,3,24,30,45];
tz = [1,5,13,15,27,45]; %pot3

%tz = [2,5,25,19,28,45];

A1 = data(1:50,tz);                  %截取第一类
B2 = data(51:100,tz);                %截取第二类
C3 = data(101:150,tz);               %截取第三类
D4 = data(151:200,tz);               %截取第四类
E5 = data(201:250,tz);               %截取第五类
%% 训练集和测试集
for g = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A的训练集和测试集和下标
[trDB,teDB,tr2] = F_CV(B2,T);            %B的训练集和测试集和下标
[trDC,teDC,tr3] = F_CV(C3,T);            %C的训练集和测试集和下标
[trDD,teDD,tr4] = F_CV(D4,T);            %D的训练集和测试集和下标
[trDE,teDE,tr5] = F_CV(E5,T);            %E的训练集和测试集和下标

trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%训练集压缩汇总
 c = 0;                                 %计数
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
            trD2 = trD((j-1)*n1+1:j*n1,:); %第二类

%基于twsvm线性核函数方法
            e1 = ones(n1,1);
            e2 = ones(n2,1);
            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];
            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);
 c = c+1;
            %权值汇总
            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
        end
    end
    %% 分类器汇总
    flq = [1,2;1,3;1,4;2,3;2,4;3,4];
    yxwht = [1,4;2,4;1,3;3,4;2,3;1,2];
    for i = 1:c
        for j = 1:c
           if(yxwht(i,:) == flq(j,:))
               lib(i) = j;
           end
        end
    end       
    teD_z = [teDA;teDB;teDC;teDD];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;];
    [row,col] = size(teD_z);   
    tic;
    %有向无环图
     for i = 1:row
        result1 = (W1(lib(1),:)*teD_z(i,:)'+B1(lib(1)))/(W1(lib(1),:)*W1(lib(1),:)');
        result2 = (W2(lib(1),:)*teD_z(i,:)'+B2(lib(1)))/(W2(lib(1),:)*W2(lib(1),:)');
        if(abs(result1) > abs(result2))     
            result1 = (W1(lib(2),:)*teD_z(i,:)'+B1(lib(2)))/(W1(lib(2),:)*W1(lib(2),:)');
            result2 = (W2(lib(2),:)*teD_z(i,:)'+B2(lib(2)))/(W2(lib(2),:)*W2(lib(2),:)'); 
            if(abs(result1) > abs(result2))  
                result1 = (W1(lib(4),:)*teD_z(i,:)'+B1(lib(4)))/(W1(lib(4),:)*W1(lib(4),:)');
                result2 = (W2(lib(4),:)*teD_z(i,:)'+B2(lib(4)))/(W2(lib(4),:)*W2(lib(4),:)'); 
                if(abs(result1) > abs(result2))  
                      jg(g,i) = flq(lib(4),2);
                else
                      jg(g,i) = flq(lib(4),1);
                end
            else
                result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
                result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
                if(abs(result1) > abs(result2))  
                      jg(g,i) = flq(lib(5),2);
                else
                      jg(g,i) = flq(lib(5),1);
                end
            end
        else
            result1 = (W1(lib(3),:)*teD_z(i,:)'+B1(lib(3)))/(W1(lib(3),:)*W1(lib(3),:)');
            result2 = (W2(lib(3),:)*teD_z(i,:)'+B2(lib(3)))/(W2(lib(3),:)*W2(lib(3),:)'); 
            if(abs(result1) > abs(result2))  
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
               if(abs(result1) > abs(result2))  
                  jg(g,i) = flq(lib(5),2);
               else
                  jg(g,i) = flq(lib(5),1);
               end 
            else
               result1 = (W1(lib(6),:)*teD_z(i,:)'+B1(lib(6)))/(W1(lib(6),:)*W1(lib(6),:)');
               result2 = (W2(lib(6),:)*teD_z(i,:)'+B2(lib(6)))/(W2(lib(6),:)*W2(lib(6),:)'); 
                if(abs(result1) > abs(result2))  
                  jg(g,i) = flq(lib(6),2);
               else
                  jg(g,i) = flq(lib(6),1);
               end 
            end
        end
     end
ll = jg(g,:)';
Accuracy(g,1) = 1-length(find((ll-biaoqian)~=0))/row;
toc;
end
jg = jg';
 %% 画图
figure(1)
plot(jg,'og')
hold on
plot(biaoqian,'r*');
legend('预测标签','实际标签')
title('twsvm预测分类与实际类别比对','fontsize',12)
ylabel('类别标签','fontsize',12)
xlabel('样本数目','fontsize',12)
%% 分类正确率
A_j = length(find(jg(1:s_,:) == 1))/(DD*s_);             %浇的正确率
A_q = length(find(jg(s_+1:2*s_,:) == 2))/(DD*s_);        %敲的正确率
A_y = length(find(jg(2*s_+1:3*s_,:) == 3))/(DD*s_);      %压的正确率
A_n = length(find(jg(3*s_+1:4*s_,:) == 4))/(DD*s_);      %无的正确率

A_z = sum(Accuracy)/DD;                                  %平均正确率


sprintf('浇的测试准确率=%0.2f',A_j)
sprintf('敲的测试准确率=%0.2f',A_q)
sprintf('压的测试准确率=%0.2f',A_y)
sprintf('无的测试准确率=%0.2f',A_n)
sprintf('总的测试准确率=%0.2f',A_z)
    

 teD = [teDA;teDB;teDC;teDD;teDE];
for i = 1:row
    jl1 = sqrt(sum((repmat(teD(i,:),n1*5,1)-trD).^2,2));
    jl(i,1) = sum(jl1(1:n1));
    jl(i,2) = sum(jl1(n1+1:2*n1));
    jl(i,3) = sum(jl1(2*n1+1:3*n1));
    jl(i,4) = sum(jl1(3*n1+1:4*n1));
    jl(i,5) = sum(jl1(4*n1+1:5*n1));
    
end  
    