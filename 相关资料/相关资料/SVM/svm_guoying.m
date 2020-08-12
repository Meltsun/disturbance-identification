clc
clear all
close all

%% 初始化参数
m = 5;                     %类别数
n = 6;                     %特征数
t = 10;                    %基于模糊隶属度压缩的个数
T = 7;                     %交叉验证的比例。例如6：4
n1_ = 50;                  %第一类的样本数
n2_ = 50;                  %第一类的样本数
n1 = n1_*T/10-t;           %第一类的训练样本数
n2 = n2_*T/10-t;           %第二类的训练样本数
s_ = n1_*(10-T)/10;        %某一类的测试样本数
k = 0;                     %防止矩阵求逆奇异值的出现
s = 1;                     %高斯核函数中的方差
DD = 100;
%% 获取数据
load alldatapot2_lable.mat
Data = alldatapot2_lable;
Data1 = feature_end(Data);           %后面新加的一个特征它反映离散点间的离散程度
qian = Data(:,end);                  %最后一列
data1 = mapminmax(Data(:,1:44),0,1); %特征归一化
data = [data1,Data1,qian];           %重新组合
tz = [3,25,7,17,9,45];  %pot2 96%
A1 = data(1:50,tz);                  %截取第一类
B2 = data(51:100,tz);                %截取第二类
C3 = data(101:150,tz);               %截取第三类
D4 = data(151:200,tz);               %截取第四类
E5 = data(201:250,tz);               %截取第五类
for h = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A的训练集和测试集和下标
[trDB,teDB,tr2] = F_CV(B2,T);            %B的训练集和测试集和下标
[trDC,teDC,tr3] = F_CV(C3,T);            %C的训练集和测试集和下标
[trDD,teDD,tr4] = F_CV(D4,T);            %D的训练集和测试集和下标
[trDE,teDE,tr5] = F_CV(E5,T);            %E的训练集和测试集和下标

trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%训练集压缩汇总
teD_z = [teDA;teDB;teDC;teDD;teDE];

%*** 设置参数
maxgen = 50; %迭代次数
sizepop = 50; %种群规模

%[ yy, Xbest, Ybest  ] = FOA( maxgen, sizepop );
X_axis = 5 * rand()+5;
Y_axis = 5 * rand()+5;

    %*** 果蝇寻优开始，利用嗅觉寻找食物
for i=1 : sizepop

    %*** 赋予果蝇个体利用嗅觉搜寻食物之随机方向与距离
    X(i) = X_axis + 2 * rand() - 1;
    Y(i) = Y_axis + 2 * rand() - 1;

    %*** 由于无法得知食物位置，因此先估计与原点的距离（Dist），再计算味道浓度判定值（S），此值为距离的倒数
    D(i) = (X(i)^2 + Y(i)^2)^0.5;
    S(i) = 1 / D(i);

    %*** 味道浓度判定值（S）代入味道浓度判定函数（或称为Fitness function），以求出该果蝇个体位置的味道浓度（Smell(i))
   % Smell(i) = Fitness(S(i));
    Smell(i) = PSO_twsvm_fitness(trDA,trDB,trDC,trDD,trDE,X(i),Y(i));

end

%*** 找出此果蝇群里中味道浓度最低的果蝇（求极小值）
% [bestSmell ,bestindex] = min(Smell);
[bestSmell ,bestindex] = max(Smell);
%*** 保留最佳味道浓度值与x，y的坐标，此时果蝇群里利用视觉往该位置飞去
X_axis = X(bestindex);
Y_axis = Y(bestindex);
Smellbest = bestSmell;

%*** 果蝇迭代寻优开始
for g=1 : maxgen

    %*** 赋予果蝇个体利用嗅觉搜寻食物的随机方向和距离
    for i=1 : sizepop
        X(i) = X_axis + 2 * rand() - 1;
        Y(i) = Y_axis + 2 * rand() - 1;

        %*** 由于无法得知食物位置，因此先估计与原点的距离（Dist），再计算味道浓度判定值（S），此值为距离的倒数
        D(i) = (X(i)^2 + Y(i)^2)^0.5;
        S(i) = 1 / D(i);

        %*** 味道浓度判定值（S）代入味道浓度判定函数，以求出该果蝇个体位置的味道浓度（Smell(i))
        Smell(i) = PSO_twsvm_fitness(trDA,trDB,trDC,trDD,trDE,X(i),Y(i));

    end;

    %*** 找出此果蝇群里中味道浓度最低的果蝇（求极小值）
    [bestSmell bestindex] = max(Smell);

    %*** 判断味道浓度是否优于前一次迭代味道浓度，若是则保留最佳味道浓度值与x，y的坐标，此时果蝇群体利用视觉往该位置飞去
    if bestSmell > Smellbest
        X_axis = X(bestindex);
        Y_axis = Y(bestindex);
        Smellbest = bestSmell;
    end;

    %*** 每次最优Semll值记录到yy数组中，并记录最优迭代坐标
    yy(g) = Smellbest;
    Xbest(g) = X_axis;
    Ybest(g) = Y_axis;

end;
Accuracy_train(h,1) = Smellbest;
Xbest(h) = X_axis;
Ybest(h) = Y_axis;
c1 = X_axis;
c2 = Y_axis;
 c = 0;                                 %计数
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %第一类 
            trD2 = trD((j-1)*n1+1:j*n1,:); %第二类

            %基于TLDM方法
%             X  = [trD1;trD2];
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             e  = ones(n1+n2,1);
%             Y = [ones(n1,1);-ones(n2,1)];
%             M  = [X,e];
%             D  = M'*(ones(n1+n2,n1+n2)-Y*Y');
%             H  = [trD1,e1];
%             G  = [trD2,e2];
%             Q  = lambda2/(n1+n2)*M'*Y;
%             P  = lambda1/(n1+n2).^2*D*M+H'*H; 
%             S  = lambda1/(n1+n2).^2*D*M+G'*G; 
%             [w1,b1,w2,b2] = TLDMtrain(H,Q,P,G,S,e1,e2,n1,n2,c1,c2);

            %基于twsvm线性核函数方法
            e1 = ones(n1,1);
            e2 = ones(n2,1);
            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];
            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);

            %基于twsvm高斯核函数方法
%             A = trD1;
%             B = trD2;
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             C = [A',B']';
%             S = [rbf(A,C',s),e1];
%             R = [rbf(B,C',s),e2];
%             L = [rbf(A,C',s),e1];
%             N = [rbf(B,C',s),e2];
%             [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,c1,c2,50,50,k,T);

            %基于LSTWSVM
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             H = [trD1,e1];
%             G = [trD2,e2];
%             I = ones(size(H'*H,1),size(H'*H,1));
%             u1 = -c1*inv(H'*H+c1*(G'*G)+c3*I)*G'*e2;
%             u2 = -c2*inv(G'*G+c2*(H'*H)+c4*I)*H'*e1;
%             w1 = u1(1:end-1);
%             b1 = u1(end);
%             w2 = u2(1:end-1);
%             b2 = u2(end);

            %分类器的个数
            c = c+1;
            %权值汇总
            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
        end
    end
    %% 分类器汇总
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    yxwht = [1,5;2,5;1,4;3,5;2,4;1,3;4,5;3,4;2,3;1,2];
    for i = 1:c
        for j = 1:c
           if(yxwht(i,:) == flq(j,:))
               lib(i) = j;
           end
        end
    end       
    teD_z = [teDA;teDB;teDC;teDD;teDE];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;
                ones(s_,1)*5];
    [row,col] = size(teD_z);     
    
    % 基于有向无环图 10个分类器
    for i = 1:row
        result1 = (W1(lib(1),:)*teD_z(i,:)'+B1(lib(1)))/(W1(lib(1),:)*W1(lib(1),:)');
        result2 = (W2(lib(1),:)*teD_z(i,:)'+B2(lib(1)))/(W2(lib(1),:)*W2(lib(1),:)');
        if(abs(result1) > abs(result2))
%                te_(i,k) = flq(lib(1),2);
            result1 = (W1(lib(2),:)*teD_z(i,:)'+B1(lib(2)))/(W1(lib(2),:)*W1(lib(2),:)');
            result2 = (W2(lib(2),:)*teD_z(i,:)'+B2(lib(2)))/(W2(lib(2),:)*W2(lib(2),:)'); 
            if(abs(result1) > abs(result2))   
               result1 = (W1(lib(4),:)*teD_z(i,:)'+B1(lib(4)))/(W1(lib(4),:)*W1(lib(4),:)');
               result2 = (W2(lib(4),:)*teD_z(i,:)'+B2(lib(4)))/(W2(lib(4),:)*W2(lib(4),:)');        
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(7),:)*teD_z(i,:)'+B1(lib(7)))/(W1(lib(7),:)*W1(lib(7),:)');
                  result2 = (W2(lib(7),:)*teD_z(i,:)'+B2(lib(7)))/(W2(lib(7),:)*W2(lib(7),:)');       
                  if(abs(result1) > abs(result2))
                      jg(h,i) = flq(lib(7),2);
                  else
                      jg(h,i) = flq(lib(7),1);
                  end
               else 
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');      
                  if(abs(result1) > abs(result2))
                      jg(h,i) = flq(lib(8),2);
                  else
                      jg(h,i) = flq(lib(8),1);
                  end
               end  
            else 
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(8),2);
                  else
                       jg(h,i) = flq(lib(8),1);
                  end
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');  
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(9),2);
                  else
                       jg(h,i) = flq(lib(9),1);
                  end
               end
            end
        else
            result1 = (W1(lib(3),:)*teD_z(i,:)'+B1(lib(3)))/(W1(lib(3),:)*W1(lib(3),:)');
            result2 = (W2(lib(3),:)*teD_z(i,:)'+B2(lib(3)))/(W2(lib(3),:)*W2(lib(3),:)'); 
            if(abs(result1) > abs(result2))
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)');  
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(8),2);
                  else
                       jg(h,i) = flq(lib(8),1);
                  end 
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(9),2);
                  else
                       jg(h,i) = flq(lib(9),1);
                  end  
               end
            else
               result1 = (W1(lib(6),:)*teD_z(i,:)'+B1(lib(6)))/(W1(lib(6),:)*W1(lib(6),:)');
               result2 = (W2(lib(6),:)*teD_z(i,:)'+B2(lib(6)))/(W2(lib(6),:)*W2(lib(6),:)');   
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(9),2);
                  else
                       jg(h,i) = flq(lib(9),1);
                  end  
               else
                  result1 = (W1(lib(10),:)*teD_z(i,:)'+B1(lib(10)))/(W1(lib(10),:)*W1(lib(10),:)');
                  result2 = (W2(lib(10),:)*teD_z(i,:)'+B2(lib(10)))/(W2(lib(10),:)*W2(lib(10),:)');     
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(10),2);
                  else
                       jg(h,i) = flq(lib(10),1);
                  end     
               end
            end
        end
    end
jg = jg';
Accuracy_test(h,1) = 1-length(find((jg(:,h)-biaoqian)~=0))/row;
end


% %*** 绘制迭代味道浓度与果蝇飞行路径趋势图
% figure(1);
% plot(yy);
% title('Optimization process', 'fontsize', 12);
% xlabel('Iteration Number', 'fontsize', 12);
% ylabel('Smell', 'fontsize', 12);
% figure(2);
% plot(Xbest, Ybest, 'b.');
% title('Fruit fly flying route', 'fontsize', 14);
% xlabel('X-axis', 'fontsize', 12);
% ylabel('Y-axis', 'fontsize', 12);
%% 分类正确率
A_j = length(find(jg(1:s_,:) == 1))/(DD*s_);             %浇的正确率
A_q = length(find(jg(s_+1:2*s_,:) == 2))/(DD*s_);        %敲的正确率
A_y = length(find(jg(2*s_+1:3*s_,:) == 3))/(DD*s_);      %压的正确率
A_p = length(find(jg(3*s_+1:4*s_,:) == 4))/(DD*s_);      %爬的正确率
A_n = length(find(jg(4*s_+1:5*s_,:) == 5))/(DD*s_);      %无的正确率
A_z = sum(Accuracy_test)/DD;                                      %平均正确率


sprintf('浇的测试准确率=%0.2f',A_j)
sprintf('敲的测试准确率=%0.2f',A_q)
sprintf('压的测试准确率=%0.2f',A_y)
sprintf('爬的测试准确率=%0.2f',A_p)
sprintf('无的测试准确率=%0.2f',A_n)
sprintf('总的测试准确率=%0.2f',A_z)