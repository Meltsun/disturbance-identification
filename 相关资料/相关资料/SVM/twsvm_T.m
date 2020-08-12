clear all
close all
clc
tic;
%% 初始化参数
n1 = 50;               %类别
n2 = 50;              %每类的样本数
S = n1+n2;             %样本总数
T = 6;               %训练样本：测试样本 = A：10-A,可调
M = 100;             %迭代次数
W = (1-T/10)*M*100;%m*n;  %测试总数
k = 0;               %用来避免奇异值，一般为0
%可通过果蝇算法得到C1和C2
C1 = 10;             %成本约束参数
C2 = 8;
% Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出
%% 加载数据
load alldatapot2_lable.mat
data = alldatapot2_lable;
%data = xlsread('iris.xlsx');

x1 = data(51:100,[3,4,5,15,24,26]);   %第一类
%y1 = ones(1,50);      %添加标签+1
x2 = data(101:150,[3,4,5,15,24,26]);%第二类
%y2 = -ones(1,50);     %添加标签-1


t1 = 0;
t2 = 0;

e1 = ones(50*T/10,1);
e2 = ones(50*T/10,1);


%trD为训练集，teD为测试集
%对十折来说每类有45个训练数据，5个测试数据
for i = 1:M    
    
    [trD1,teD1] = F_CV(x1,T);
    [trD2,teD2] = F_CV(x2,T);
    %全一向量
  
     
    
    H = [trD1,e1];
    G = [trD2,e2];
    P = [trD1,e1];
    Q = [trD2,e2];
 

    %训练样本
    Xtrain = [trD1',trD2']; %二维用来画图
    %测试样本
    Xtest  = [teD1',teD2'];
    %% 二次规划来求解问题
    %得到w和b
    [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,C1,C2,n1,n2,k,T,options);%n = 2,m = 45
    %% 画图
   %{
    %特征超过两个时没有办法画图
    figure
    %训练点
    plot(trD1(:,1)',trD1(:,2)','b*',trD2(:,1)',trD2(:,2)','k+');  %画图，两堆点
    axis([min(Xtrain(1,:)) max(Xtrain(1,:)) min(Xtrain(2,:)) max(Xtrain(2,:))]);  %设置坐标轴范围
    hold on
    plot(teD1(:,1)',teD1(:,2)','*c',teD2(:,1)',teD2(:,2)','+m');  %画图，两堆点
    axis([min(Xtrain(1,:)) max(Xtrain(1,:)) min(Xtrain(2,:)) max(Xtrain(2,:))]);  %设置坐标轴范围    
    hold on;
    %测试点
    
    jg1 = (max(Xtrain(1,:))-min(Xtrain(1,:)))/180;
    jg2 = (max(Xtrain(2,:))-min(Xtrain(2,:)))/180; 
    [x1_,x2_] = meshgrid(min(Xtrain(1,:)):jg1:max(Xtrain(1,:)),min(Xtrain(2,:)):jg2:max(Xtrain(2,:)));  %x1和x2都是181*181的矩阵
    [rows,cols] = size(x1_);  
    nt = rows*cols;                  
    Xt = [reshape(x1_,1,nt);reshape(x2_,1,nt)];
    %前半句reshape(x1_,1,nt)是将x1_转成1*（181*181）的矩阵，所以xt是2*（181*181）的矩阵
    %reshape函数重新调整矩阵的行、列、维数

    result1 = w1'*Xt+b1;
    result2 = w2'*Xt+b2;
  
    %画曲线的等高线图
    Yd1 = reshape(result1,rows,cols);
    Yd2 = reshape(result2,rows,cols);
    contour(x1_,x2_,Yd1,[0,0],'ShowText','off','linecolor','r');%画等高线
    contour(x1_,x2_,Yd2,[0,0],'ShowText','off','linecolor','g');
    title('twsvm分类结果图');
    x1_=xlabel('X轴'); 
    x2_=ylabel('Y轴'); 
     %}
    %% 测试
    result1_ = (w1'*Xtest+b1)/sqrt(w1'*w1);
    result2_ = (w2'*Xtest+b2)/sqrt(w2'*w2);
 
    [~,p] = size(result1_);
    
    for j = 1:n1*(10-T)/10
        if(abs(result1_(j)) > abs(result2_(j)))
            t1 = t1+1; 
        end
    end
        %hold on
    for j = n1*(10-T)/10+1:p
        if(abs(result1_(j)) < abs(result2_(j)))
            t2 = t2+1;
        end
    end   
end  
   
t = t1+t2;
Accuracy = 1-t/W;%这里个数可能要改
toc;