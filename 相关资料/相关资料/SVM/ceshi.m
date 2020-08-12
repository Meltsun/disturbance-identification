%------------主函数----------------
%这个是svm分类
clear all;
close all;
clc
C = 10;  %成本约束参数
kertype = 'linear';  %线性核
%{ 
%①------数据准备
n = 30;
%randn('state',6);   %指定状态，一般可以不用
x1 = randn(2,n);    %2行N列矩阵，元素服从正态分布
y1 = ones(1,n);       %1*N个1
x2 = 4+randn(2,n);   %2*N矩阵，元素服从正态分布且均值为5，测试高斯核可x2 = 3+randn(2,n); 
y2 = -ones(1,n);      %1*N个-1
%}
%①-------数据准备 
load('D:\各种代码\模式识别\数据特征提取\Data\50组数据得到的特征值\用于模式识别\最后的44个特征\alldatapot2_lable.mat');
data = alldatapot2_lable;
n = 50;
f = [16,17,18,38,39,40];
t = 1;
for i = 1:size(f,2)-1
    for j = i+1:size(f,2)
        x1 = [data(151:200,f(i))';data(151:200,f(j))'];%2*50
        y1 = ones(1,n);
        x2 = [data(201:250,f(i))';data(201:250,f(j))'];%2*50
        y2 = -ones(1,n);
        figure(t);  %创建一个用来显示图形输出的一个窗口对象
        
        %    横坐标   纵坐标
        plot(x1(1,:),x1(2,:),'bs',x2(1,:),x2(2,:),'k+');  %画图，两堆点

        min1 = min([data(151:200,f(i))',data(201:250,f(i))']);
        max1 = max([data(151:200,f(i))',data(201:250,f(i))']);
        min2 = min([data(151:200,f(j))',data(201:250,f(j))']);
        max2 = max([data(151:200,f(j))',data(201:250,f(j))']);
        axis([min1,max1,min2,max2]);  %设置坐标轴范围
        hold on;    %在同一个figure中画几幅图时，用此句

        %②-------------训练样本
        X = [x1,x2];        %训练样本2*n矩阵，n为样本个数，d为特征向量个数
        Y = [y1,y2];        %训练目标1*n矩阵，n为样本个数，值为+1或-1



        svm = svmTrain(X,Y,kertype,C);  %训练样本
        plot(svm.Xsv(1,:),svm.Xsv(2,:),'ro');   %把支持向量标出来




        %③-------------测试
        [x1,x2] = meshgrid(min1:(max1-min1)/180:max1,min2:(max2-min2)/180:max2);  %x1和x2都是181*181的矩阵
        [rows,cols] = size(x1);  
        nt = rows*cols;                  
        Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];
        %前半句reshape(x1,1,nt)是将x1转成1*（181*181）的矩阵，所以xt是2*（181*181）的矩阵
        %reshape函数重新调整矩阵的行、列、维数
        Yt = ones(1,nt);
        %对所有点进行测试
        result = svmTest(svm, Xt, Yt, kertype);

        %④--------------画曲线的等高线图
        Yd = reshape(result.Y,rows,cols);
        contour(x1,x2,Yd,[0,0],'ShowText','off');%画等高线
        title('svm分类结果图');
        x1=xlabel('X轴'); 
        x2=ylabel('Y轴');
        t = t+1;
    end
end

