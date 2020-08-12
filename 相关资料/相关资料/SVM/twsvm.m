%------------主函数----------------
%twsvm可以基于二特征实现多分类问题
%线性可分，线性不可分要加核函数
clear all
close all
clc
tic;
m = 2;%类别
n = 50;%每类的样本数
S = n*m;%样本总数
C1 = 10;  %成本约束参数
C2 = 8;



% Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出

%data = xlsread('iris.xlsx');
load alldatapot2_lable.mat
data = alldatapot2_lable;
X = data(:,16:18);%截取数据集
x1 = X(1:50,[1,2]);%第一类
y1 = ones(1,50);
x2 = X(201:250,[1,2]);%第二类
y2 = -ones(1,50);
e1 = ones(50,1);%全一向量
e2 = ones(50,1);


H = [x1,e1];
G = [x2,e2];
P = [x1,e1];
Q = [x2,e2];

%②-------------训练样本
X = [x1',x2'];       
Y = [y1,y2];        

%% 二次规划来求解问题，可输入命令help quadprog查看详情

H1 = G*inv(H'*H)*G';
H2 = P*inv(Q'*Q)*P';
f1 = -e2; %f为1*n个-1,f相当于Quadprog函数中的c
f2 = -e1;
A1 = [];
b1 = [];
A2 = [];
b2 = [];
Aeq1 = []; 
beq1 = [];
Aeq2 = []; 
beq2 = [];
lb = zeros(m*n,1); %相当于Quadprog函数中的LB，UB
ub1 = C1*ones(m*n,1);
ub2 = C2*ones(m*n,1);
a0 = zeros(50,1);  % a0是解的初始近似值
[a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb,ub1,a0,options);
[a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb,ub2,a0,options);
%a是输出变量，问题的解
%fval是目标函数在解a处的值
%eXitflag>0,则程序收敛于解x；=0则函数的计算达到了最大次数；<0则问题无可行解，或程序运行失败
%output输出程序运行的某些信息
%lambda为在解a处的值Lagrange乘子

u  = -inv(H'*H)*G'*a1;
v  = -inv(Q'*Q)*P'*a2;
w1 = u(1:end-1);
b1 = u(end);
w2 = v(1:end-1);
b2 = v(end);


%{
epsilon = 1e-8;  
 %0<a<a(max)则认为x为支持向量,find返回一个包含数组X中每个非零元素的线性索引的向量。 
sv_label = find(abs(a)>epsilon); %找到支持向量对应a的下标，同时对应的也是测试点X的下标    
svm.a = a(sv_label);%支持向量对应的w的值
svm.Xsv = X(:,sv_label);%支持向量点          第一行为X的的横坐标，第二行为X的纵坐标
svm.Ysv = Y(sv_label);%支持向量的类别
svm.svnum = length(sv_label);%支持向量的个数
%}



%% 作图
%{
figure;  %创建一个用来显示图形输出的一个窗口对象
%    横坐标   纵坐标
plot(x1(:,1)',x1(:,2)','bs',x2(:,1)',x2(:,2)','k+');  %画图，两堆点
axis([min(X(1,:)) max(X(1,:)) min(X(2,:)) max(X(2,:))]);  %设置坐标轴范围
hold on;    %在同一个figure中画几幅图时，用此句
 

%plot(svm.Xsv(1,:),svm.Xsv(2,:),'ro');   %把支持向量标出来
 
%③-------------测试
jg1 = (max(X(1,:))-min(X(1,:)))/180;
jg2 = (max(X(2,:))-min(X(2,:)))/180; 
[x1,x2] = meshgrid(min(X(1,:)):jg1:max(X(1,:)),min(X(2,:)):jg2:max(X(2,:)));  %x1和x2都是181*181的矩阵
[rows,cols] = size(x1);  
nt = rows*cols;                  
Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];
%前半句reshape(x1,1,nt)是将x1转成1*（181*181）的矩阵，所以xt是2*（181*181）的矩阵
%reshape函数重新调整矩阵的行、列、维数
result1 = w1'*Xt+b1;
result2 = w2'*Xt+b2;

%Yt = ones(1,nt);
%对所有点进行测试
%result = svmTest(svm, Xt, Yt, kertype);
%④--------------画曲线的等高线图
Yd1 = reshape(result1,rows,cols);
Yd2 = reshape(result2,rows,cols);
contour(x1,x2,Yd1,[0,0],'ShowText','off','linecolor','g');%画等高线
contour(x1,x2,Yd2,[0,0],'ShowText','off','linecolor','r');
title('twsvm分类结果图');
x1=xlabel('X轴'); 
x2=ylabel('Y轴'); 
%}
%% 分类正确率
%这里用来测试。。
result1_ = w1'*X+b1;
result2_ = w2'*X+b2;
t1 = 0;
t2 = 0;
for i = 1:n
    if(abs(result1_(i)) > abs(result1_(i+n)))
        t1 = t1+1;
    end
    if(abs(result2_(i)) < abs(result2_(i+n)))
        t2 = t2+1;
    end
end
t = t1+t2;
Accuracy = 1-t/(n*m);
    
toc;

