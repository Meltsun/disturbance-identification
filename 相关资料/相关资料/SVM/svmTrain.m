%-----------训练样本的函数---------
function svm = svmTrain(X,Y,kertype,C)
 
% Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出
 n=50;
%二次规划来求解问题，可输入命令help quadprog查看详情
n = length(Y);  %返回Y最长维数
H = (Y'*Y).*kernel(X,X,kertype);    
f = -ones(n,1); %f为1*n个-1,f相当于Quadprog函数中的c
A = [];
b = [];
Aeq = Y; %相当于Quadprog函数中的A1,b1
beq = 0;
lb = zeros(n,1); %相当于Quadprog函数中的LB，UB
ub = C*ones(n,1);
a0 = zeros(n,1);  % a0是解的初始近似值
[a,fval,eXitflag,output,lambda]  = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
%a是输出变量，问题的解
%fval是目标函数在解a处的值
%eXitflag>0,则程序收敛于解x；=0则函数的计算达到了最大次数；<0则问题无可行解，或程序运行失败
%output输出程序运行的某些信息
%lambda为在解a处的值Lagrange乘子
 
epsilon = 1e-8;  
 %0<a<a(max)则认为x为支持向量,find返回一个包含数组X中每个非零元素的线性索引的向量。 
sv_label = find(abs(a)>epsilon); %找到支持向量对应a的下标，同时对应的也是测试点X的下标    
svm.a = a(sv_label);%支持向量对应的w的值
svm.Xsv = X(:,sv_label);%支持向量点          第一行为X的的横坐标，第二行为X的纵坐标
svm.Ysv = Y(sv_label);%支持向量的类别
svm.svnum = length(sv_label);%支持向量的个数
%svm.label = sv_label;
end

