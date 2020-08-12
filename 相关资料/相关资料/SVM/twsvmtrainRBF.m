function [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,C1,C2,n1,n2,k,T)
%二分类仅
%Options是用来控制算法的选项参数的向量，optimset无参时，创建一个选项结构所有字段为默认值的选项
options = optimset;    
options.LargeScale = 'off';%LargeScale指大规模搜索，off表示在规模搜索模式关闭
options.Display = 'off';    %表示无输出

I1 = ones(size(S'*S,1),size(S'*S,1));
I2 = ones(size(N'*N,1),size(N'*N,1));
H1 = R*inv(S'*S+k*I1)*R';
H2 = L*inv(N'*N+k*I2)*L';
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
lb1 = zeros(n2*T/10,1); %相当于Quadprog函数中的LB，UB
lb2 = zeros(n1*T/10,1);
ub1 = C1*ones(n2*T/10,1);
ub2 = C2*ones(n1*T/10,1);
a01 = zeros(n2*T/10,1);  % a0是解的初始近似值
a02 = zeros(n1*T/10,1);  
[a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb1,ub1,a01,options);
[a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb2,ub2,a02,options);
%a是输出变量，问题的解
%fval是目标函数在解a处的值
%eXitflag>0,则程序收敛于解x；=0则函数的计算达到了最大次数；<0则问题无可行解，或程序运行失败
%output输出程序运行的某些信息
%lambda为在解a处的值Lagrange乘子

u  = -inv(S'*S+k*I1)*R'*a1;
v  =  inv(N'*N+k*I2)*L'*a2;
w1 = u(1:end-1);
b1 = u(end);
w2 = v(1:end-1);
b2 = v(end);
end
