function [w1,b1,w2,b2] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T,SA,SB,options)
%二分类仅
%k为充分小的正实数是为了避免奇异值的情况，若无此情况可设为0
I1 = ones(size(H'*H,1),size(H'*H,1));
I2 = ones(size(Q'*Q,1),size(Q'*Q,1));
H1 = G*inv(H'*H+k*I1)*G';
H2 = P*inv(Q'*Q+k*I2)*P';
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
lb1 = zeros(n2,1); %相当于Quadprog函数中的LB，UB
lb2 = zeros(n1,1);
ub1 = c1*ones(n2,1);%SB;
ub2 = c2*ones(n1,1);%SA;
a01 = zeros(n2,1);  % a0是解的初始近似值
a02 = zeros(n1,1);
[a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb1,ub1,a01,options);
[a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb2,ub2,a02,options);
%a是输出变量，问题的解
%fval是目标函数在解a处的值
%eXitflag>0,则程序收敛于解x；=0则函数的计算达到了最大次数；<0则问题无可行解，或程序运行失败
%output输出程序运行的某些信息
%lambda为在解a处的值Lagrange乘子

u  = -inv(H'*H+k*I1)*G'*a1;
v  = -inv(Q'*Q+k*I2)*P'*a2;
w1 = u(1:end-1);
b1 = u(end);
w2 = v(1:end-1);
b2 = v(end);
end
