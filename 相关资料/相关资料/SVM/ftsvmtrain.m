function [w1,b1,w2,b2] = ftsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T,SA,SB,options)
%�������
%kΪ���С����ʵ����Ϊ�˱�������ֵ����������޴��������Ϊ0
I1 = ones(size(H'*H,1),size(H'*H,1));
I2 = ones(size(Q'*Q,1),size(Q'*Q,1));
H1 = G*inv(H'*H+k*I1)*G';
H2 = P*inv(Q'*Q+k*I2)*P';
f1 = -e2; %fΪ1*n��-1,f�൱��Quadprog�����е�c
f2 = -e1;
A1 = [];
b1 = [];
A2 = [];
b2 = [];
Aeq1 = []; 
beq1 = [];
Aeq2 = []; 
beq2 = [];
lb1 = zeros(n2,1); %�൱��Quadprog�����е�LB��UB
lb2 = zeros(n1,1);
ub1 = c1*ones(n2,1);%SB;
ub2 = c2*ones(n1,1);%SA;
a01 = zeros(n2,1);  % a0�ǽ�ĳ�ʼ����ֵ
a02 = zeros(n1,1);
[a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb1,ub1,a01,options);
[a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb2,ub2,a02,options);
%a���������������Ľ�
%fval��Ŀ�꺯���ڽ�a����ֵ
%eXitflag>0,����������ڽ�x��=0�����ļ���ﵽ����������<0�������޿��н⣬���������ʧ��
%output����������е�ĳЩ��Ϣ
%lambdaΪ�ڽ�a����ֵLagrange����

u  = -inv(H'*H+k*I1)*G'*a1;
v  = -inv(Q'*Q+k*I2)*P'*a2;
w1 = u(1:end-1);
b1 = u(end);
w2 = v(1:end-1);
b2 = v(end);
end
