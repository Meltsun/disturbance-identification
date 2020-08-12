function [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,C1,C2,n1,n2,k,T)
%�������
%Options�����������㷨��ѡ�������������optimset�޲�ʱ������һ��ѡ��ṹ�����ֶ�ΪĬ��ֵ��ѡ��
options = optimset;    
options.LargeScale = 'off';%LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
options.Display = 'off';    %��ʾ�����

I1 = ones(size(S'*S,1),size(S'*S,1));
I2 = ones(size(N'*N,1),size(N'*N,1));
H1 = R*inv(S'*S+k*I1)*R';
H2 = L*inv(N'*N+k*I2)*L';
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
lb1 = zeros(n2*T/10,1); %�൱��Quadprog�����е�LB��UB
lb2 = zeros(n1*T/10,1);
ub1 = C1*ones(n2*T/10,1);
ub2 = C2*ones(n1*T/10,1);
a01 = zeros(n2*T/10,1);  % a0�ǽ�ĳ�ʼ����ֵ
a02 = zeros(n1*T/10,1);  
[a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb1,ub1,a01,options);
[a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb2,ub2,a02,options);
%a���������������Ľ�
%fval��Ŀ�꺯���ڽ�a����ֵ
%eXitflag>0,����������ڽ�x��=0�����ļ���ﵽ����������<0�������޿��н⣬���������ʧ��
%output����������е�ĳЩ��Ϣ
%lambdaΪ�ڽ�a����ֵLagrange����

u  = -inv(S'*S+k*I1)*R'*a1;
v  =  inv(N'*N+k*I2)*L'*a2;
w1 = u(1:end-1);
b1 = u(end);
w2 = v(1:end-1);
b2 = v(end);
end
