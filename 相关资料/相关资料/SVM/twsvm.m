%------------������----------------
%twsvm���Ի��ڶ�����ʵ�ֶ��������
%���Կɷ֣����Բ��ɷ�Ҫ�Ӻ˺���
clear all
close all
clc
tic;
m = 2;%���
n = 50;%ÿ���������
S = n*m;%��������
C1 = 10;  %�ɱ�Լ������
C2 = 8;



% Options�����������㷨��ѡ�������������optimset�޲�ʱ������һ��ѡ��ṹ�����ֶ�ΪĬ��ֵ��ѡ��
options = optimset;    
options.LargeScale = 'off';%LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
options.Display = 'off';    %��ʾ�����

%data = xlsread('iris.xlsx');
load alldatapot2_lable.mat
data = alldatapot2_lable;
X = data(:,16:18);%��ȡ���ݼ�
x1 = X(1:50,[1,2]);%��һ��
y1 = ones(1,50);
x2 = X(201:250,[1,2]);%�ڶ���
y2 = -ones(1,50);
e1 = ones(50,1);%ȫһ����
e2 = ones(50,1);


H = [x1,e1];
G = [x2,e2];
P = [x1,e1];
Q = [x2,e2];

%��-------------ѵ������
X = [x1',x2'];       
Y = [y1,y2];        

%% ���ι滮��������⣬����������help quadprog�鿴����

H1 = G*inv(H'*H)*G';
H2 = P*inv(Q'*Q)*P';
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
lb = zeros(m*n,1); %�൱��Quadprog�����е�LB��UB
ub1 = C1*ones(m*n,1);
ub2 = C2*ones(m*n,1);
a0 = zeros(50,1);  % a0�ǽ�ĳ�ʼ����ֵ
[a1,fval1,eXitflag1,output1,lambda1]  = quadprog(H1,f1,A1,b1,Aeq1,beq1,lb,ub1,a0,options);
[a2,fval2,eXitflag2,output2,lambda2]  = quadprog(H2,f2,A2,b2,Aeq2,beq2,lb,ub2,a0,options);
%a���������������Ľ�
%fval��Ŀ�꺯���ڽ�a����ֵ
%eXitflag>0,����������ڽ�x��=0�����ļ���ﵽ����������<0�������޿��н⣬���������ʧ��
%output����������е�ĳЩ��Ϣ
%lambdaΪ�ڽ�a����ֵLagrange����

u  = -inv(H'*H)*G'*a1;
v  = -inv(Q'*Q)*P'*a2;
w1 = u(1:end-1);
b1 = u(end);
w2 = v(1:end-1);
b2 = v(end);


%{
epsilon = 1e-8;  
 %0<a<a(max)����ΪxΪ֧������,find����һ����������X��ÿ������Ԫ�ص����������������� 
sv_label = find(abs(a)>epsilon); %�ҵ�֧��������Ӧa���±꣬ͬʱ��Ӧ��Ҳ�ǲ��Ե�X���±�    
svm.a = a(sv_label);%֧��������Ӧ��w��ֵ
svm.Xsv = X(:,sv_label);%֧��������          ��һ��ΪX�ĵĺ����꣬�ڶ���ΪX��������
svm.Ysv = Y(sv_label);%֧�����������
svm.svnum = length(sv_label);%֧�������ĸ���
%}



%% ��ͼ
%{
figure;  %����һ��������ʾͼ�������һ�����ڶ���
%    ������   ������
plot(x1(:,1)',x1(:,2)','bs',x2(:,1)',x2(:,2)','k+');  %��ͼ�����ѵ�
axis([min(X(1,:)) max(X(1,:)) min(X(2,:)) max(X(2,:))]);  %���������᷶Χ
hold on;    %��ͬһ��figure�л�����ͼʱ���ô˾�
 

%plot(svm.Xsv(1,:),svm.Xsv(2,:),'ro');   %��֧�����������
 
%��-------------����
jg1 = (max(X(1,:))-min(X(1,:)))/180;
jg2 = (max(X(2,:))-min(X(2,:)))/180; 
[x1,x2] = meshgrid(min(X(1,:)):jg1:max(X(1,:)),min(X(2,:)):jg2:max(X(2,:)));  %x1��x2����181*181�ľ���
[rows,cols] = size(x1);  
nt = rows*cols;                  
Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];
%ǰ���reshape(x1,1,nt)�ǽ�x1ת��1*��181*181���ľ�������xt��2*��181*181���ľ���
%reshape�������µ���������С��С�ά��
result1 = w1'*Xt+b1;
result2 = w2'*Xt+b2;

%Yt = ones(1,nt);
%�����е���в���
%result = svmTest(svm, Xt, Yt, kertype);
%��--------------�����ߵĵȸ���ͼ
Yd1 = reshape(result1,rows,cols);
Yd2 = reshape(result2,rows,cols);
contour(x1,x2,Yd1,[0,0],'ShowText','off','linecolor','g');%���ȸ���
contour(x1,x2,Yd2,[0,0],'ShowText','off','linecolor','r');
title('twsvm������ͼ');
x1=xlabel('X��'); 
x2=ylabel('Y��'); 
%}
%% ������ȷ��
%�����������ԡ���
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

