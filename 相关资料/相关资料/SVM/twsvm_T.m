clear all
close all
clc
tic;
%% ��ʼ������
n1 = 50;               %���
n2 = 50;              %ÿ���������
S = n1+n2;             %��������
T = 6;               %ѵ���������������� = A��10-A,�ɵ�
M = 100;             %��������
W = (1-T/10)*M*100;%m*n;  %��������
k = 0;               %������������ֵ��һ��Ϊ0
%��ͨ����Ӭ�㷨�õ�C1��C2
C1 = 10;             %�ɱ�Լ������
C2 = 8;
% Options�����������㷨��ѡ�������������optimset�޲�ʱ������һ��ѡ��ṹ�����ֶ�ΪĬ��ֵ��ѡ��
options = optimset;    
options.LargeScale = 'off';%LargeScaleָ���ģ������off��ʾ�ڹ�ģ����ģʽ�ر�
options.Display = 'off';    %��ʾ�����
%% ��������
load alldatapot2_lable.mat
data = alldatapot2_lable;
%data = xlsread('iris.xlsx');

x1 = data(51:100,[3,4,5,15,24,26]);   %��һ��
%y1 = ones(1,50);      %��ӱ�ǩ+1
x2 = data(101:150,[3,4,5,15,24,26]);%�ڶ���
%y2 = -ones(1,50);     %��ӱ�ǩ-1


t1 = 0;
t2 = 0;

e1 = ones(50*T/10,1);
e2 = ones(50*T/10,1);


%trDΪѵ������teDΪ���Լ�
%��ʮ����˵ÿ����45��ѵ�����ݣ�5����������
for i = 1:M    
    
    [trD1,teD1] = F_CV(x1,T);
    [trD2,teD2] = F_CV(x2,T);
    %ȫһ����
  
     
    
    H = [trD1,e1];
    G = [trD2,e2];
    P = [trD1,e1];
    Q = [trD2,e2];
 

    %ѵ������
    Xtrain = [trD1',trD2']; %��ά������ͼ
    %��������
    Xtest  = [teD1',teD2'];
    %% ���ι滮���������
    %�õ�w��b
    [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,C1,C2,n1,n2,k,T,options);%n = 2,m = 45
    %% ��ͼ
   %{
    %������������ʱû�а취��ͼ
    figure
    %ѵ����
    plot(trD1(:,1)',trD1(:,2)','b*',trD2(:,1)',trD2(:,2)','k+');  %��ͼ�����ѵ�
    axis([min(Xtrain(1,:)) max(Xtrain(1,:)) min(Xtrain(2,:)) max(Xtrain(2,:))]);  %���������᷶Χ
    hold on
    plot(teD1(:,1)',teD1(:,2)','*c',teD2(:,1)',teD2(:,2)','+m');  %��ͼ�����ѵ�
    axis([min(Xtrain(1,:)) max(Xtrain(1,:)) min(Xtrain(2,:)) max(Xtrain(2,:))]);  %���������᷶Χ    
    hold on;
    %���Ե�
    
    jg1 = (max(Xtrain(1,:))-min(Xtrain(1,:)))/180;
    jg2 = (max(Xtrain(2,:))-min(Xtrain(2,:)))/180; 
    [x1_,x2_] = meshgrid(min(Xtrain(1,:)):jg1:max(Xtrain(1,:)),min(Xtrain(2,:)):jg2:max(Xtrain(2,:)));  %x1��x2����181*181�ľ���
    [rows,cols] = size(x1_);  
    nt = rows*cols;                  
    Xt = [reshape(x1_,1,nt);reshape(x2_,1,nt)];
    %ǰ���reshape(x1_,1,nt)�ǽ�x1_ת��1*��181*181���ľ�������xt��2*��181*181���ľ���
    %reshape�������µ���������С��С�ά��

    result1 = w1'*Xt+b1;
    result2 = w2'*Xt+b2;
  
    %�����ߵĵȸ���ͼ
    Yd1 = reshape(result1,rows,cols);
    Yd2 = reshape(result2,rows,cols);
    contour(x1_,x2_,Yd1,[0,0],'ShowText','off','linecolor','r');%���ȸ���
    contour(x1_,x2_,Yd2,[0,0],'ShowText','off','linecolor','g');
    title('twsvm������ͼ');
    x1_=xlabel('X��'); 
    x2_=ylabel('Y��'); 
     %}
    %% ����
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
Accuracy = 1-t/W;%�����������Ҫ��
toc;