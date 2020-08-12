%------------������----------------
%�����svm����
clear all;
close all;
clc
C = 10;  %�ɱ�Լ������
kertype = 'linear';  %���Ժ�
%{ 
%��------����׼��
n = 30;
%randn('state',6);   %ָ��״̬��һ����Բ���
x1 = randn(2,n);    %2��N�о���Ԫ�ط�����̬�ֲ�
y1 = ones(1,n);       %1*N��1
x2 = 4+randn(2,n);   %2*N����Ԫ�ط�����̬�ֲ��Ҿ�ֵΪ5�����Ը�˹�˿�x2 = 3+randn(2,n); 
y2 = -ones(1,n);      %1*N��-1
%}
%��-------����׼�� 
load('D:\���ִ���\ģʽʶ��\����������ȡ\Data\50�����ݵõ�������ֵ\����ģʽʶ��\����44������\alldatapot2_lable.mat');
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
        figure(t);  %����һ��������ʾͼ�������һ�����ڶ���
        
        %    ������   ������
        plot(x1(1,:),x1(2,:),'bs',x2(1,:),x2(2,:),'k+');  %��ͼ�����ѵ�

        min1 = min([data(151:200,f(i))',data(201:250,f(i))']);
        max1 = max([data(151:200,f(i))',data(201:250,f(i))']);
        min2 = min([data(151:200,f(j))',data(201:250,f(j))']);
        max2 = max([data(151:200,f(j))',data(201:250,f(j))']);
        axis([min1,max1,min2,max2]);  %���������᷶Χ
        hold on;    %��ͬһ��figure�л�����ͼʱ���ô˾�

        %��-------------ѵ������
        X = [x1,x2];        %ѵ������2*n����nΪ����������dΪ������������
        Y = [y1,y2];        %ѵ��Ŀ��1*n����nΪ����������ֵΪ+1��-1



        svm = svmTrain(X,Y,kertype,C);  %ѵ������
        plot(svm.Xsv(1,:),svm.Xsv(2,:),'ro');   %��֧�����������




        %��-------------����
        [x1,x2] = meshgrid(min1:(max1-min1)/180:max1,min2:(max2-min2)/180:max2);  %x1��x2����181*181�ľ���
        [rows,cols] = size(x1);  
        nt = rows*cols;                  
        Xt = [reshape(x1,1,nt);reshape(x2,1,nt)];
        %ǰ���reshape(x1,1,nt)�ǽ�x1ת��1*��181*181���ľ�������xt��2*��181*181���ľ���
        %reshape�������µ���������С��С�ά��
        Yt = ones(1,nt);
        %�����е���в���
        result = svmTest(svm, Xt, Yt, kertype);

        %��--------------�����ߵĵȸ���ͼ
        Yd = reshape(result.Y,rows,cols);
        contour(x1,x2,Yd,[0,0],'ShowText','off');%���ȸ���
        title('svm������ͼ');
        x1=xlabel('X��'); 
        x2=ylabel('Y��');
        t = t+1;
    end
end

