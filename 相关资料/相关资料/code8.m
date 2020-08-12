clc
close all
clear all
tic;
m = 5;
n = 3;
pdatanum = 33;
dalt = 33;
M = pdatanum;	%  ÿ�����ݸ���

pot1 = 36;      %  �Ŷ���1 900m
pot2 = 260;     %  �Ŷ���2 6500m
pot3 = 860;     %  �Ŷ���3 21500m

load datajiao.mat
load dataqiao.mat
load datapa.mat
load dataya.mat
load datano.mat

ordatajiao	=   -datajiao;
ordatapa	=   -datapa;
ordataqiao	=   -dataqiao;
ordataya	=   -dataya;
ordatano	=   -datano;
%��һ��
% ordatajiao	=	mapnormal(ordatajiao);
% ordatapa	=	mapnormal(ordatapa);
% ordataqiao	=	mapnormal(ordataqiao);
% ordataya	=	mapnormal(ordataya);
% ordatano	=	mapnormal(ordatano);

%  �����Ŷ����ݸ���
[jiaonum, ~]	= size(ordatajiao);   
[panum, ~]      = size(ordatapa);
[qiaonum, ~]	= size(ordataqiao);
[yanum, ~]      = size(ordataya);
[nonum, ~]      = size(ordatano);
% ��С��
datanum     = min([jiaonum panum qiaonum yanum nonum]);

%% �Ŷ��ź���ȡ
datajiao1       =	ordatajiao	(1:datanum, pot1 );%��һ���Ŷ������ڵ�ʱ���У�һ�У�
datapa1         =	ordatapa	(1:datanum, pot1 );%��һ���Ŷ������ڵ�ʱ���У�һ�У�
dataqiao1       =	ordataqiao	(1:datanum, pot1 );%��һ���Ŷ������ڵ�ʱ���У�һ�У�
dataya1         =	ordataya	(1:datanum, pot1 );%��һ���Ŷ������ڵ�ʱ���У�һ�У�
datano1         =	ordatano	(1:datanum, pot1 );%��һ���Ŷ������ڵ�ʱ���У�һ�У�

datajiao2       =	ordatajiao	(1:datanum, pot2 );%�ڶ����Ŷ������ڵ�ʱ���У�һ�У�
datapa2         =	ordatapa	(1:datanum, pot2 );%�ڶ����Ŷ������ڵ�ʱ���У�һ�У�
dataqiao2       =	ordataqiao	(1:datanum, pot2 );%�ڶ����Ŷ������ڵ�ʱ���У�һ�У�
dataya2         =	ordataya	(1:datanum, pot2 );%�ڶ����Ŷ������ڵ�ʱ���У�һ�У�
datano2         =	ordatano	(1:datanum, pot2 );%�ڶ����Ŷ������ڵ�ʱ���У�һ�У�

datajiao3       =	ordatajiao	(1:datanum, pot3 );%�������Ŷ������ڵ�ʱ���У�һ�У�
datapa3         =	ordatapa	(1:datanum, pot3 );%�������Ŷ������ڵ�ʱ���У�һ�У�
dataqiao3       =	ordataqiao	(1:datanum, pot3 );%�������Ŷ������ڵ�ʱ���У�һ�У�
dataya3         =	ordataya	(1:datanum, pot3 );%�������Ŷ������ڵ�ʱ���У�һ�У�
datano3         =	ordatano	(1:datanum, pot3 );%�������Ŷ������ڵ�ʱ���У�һ�У�

%% ����
datajiao_pot    =   [datajiao1';datajiao2';datajiao3'];
datapa_pot      =   [datapa1';datapa2';datapa3'];
dataqiao_pot    =   [dataqiao1';dataqiao2';dataqiao3'];
dataya_pot      =   [dataya1';dataya2';dataya3'];
datano_pot      =   [datano1';datano2';datano3'];


Data = {datajiao_pot,datapa_pot,dataqiao_pot,dataya_pot,datano_pot};
%% ����--������ȡ�����������ص�
%�����ص���a������һ��1-33���ڶ���a+1 - a+33��.... ,��t��(t-1)*a+1 - (t-1)*a+33
%������(t-1)*a+33 <= 1127   t >0,0<a<=33������
% a = 33  t = 34  �����ص�

fenzu= [];
for i = 1:dalt    %dalt = 33
    fenzu(i) = floor((datanum - dalt)/i+1);  
end
% fenzu(22) = 50

a = 22;   %����a��ֵ�ı�������

groupnum = fenzu(a);
Odata = {};
for i = 1:m
    for j = 1:n
        data = Data{1,i}(j,:);
        P = [];
        for k = 1:groupnum
            P(k,:) = data(1,(k-1)*a+1:(k-1)*a+33); 
        end
        Odata{i,j} = P;
    end
end
%% ���ﱾ������������ֿ��ģ�����һ��Ҳû�н��и���Ĵ���
%�޲��
Data = Odata;
tdatajiao1 = Data{1,1};
tdatajiao2 = Data{1,2};
tdatajiao3 = Data{1,3};


tdatapa1 = Data{2,1};
tdatapa2 = Data{2,2};
tdatapa3 = Data{2,3};


tdataqiao1 = Data{3,1};
tdataqiao2 = Data{3,2};
tdataqiao3 = Data{3,3};


tdataya1 = Data{4,1};
tdataya2 = Data{4,2};
tdataya3 = Data{4,3};


tdatano1 = Data{5,1};
tdatano2 = Data{5,2};
tdatano3 = Data{5,3};

oalldatapot1 = [tdatajiao1; tdataqiao1; tdatapa1; tdataya1; tdatano1]; 
oalldatapot2 = [tdatajiao2; tdataqiao2; tdatapa2; tdataya2; tdatano2]; 
oalldatapot3 = [tdatajiao3; tdataqiao3; tdatapa3; tdataya3; tdatano3]; 
tic;
alldatapot1 = datafeature(oalldatapot1);
alldatapot2 = datafeature(oalldatapot2);
alldatapot3 = datafeature(oalldatapot3);

toc;
