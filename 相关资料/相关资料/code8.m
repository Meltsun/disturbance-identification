clc
close all
clear all
tic;
m = 5;
n = 3;
pdatanum = 33;
dalt = 33;
M = pdatanum;	%  每组数据个数

pot1 = 36;      %  扰动点1 900m
pot2 = 260;     %  扰动点2 6500m
pot3 = 860;     %  扰动点3 21500m

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
%归一化
% ordatajiao	=	mapnormal(ordatajiao);
% ordatapa	=	mapnormal(ordatapa);
% ordataqiao	=	mapnormal(ordataqiao);
% ordataya	=	mapnormal(ordataya);
% ordatano	=	mapnormal(ordatano);

%  各种扰动数据个数
[jiaonum, ~]	= size(ordatajiao);   
[panum, ~]      = size(ordatapa);
[qiaonum, ~]	= size(ordataqiao);
[yanum, ~]      = size(ordataya);
[nonum, ~]      = size(ordatano);
% 最小量
datanum     = min([jiaonum panum qiaonum yanum nonum]);

%% 扰动信号提取
datajiao1       =	ordatajiao	(1:datanum, pot1 );%第一个扰动点所在的时间列（一列）
datapa1         =	ordatapa	(1:datanum, pot1 );%第一个扰动点所在的时间列（一列）
dataqiao1       =	ordataqiao	(1:datanum, pot1 );%第一个扰动点所在的时间列（一列）
dataya1         =	ordataya	(1:datanum, pot1 );%第一个扰动点所在的时间列（一列）
datano1         =	ordatano	(1:datanum, pot1 );%第一个扰动点所在的时间列（一列）

datajiao2       =	ordatajiao	(1:datanum, pot2 );%第二个扰动点所在的时间列（一列）
datapa2         =	ordatapa	(1:datanum, pot2 );%第二个扰动点所在的时间列（一列）
dataqiao2       =	ordataqiao	(1:datanum, pot2 );%第二个扰动点所在的时间列（一列）
dataya2         =	ordataya	(1:datanum, pot2 );%第二个扰动点所在的时间列（一列）
datano2         =	ordatano	(1:datanum, pot2 );%第二个扰动点所在的时间列（一列）

datajiao3       =	ordatajiao	(1:datanum, pot3 );%第三个扰动点所在的时间列（一列）
datapa3         =	ordatapa	(1:datanum, pot3 );%第三个扰动点所在的时间列（一列）
dataqiao3       =	ordataqiao	(1:datanum, pot3 );%第三个扰动点所在的时间列（一列）
dataya3         =	ordataya	(1:datanum, pot3 );%第三个扰动点所在的时间列（一列）
datano3         =	ordatano	(1:datanum, pot3 );%第三个扰动点所在的时间列（一列）

%% 整合
datajiao_pot    =   [datajiao1';datajiao2';datajiao3'];
datapa_pot      =   [datapa1';datapa2';datapa3'];
dataqiao_pot    =   [dataqiao1';dataqiao2';dataqiao3'];
dataya_pot      =   [dataya1';dataya2';dataya3'];
datano_pot      =   [datano1';datano2';datano3'];


Data = {datajiao_pot,datapa_pot,dataqiao_pot,dataya_pot,datano_pot};
%% 分组--特征提取————有重叠
%若不重叠有a个，第一组1-33，第二组a+1 - a+33，.... ,第t组(t-1)*a+1 - (t-1)*a+33
%所以有(t-1)*a+33 <= 1127   t >0,0<a<=33的整数
% a = 33  t = 34  即不重叠

fenzu= [];
for i = 1:dalt    %dalt = 33
    fenzu(i) = floor((datanum - dalt)/i+1);  
end
% fenzu(22) = 50

a = 22;   %调节a的值改变分组个数

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
%% 这里本来是两个程序分开的，合在一起也没有进行更多的处理
%无差分
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
