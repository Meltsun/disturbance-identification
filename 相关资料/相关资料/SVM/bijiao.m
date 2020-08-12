clear all
close all
clc
T = 6;
n = 50;
load alldatapot2_lable.mat
Data = alldatapot2_lable;
% Data1 = feature_end(Data);
% Data = [Data(:,1:44),Data1,Data(:,end)];
% %data = Z_mean(Data);%归一化处理
% qian = Data(:,end);
% data1 = mapminmax(Data(:,1:44),0,1);
% data = [data1,Data1,qian];
% data = mapminmax(Data(:,1:44));
% data = [data,Data1,Data(:,end)];
% tz = (1:44); 
% %tz = randi(44,1,2);
data = Data;
tz = [3,7];
teDA = data(1:50,tz);
teDB = data(51:100,tz);
teDC = data(101:150,tz);
teDD = data(151:200,tz);
teDE = data(201:250,tz);
% data2 = yasuo(A,B,C,D,E,10);
% A = data2(1:40,:);
% B = data2(41:80,:);
% C = data2(81:120,:);
% D = data2(121:160,:);
% E = data2(161:200,:);
% [trDA,teDA] = F_CV(A,T);
% [trDB,teDB] = F_CV(B,T);
% [trDC,teDC] = F_CV(C,T);
% [trDD,teDD] = F_CV(D,T);
% [trDE,teDE] = F_CV(E,T);


figure
%plot(A(:,1),A(:,2),'k*',B(:,1),B(:,2),'mp',D(:,1),D(:,2),'r.');
% plot(trDA(:,1),trDA(:,2),'k*',trDB(:,1),trDB(:,2),'mp',trDD(:,1),trDD(:,2),'r.');
 plot(teDA(:,1),teDA(:,2),'k*',teDB(:,1),teDB(:,2),'mp',teDC(:,1),teDC(:,2),'gd',teDD(:,1),teDD(:,2),'r.',teDE(:,1),teDE(:,2),'bs');
