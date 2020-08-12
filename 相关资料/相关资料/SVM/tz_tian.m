clc
clear all
close all
load Over_lap.mat
t = 1;
data1 = [Odata{1,1};Odata{3,1};Odata{2,1};Odata{4,1};Odata{5,1}];
data2 = [Odata{1,2};Odata{3,2};Odata{2,2};Odata{4,2};Odata{5,2}];
data3 = [Odata{1,3};Odata{3,3};Odata{2,3};Odata{4,3};Odata{5,3}];

% data = [Odata{1,t};Odata{2,t};Odata{3,t};Odata{4,t};Odata{5,t}];

% data1 = mapnormal(data1);
% data2 = mapnormal(data2);
% data3 = mapnormal(data3);

% alldatapot1 = datafeaturenew(data1);%,datafeature(oalldatapot1diff)];
% alldatapot2 = datafeaturenew(data2);%,datafeature(oalldatapot2diff)];
% alldatapot3 = datafeaturenew(data3);%,datafeature(oalldatapot3diff)];
% for i = 1:32 
%     alldata1(:,i) = (mapnormal(alldatapot1(:,i)'))';
%     alldata2(:,i) = (mapnormal(alldatapot2(:,i)'))';
%     alldata3(:,i) = (mapnormal(alldatapot3(:,i)'))';
% end

% groupnum = 50;
% %  标签
% alllable = zeros(groupnum*5, 1);
% alllable(groupnum*0 + 1 : groupnum*1, 1) = 1;   %  jiao
% alllable(groupnum*1 + 1 : groupnum*2, 1) = 2;   %  qiao
% alllable(groupnum*2 + 1 : groupnum*3, 1) = 3;   %  ya
% alllable(groupnum*3 + 1 : groupnum*4, 1) = 4;   %  pa
% alllable(groupnum*4 + 1 : groupnum*5, 1) = 5;   %  no
% 
% %% 组合
% alldatapot1_lable = [alldata1,alllable];
% alldatapot2_lable = [alldata2,alllable];
% alldatapot3_lable = [alldata3,alllable];
% toc;
% for i = 1:250
%     x(i,:) =mean(xcorr(dataf(i,:))); %dataf = mapnormal(dataT);
% end
%   t = [1:50];
% for i = 1:250
%     x(i,:) = abs(fft(data3(i,:)));
% end
% for i = 1:250
%     EP(i,:) = x(i,:).*x(i,:);
%     ep1(i,:) = EP(i,:)/sum( EP(i,:));
% end
% for i = 1:250
%     S = 0;
%     for j = 1:33
%        S =S - ep1(i,j)*log(ep1(i,j));
%     end
%     H1(i,1) = S;
% end

% xx = mean(dataf,2);
% yy = dataf-repmat(xx,1,33);
% for i = 1:250
%     %SNR(i,:) = 10*log(norm(xx(i,:)).^2/norm(yy(i,:)).^2);
%     SNR(i,:) = xx(i,:);
% end
% x1 = SNR(1:50,1);
% x2 = SNR(51:100,1);
% x3 = SNR(101:150,1);
% x4 = SNR(151:200,1);
% x5 = SNR(201:250,1);
% wpt  = wpdec(dataf,3,'db1');
% p = 3;
% for i = 1:2^p
%     X(1,i)= wpcoef(wpt,[3 i-1]);
% end
% figure
% plot(t,x1,'k^',t,x2,'m.',t,x3,'r*',t,x4,'bs',t,x5,'gd');
% % load alldatapot2_lable.mat
% % data = alldatapot2_lable;
% % data = data(:,1:44);
% % data1 = Main(data);

% m = 'db10';
% for j =1:250
%     Data = data(j,:);
%     q = 3;
%     wpt  = wpdec(Data,q,m);
%     H = [];
%     X    = {};
%     Y    = {};
%     E    = [];
%     p = 3;% 取第三层
%    % 得到第三层小波系数
%     for i = 1:2^p
%         X{1,i} = wpcoef(wpt,[p i-1]); 
%     end
%     %得到第三层各小波系数的重构信号
%     for i = 1:2^p
%         Y{1,i} = wprcoef(wpt,[p i-1]); 
%     end
%     %求能量
%     [~,k] = size(X);
%     for i = 1:k
%         E(1,i) = sum(Y{1,i}.^2); 
%     end
%      %能量归一化
%     P = 1/sum(E)*E;
%     [n,~] = size(P);
%     for i =1:n
%         H(1,i) = -P(i,1)*log10(P(i,1));
%     end
%     x1(j,:) = sum(H);
% end
% for j =1:250
%     Data = data(j,:);
%     Data = datacolumndiff(Data,1);
%     q = 3;
%     wpt  = wpdec(Data,q,m);
%     H = [];
%     X    = {};
%     Y    = {};
%     E    = [];
%     p = 3;% 取第三层
%    % 得到第三层小波系数
%     for i = 1:2^p
%         X{1,i} = wpcoef(wpt,[p i-1]); 
%     end
%     %得到第三层各小波系数的重构信号
%     for i = 1:2^p
%         Y{1,i} = wprcoef(wpt,[p i-1]); 
%     end
%     %求能量
%     [~,k] = size(X);
%     for i = 1:k
%         E(1,i) = sum(Y{1,i}.^2); 
%     end
%      %能量归一化
%     P = 1/sum(E)*E;
%     [n,~] = size(P);
%     for i =1:n
%         H(1,i) = -P(i,1)*log10(P(i,1));
%     end
%     x2(j,:) = sum(H);
% end
% for i = 1:250
%     Data = datacolumndiff(dataT(i,:),1);
%     x(i,1) = log(sum(Data.*Data));
% 
% end
% for i = 1:250
%     Data = dataT(i,:);
% %     Data = datacolumndiff(Data,1);
%     [S,F,T] = specgram(Data);
%     abs_S = abs(S);
%     %max_abs_S = max(max(abs_S));
%     abs_S = abs_S/sum(abs_S);
%     x(i,1) = -sum(abs_S.*log(abs_S));
%     %log10_abs_S = 20*log10(abs_S);
% end

% % for i= 1:250
% %     dataf = dataT(i,:);
% %     dataf = dataf-mean(dataf);
% %     x(i,1) = length(find(dataf>0))/33;
% %     datag = datacolumndiff(dataf,1);
% %     datag = datag-mean(datag);
% %     y2(i,1) = length(find(datag>0))/33;
% % end
% m = 3;
% for i= 1:250
%     dataf = data(i,:);
%     x1(i,1) = columnEsvd1(dataf,m);
%     Data = datacolumndiff(dataf,1);
%     x2(i,1) = columnEsvd1(Data,m);
% end

% for i= 1:250
%     dataf = dataT(i,:);
%     x(i,1) = columnEMDenergyH(dataf);
% end
% for i= 1:250
%     dataf = dataT(i,:);
%     x(i,1) = columnDWTenergyH(dataf);
% end
% for i= 1:250
%    
%     x(i,1) = columnzcr(dataf(i,:));
% end
% x = [x1,x2];
% for i = 1:250
% DEFSPL = 400;
%     bjp = [];
%     dalt = 33;%每组数据的个数
%     
%     date = data2(i,:);
%     imf=emd(date);
%     
%     [A,f,~]=hhspectrum(imf);
%     [E,~,Cenf]=toimage(A,f);
%     for k=1:DEFSPL
%         bjp(k)=sum(E(k,:))*1/dalt;
%     end
%     bjp = bjp(find(bjp~=0));
%     x2(i,1) = sum(-bjp.^2/sum(bjp.^2,2).*log10(bjp.^2/sum(bjp.^2,2)));
% end
% for i = 1:250
% DEFSPL = 400;
%     bjp = [];
%     dalt = 33;%每组数据的个数
%     
%     date = data3(i,:);
%     imf=emd(date);
%     
%     [A,f,~]=hhspectrum(imf);
%     [E,~,Cenf]=toimage(A,f);
%     for k=1:DEFSPL
%         bjp(k)=sum(E(k,:))*1/dalt;
%     end
%     bjp = bjp(find(bjp~=0));
%     x3(i,1) = sum(-bjp.^2/sum(bjp.^2,2).*log10(bjp.^2/sum(bjp.^2,2)));
% end
% for i = 1:250
% DEFSPL = 400;
%     bjp = [];
%     dalt = 33;%每组数据的个数
%     
%     date = data1(i,:);
%     imf=emd(date);
%     
%     [A,f,~]=hhspectrum(imf);
%     [E,~,Cenf]=toimage(A,f);
%     for k=1:DEFSPL
%         bjp(k)=sum(E(k,:))*1/dalt;
%     end
%     bjp = bjp(find(bjp~=0));
%     x1(i,1) = sum(-bjp.^2/sum(bjp.^2,2).*log10(bjp.^2/sum(bjp.^2,2)));
% end
%重心频率：
% for i = 1:250
%     x(i,:) = abs(fft(data3(i,:)));
%     y(i,1)=sum(x(i,:).^2.*data3(i,:))/sum(x(i,:));
% end



% 计算短时功率谱密度函数

% x是信号，nwind是每帧长度，noverlap是每帧重叠的样点数

% w_nwind是每段的窗函数，或相应的段长，

% w_noverlap是每段之间的重叠的样点数，nfft是FFT的长度
% for i = 1:250
% x = data1(i,:);
% noverlap = 1;
% inc=33-noverlap;       % 计算帧移
% 
% X=enframe(x,33,inc)';  % 分帧
% 
% frameNum=size(X,2);       % 计算帧数

%用pwelch函数对每帧计算功率谱密度函数

% for k=1 : frameNum
% 
%     Pxx(:,k)=pwelch(X(:,k));
% end
% H(i,1) = log10(sum(Pxx));
% end
% x1 = x(1:50);
% x2 = x(51:100);
% x3 = x(101:150);
% x4 = x(151:200);
% x5 = x(201:250);
% q = [x1,x2,x3,x4,x5];
% [trainData,testData,tr] = F_CV(q,6);
% testData;
% for i=1:250
%     s = 1;
%     x = [1:33];
%     data = data3(i,:);
%     m = 1;
%     while(s>0.01)
%         A = polyfit(x,data,m);
%         z = polyval(A,x);
%         s = sum((z-data).^2)/33;
%         m = m+1;
%     end
%     H(i,:) = polyval(A,x);
% end
% for i = 1:250
%    E(i,1) = log10(sum(data2(i,:).^2));
% end
% for i = 1:5
%     for j = 1:50
%         if (j == 50)
%             H((i-1)*50+j,1) = E((i-1)*50+1)-E(i*50);
%         else
%             H((i-1)*50+j,1)= E((i-1)*50+j+1,1)-E((i-1)*50+j,1);
%         end
%     end
% end
for i = 1:250
    data = data1(i,:);
%    E(i,1) = log10(sum(data.^2))/(max(data.^2)-min(data.^2));
%     E(i,1) = sum(data)/log10(sum(data.^2))/10000;
     E(i,1) = log10(columnkur(data));
end 

x = E;
x1 = x(1:50);
x2 = x(51:100);
x3 = x(101:150);
x4 = x(151:200);
x5 = x(201:250);
q = [x1,x2,x3,x4,x5];
[trainData,testData,tr] = F_CV(q,6);
testData;
