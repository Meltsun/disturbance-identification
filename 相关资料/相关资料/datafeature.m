function [ output ] = datafeature( input )


[m, ~] = size(input);
output = zeros(m, 30);

D = 1;
for i = 1:m
    output(i, 1) = columnmax(input(i, :));                      %  最大值
    output(i, 2) = columnmax(datacolumndiff(input(i, :), D));                      %  最大值

    output(i, 3) = columnmin(input(i, :));                      %  最小值
    output(i, 4) = columnmin(datacolumndiff(input(i, :), D));                      %  最小值

    output(i, 5) = columnpk(input(i, :));                       %  峰值
    output(i, 6) = columnpk(datacolumndiff(input(i, :), D));                       %  峰值

    output(i, 7) = columnmean(input(i, :));                     %  均值   
    output(i, 8) = columnmean(datacolumndiff(input(i, :), D));                     %  均值   

    output(i, 9) = columnavmean(input(i, :));                   %  整流平均值
    output(i, 10) = columnavmean(datacolumndiff(input(i, :), D));                  %  整流平均值

    output(i, 11) = columnrms(input(i, :));                     %  均方根
    output(i, 12) = columnrms(datacolumndiff(input(i, :), D));                     %  均方根

    output(i, 13) = columnvar(input(i, :));                     %  方差
    output(i, 14) = columnvar(datacolumndiff(input(i, :), D));                     %  方差

    output(i, 15) = columnstd(input(i, :));                     %  标准差
    output(i, 16) = columnstd(datacolumndiff(input(i, :), D));                     %  标准差

    output(i, 17) = columnS(input(i, :));                       %  波形因子
    output(i, 18) = columnS(datacolumndiff(input(i, :), D));                       %  波形因子

    output(i, 19) = columnC(input(i, :));                       %  峰值因子
    output(i, 20) = columnC(datacolumndiff(input(i, :), D));                       %  峰值因子

    output(i, 21) = columnI(input(i, :));                       %  脉冲因子
    output(i, 22) = columnI(datacolumndiff(input(i, :), D));                       %  脉冲因子

    output(i, 23) = columnL(input(i, :));                       %  裕度因子
    output(i, 24) = columnL(datacolumndiff(input(i, :), D));                       %  裕度因子

    output(i, 25) = columnKr(input(i, :));                      %  峭度因子
    output(i, 26) = columnKr(datacolumndiff(input(i, :), D));                      %  峭度因子

    output(i, 27) = columnSK(input(i, :));                      %  偏度因子
    output(i, 28) = columnSK(datacolumndiff(input(i, :), D));                      %  偏度因子

    output(i, 29) = columnxcorr(input(i,:));                    %  自相关系数均值
    output(i, 30) = columnxcorr(datacolumndiff(input(i, :), D));                   %  自相关系数均值

%     output(i, 31) = columnE(input(i,:));                        %  短时能量
%     output(i, 32) = columnE(datacolumndiff(input(i, :), D));                       %  短时能量

%     output(i, 33) = columnDWTenergyH(input(i, :));              %  小波能量熵
%     output(i, 34) = columnDWTenergyH(datacolumndiff(input(i, :),D));               %  小波能量熵
% 
%     output(i, 35) = columnEMDenergyH(input(i, :));              %  EMD能量熵
%     output(i, 36) = columnEMDenergyH(datacolumndiff(input(i, :),D));               %  EMD能量熵
% 
%     output(i, 37) = columnSTFT(input(i,:));                     %  短时傅里叶变换能量熵
%     output(i, 38) = columnSTFT(datacolumndiff(input(i, :), D));                    %  短时傅里叶变换能量熵
end

