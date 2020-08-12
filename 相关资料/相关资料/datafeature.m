function [ output ] = datafeature( input )


[m, ~] = size(input);
output = zeros(m, 30);

D = 1;
for i = 1:m
    output(i, 1) = columnmax(input(i, :));                      %  ���ֵ
    output(i, 2) = columnmax(datacolumndiff(input(i, :), D));                      %  ���ֵ

    output(i, 3) = columnmin(input(i, :));                      %  ��Сֵ
    output(i, 4) = columnmin(datacolumndiff(input(i, :), D));                      %  ��Сֵ

    output(i, 5) = columnpk(input(i, :));                       %  ��ֵ
    output(i, 6) = columnpk(datacolumndiff(input(i, :), D));                       %  ��ֵ

    output(i, 7) = columnmean(input(i, :));                     %  ��ֵ   
    output(i, 8) = columnmean(datacolumndiff(input(i, :), D));                     %  ��ֵ   

    output(i, 9) = columnavmean(input(i, :));                   %  ����ƽ��ֵ
    output(i, 10) = columnavmean(datacolumndiff(input(i, :), D));                  %  ����ƽ��ֵ

    output(i, 11) = columnrms(input(i, :));                     %  ������
    output(i, 12) = columnrms(datacolumndiff(input(i, :), D));                     %  ������

    output(i, 13) = columnvar(input(i, :));                     %  ����
    output(i, 14) = columnvar(datacolumndiff(input(i, :), D));                     %  ����

    output(i, 15) = columnstd(input(i, :));                     %  ��׼��
    output(i, 16) = columnstd(datacolumndiff(input(i, :), D));                     %  ��׼��

    output(i, 17) = columnS(input(i, :));                       %  ��������
    output(i, 18) = columnS(datacolumndiff(input(i, :), D));                       %  ��������

    output(i, 19) = columnC(input(i, :));                       %  ��ֵ����
    output(i, 20) = columnC(datacolumndiff(input(i, :), D));                       %  ��ֵ����

    output(i, 21) = columnI(input(i, :));                       %  ��������
    output(i, 22) = columnI(datacolumndiff(input(i, :), D));                       %  ��������

    output(i, 23) = columnL(input(i, :));                       %  ԣ������
    output(i, 24) = columnL(datacolumndiff(input(i, :), D));                       %  ԣ������

    output(i, 25) = columnKr(input(i, :));                      %  �Ͷ�����
    output(i, 26) = columnKr(datacolumndiff(input(i, :), D));                      %  �Ͷ�����

    output(i, 27) = columnSK(input(i, :));                      %  ƫ������
    output(i, 28) = columnSK(datacolumndiff(input(i, :), D));                      %  ƫ������

    output(i, 29) = columnxcorr(input(i,:));                    %  �����ϵ����ֵ
    output(i, 30) = columnxcorr(datacolumndiff(input(i, :), D));                   %  �����ϵ����ֵ

%     output(i, 31) = columnE(input(i,:));                        %  ��ʱ����
%     output(i, 32) = columnE(datacolumndiff(input(i, :), D));                       %  ��ʱ����

%     output(i, 33) = columnDWTenergyH(input(i, :));              %  С��������
%     output(i, 34) = columnDWTenergyH(datacolumndiff(input(i, :),D));               %  С��������
% 
%     output(i, 35) = columnEMDenergyH(input(i, :));              %  EMD������
%     output(i, 36) = columnEMDenergyH(datacolumndiff(input(i, :),D));               %  EMD������
% 
%     output(i, 37) = columnSTFT(input(i,:));                     %  ��ʱ����Ҷ�任������
%     output(i, 38) = columnSTFT(datacolumndiff(input(i, :), D));                    %  ��ʱ����Ҷ�任������
end

