function [ output ] = columnS( input )
%UNTITLED5 ������������
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    output(i, 1) = rms(y)/mean(abs(y));
end
end