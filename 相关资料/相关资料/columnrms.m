function [ output ] = columnrms( input )
%UNTITLED5 �����������
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = rms(input(i, :));
end
end

