function [ output ] = columnKr( input )
%UNTITLED5 �������Ͷ�����
%   �˴���ʾ��ϸ˵��
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    y = input(i, :);
    v = sum((y-mean(y)).^2)/length(y);
    output(i, 1) = sum((y-mean(y)).^4)/(v.^2)/length(y);
    %output(i, 1) = sum(y.^4)/sqrt(sum(y.^2));
end
end