function [ output ] = columnvar( input )
%columnvar
%   �����󷽲�
%
[m, ~] = size(input);
output = zeros(m, 1);
for i = 1:m
    output(i, 1) = var(input(i, :));
end
end