function [ output ] = mapnormal( input )
%mapnormal L2·¶Êý¹éÒ»»¯
%   
[m, ~] = size(input);
% normalize each row to unit
output = input;
for i = 1:m
    output(i,:)= input(i,:)/norm(input(i,:));
end
end

