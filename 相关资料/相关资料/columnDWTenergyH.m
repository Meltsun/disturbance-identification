function [output] = columnDWTenergyH(input)
%小波能量熵
    Data = input;
    p =2;
    wpt  = wpdec(Data,p,'db3');
    H = [];
    X    = {};
    Y    = {};
    E    = [];
    q = 2;% 取第三层
    %得到第三层小波系数
    for i = 1:2^q
        X{1,i} = wpcoef(wpt,[q i-1]); 
    end
    %得到第三层各小波系数的重构信号
%     for i = 1:2^q
%         Y{1,i} = wprcoef(wpt,[p i-1]); 
%     end
%     %求能量
%     [~,k] = size(X);
%     for i = 1:k
%         E(1,i) = sum(X{1,i}.^2); 
%     end
     %能量归一化
%     P = 1/sum(E)*E;
%     [n,~] = size(P);
%     for i =1:n
%         H(1,i) = -P(i,1)*log10(P(i,1));
%     end
%     output = sum(H);
%     output = 1/10*log10(E(1,4));
     [~,S,~] = svd(X{1,1});
     output = S(1);
end

