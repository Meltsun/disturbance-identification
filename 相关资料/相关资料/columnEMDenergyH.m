function [output] = columnEMDenergyH(input)
    %EMDÄÜÁ¿ìØ
    Data = input;
    H = [];
    imf = emd(Data);
    E = sum(imf.^2,2);
    P = 1/sum(E)*E;
    [n,~] = size(P);
%     for i =1:n
%         H(1,i) = -P(i,1)*log10(P(i,1));
%     end
%     output = -sum(P.*log(P));
%     output =sum(H);
 output = P(end);
end


