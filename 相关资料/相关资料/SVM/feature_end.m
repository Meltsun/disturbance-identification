function data1  = feature_end(data)
% tz = (1:44);
 tz = (1:32);
m = 50;
A1 = data(1:50,tz);
B2 = data(51:100,tz);
C3 = data(101:150,tz);
D4 = data(151:200,tz);
E5 = data(201:250,tz);
CA = mean(A1,1);
CB = mean(B2,1);
CC = mean(C3,1);
CD = mean(D4,1);
CE = mean(E5,1);
for i = 1:m
    A(i,1) = norm(A1(i,:)-CA);
    B(i,1) = norm(B2(i,:)-CB);        
    C(i,1) = norm(C3(i,:)-CC);        
    D(i,1) = norm(D4(i,:)-CD);        
    E(i,1) = norm(E5(i,:)-CE);        
end
data1 = [A;B;C;D;E]; 
end