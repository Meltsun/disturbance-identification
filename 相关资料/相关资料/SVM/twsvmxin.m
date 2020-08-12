%�Ȱ����ֳ�����ʣ��������޻�ͼ
clc
clear all
close all

%% ��ʼ������
m = 4;                     %�����
n = 6;                     %������
t = 10;                    %����ģ��������ѹ���ĸ���
T = 8;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = n1_*T/10-t;           %��һ���ѵ��������
n2 = n2_*T/10-t;           %�ڶ����ѵ��������
s_ = n1_*(10-T)/10;        %ĳһ��Ĳ���������
k = 0.1;                     %��ֹ������������ֵ�ĳ���
s = 1;                     %��˹�˺����еķ���
c1 = 8.9944;               %��������
c2 = 7.5212;               %��������
DD = 100;
%% ��������
load alldatapot3_lable.mat
Data = alldatapot3_lable;
%���������ݷ���������
Data = Data(:,1:end-1);
Data1 = Data(151:200,:);
Data(151:200,:) = [];
Data = [Data;Data1];
qian = [ones(n1_,1)*1;
        ones(n1_,1)*2;
        ones(n1_,1)*3;
        ones(n1_,1)*4;
        ones(n1_,1)*5];
Data = [Data,qian];   %����ѹ����
Data1 = feature_end(Data);           %�����¼ӵ�һ����������ӳ��ɢ������ɢ�̶�
qian = Data(:,end);                  %���һ��
data1 = mapminmax(Data(:,1:44),0,1); %������һ��
data = [data1,Data1,qian];           %�������

tz = randi(44,1,5);                  %������5������
tz = [tz,45];                        %�����һ���������ȥ
 %tz = [40,8,32,6,21,45]; %pot1 76%  
%tz = [25,43,9,2,20,45]; %pot2 90%
%tz = [1,35,18,28,43,45];%pot2 84% ����������ȫ��
%tz = [3,25,7,17,9,45];  %pot2 96%
%tz = [35,9,14,28,11,45];%pot2 94% 0.026s
tz = [22,23,42,17,7,45];%pot3 89% 0.025s
%tz = [39,43,11,34,36,45];%pot1 78% 0.024s
%tz = [18,39,10,19,30,45];%pot3
%tz = [15,17,3,24,30,45];
tz = [1,5,13,15,27,45]; %pot3

%tz = [2,5,25,19,28,45];

A1 = data(1:50,tz);                  %��ȡ��һ��
B2 = data(51:100,tz);                %��ȡ�ڶ���
C3 = data(101:150,tz);               %��ȡ������
D4 = data(151:200,tz);               %��ȡ������
E5 = data(201:250,tz);               %��ȡ������
%% ѵ�����Ͳ��Լ�
for g = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A��ѵ�����Ͳ��Լ����±�
[trDB,teDB,tr2] = F_CV(B2,T);            %B��ѵ�����Ͳ��Լ����±�
[trDC,teDC,tr3] = F_CV(C3,T);            %C��ѵ�����Ͳ��Լ����±�
[trDD,teDD,tr4] = F_CV(D4,T);            %D��ѵ�����Ͳ��Լ����±�
[trDE,teDE,tr5] = F_CV(E5,T);            %E��ѵ�����Ͳ��Լ����±�

trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%ѵ����ѹ������
 c = 0;                                 %����
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %��һ�� 
            trD2 = trD((j-1)*n1+1:j*n1,:); %�ڶ���

%����twsvm���Ժ˺�������
            e1 = ones(n1,1);
            e2 = ones(n2,1);
            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];
            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);
 c = c+1;
            %Ȩֵ����
            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
        end
    end
    %% ����������
    flq = [1,2;1,3;1,4;2,3;2,4;3,4];
    yxwht = [1,4;2,4;1,3;3,4;2,3;1,2];
    for i = 1:c
        for j = 1:c
           if(yxwht(i,:) == flq(j,:))
               lib(i) = j;
           end
        end
    end       
    teD_z = [teDA;teDB;teDC;teDD];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;];
    [row,col] = size(teD_z);   
    tic;
    %�����޻�ͼ
     for i = 1:row
        result1 = (W1(lib(1),:)*teD_z(i,:)'+B1(lib(1)))/(W1(lib(1),:)*W1(lib(1),:)');
        result2 = (W2(lib(1),:)*teD_z(i,:)'+B2(lib(1)))/(W2(lib(1),:)*W2(lib(1),:)');
        if(abs(result1) > abs(result2))     
            result1 = (W1(lib(2),:)*teD_z(i,:)'+B1(lib(2)))/(W1(lib(2),:)*W1(lib(2),:)');
            result2 = (W2(lib(2),:)*teD_z(i,:)'+B2(lib(2)))/(W2(lib(2),:)*W2(lib(2),:)'); 
            if(abs(result1) > abs(result2))  
                result1 = (W1(lib(4),:)*teD_z(i,:)'+B1(lib(4)))/(W1(lib(4),:)*W1(lib(4),:)');
                result2 = (W2(lib(4),:)*teD_z(i,:)'+B2(lib(4)))/(W2(lib(4),:)*W2(lib(4),:)'); 
                if(abs(result1) > abs(result2))  
                      jg(g,i) = flq(lib(4),2);
                else
                      jg(g,i) = flq(lib(4),1);
                end
            else
                result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
                result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
                if(abs(result1) > abs(result2))  
                      jg(g,i) = flq(lib(5),2);
                else
                      jg(g,i) = flq(lib(5),1);
                end
            end
        else
            result1 = (W1(lib(3),:)*teD_z(i,:)'+B1(lib(3)))/(W1(lib(3),:)*W1(lib(3),:)');
            result2 = (W2(lib(3),:)*teD_z(i,:)'+B2(lib(3)))/(W2(lib(3),:)*W2(lib(3),:)'); 
            if(abs(result1) > abs(result2))  
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
               if(abs(result1) > abs(result2))  
                  jg(g,i) = flq(lib(5),2);
               else
                  jg(g,i) = flq(lib(5),1);
               end 
            else
               result1 = (W1(lib(6),:)*teD_z(i,:)'+B1(lib(6)))/(W1(lib(6),:)*W1(lib(6),:)');
               result2 = (W2(lib(6),:)*teD_z(i,:)'+B2(lib(6)))/(W2(lib(6),:)*W2(lib(6),:)'); 
                if(abs(result1) > abs(result2))  
                  jg(g,i) = flq(lib(6),2);
               else
                  jg(g,i) = flq(lib(6),1);
               end 
            end
        end
     end
ll = jg(g,:)';
Accuracy(g,1) = 1-length(find((ll-biaoqian)~=0))/row;
toc;
end
jg = jg';
 %% ��ͼ
figure(1)
plot(jg,'og')
hold on
plot(biaoqian,'r*');
legend('Ԥ���ǩ','ʵ�ʱ�ǩ')
title('twsvmԤ�������ʵ�����ȶ�','fontsize',12)
ylabel('����ǩ','fontsize',12)
xlabel('������Ŀ','fontsize',12)
%% ������ȷ��
A_j = length(find(jg(1:s_,:) == 1))/(DD*s_);             %������ȷ��
A_q = length(find(jg(s_+1:2*s_,:) == 2))/(DD*s_);        %�õ���ȷ��
A_y = length(find(jg(2*s_+1:3*s_,:) == 3))/(DD*s_);      %ѹ����ȷ��
A_n = length(find(jg(3*s_+1:4*s_,:) == 4))/(DD*s_);      %�޵���ȷ��

A_z = sum(Accuracy)/DD;                                  %ƽ����ȷ��


sprintf('���Ĳ���׼ȷ��=%0.2f',A_j)
sprintf('�õĲ���׼ȷ��=%0.2f',A_q)
sprintf('ѹ�Ĳ���׼ȷ��=%0.2f',A_y)
sprintf('�޵Ĳ���׼ȷ��=%0.2f',A_n)
sprintf('�ܵĲ���׼ȷ��=%0.2f',A_z)
    

 teD = [teDA;teDB;teDC;teDD;teDE];
for i = 1:row
    jl1 = sqrt(sum((repmat(teD(i,:),n1*5,1)-trD).^2,2));
    jl(i,1) = sum(jl1(1:n1));
    jl(i,2) = sum(jl1(n1+1:2*n1));
    jl(i,3) = sum(jl1(2*n1+1:3*n1));
    jl(i,4) = sum(jl1(3*n1+1:4*n1));
    jl(i,5) = sum(jl1(4*n1+1:5*n1));
    
end  
    