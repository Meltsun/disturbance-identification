clear all
close all
clc

data = xlsread('iris.xlsx');
m = 3;                     %�����
n = 4;                     %������
T = 7;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = n1_*T/10;             %��һ���ѵ��������
n2 = n2_*T/10;             %�ڶ����ѵ��������
s_ = n1_*(10-T)/10;        %ĳһ��Ĳ���������
k = 0;                     %��ֹ������������ֵ�ĳ���
s = 1;                     %��˹�˺����еķ���
c1 = 32;                 %��������
c2 = 32;                    %��������
c3 = c1;
c4 = c2;
ekcl = 0.15;
A1 = data(1:50,1:4);                  %��ȡ��һ��
B2 = data(51:100,1:4);                %��ȡ�ڶ���
C3 = data(101:150,1:4);               %��ȡ������
%% ѵ�����Ͳ��Լ�
[trDA,teDA,tr1] = F_CV(A1,T);            %A��ѵ�����Ͳ��Լ����±�
[trDB,teDB,tr2] = F_CV(B2,T);            %B��ѵ�����Ͳ��Լ����±�
[trDC,teDC,tr3] = F_CV(C3,T);            %C��ѵ�����Ͳ��Լ����±�
trD =[trDA;trDB;trDC];                   %ѵ����ѹ������
tr = [1,2,3];
c= 0;
for i = 1:m-1
    for j = i+1:m
        trD =[trDA;trDB;trDC];                   %ѵ����ѹ������
        trD1 = trD((i-1)*n1+1:i*n1,:); %��һ�� 
        trD2 = trD((j-1)*n1+1:j*n1,:); %�ڶ���
        %����
%         A = trD1;
%         B = trD2;
%         trD([(i-1)*n1+1:i*n1,(j-1)*n1+1:j*n1],:) = [];
%         C = trD;
%         e1 = ones(n1,1);
%         e2 = ones(n2,1);
%         e3 = ones(n1,1);
%         E = [A,e1];
%         F = [B,e2];
%         G = [C,e3];
%         R1 = inv(-c1*F'*F+E'*E+c2*G'*G)*(c1*F'*e2+c2*(1-ekcl)*G'*e3);
%         R2 = inv(-c3*E'*E+F'*F+c4*G'*G)*(c3*E'*e1+c4*(1-ekcl)*G'*e3);
%         w1 = R1(1:end-1);
%         b1 = R1(end);
%         w2 = R2(1:end-1);
%         b2 = R2(end);
          A = trD1;
          B = trD2;
          D = trD;
          trD([(i-1)*n1+1:i*n1,(j-1)*n1+1:j*n1],:) = [];
          C = trD;
          e1 = ones(n1,1);
          e2 = ones(n2,1);
          e3 = ones(n1,1);
          M = [rbf(A,D',0.5),e1];
          N = [rbf(B,D',0.5),e2];
          O = [rbf(C,D',0.5),e3];
          R1 = -inv(-c1*N'*N+M'*M+c2*O'*O)*(c1*N'*e2+c2*(1-ekcl)*O'*e3);
          R2 = inv(-c3*M'*M+N'*N+c4*O'*O)*(c3*M'*e1+c4*(1-ekcl)*O'*e3);
          w1 = R1(1:end-1);
          b1 = R1(end);
          w2 = R2(1:end-1);
          b2 = R2(end);
          
        c = c+1;
        %Ȩֵ����
        W1(c,:) = w1';
        W2(c,:) = w2';
        B1(c)   = b1;
        B2(c)   = b2;
    end
end
 %% ����������
flq = [1,2,3;1,3,2;2,3,1];
%% ���Լ�����ǩ
teD_z = [teDA;teDB;teDC];
biaoqian = [ones(s_,1)*1;
            ones(s_,1)*2;
            ones(s_,1)*3];
 %% ����
tic;
[row,col] = size(teD_z);
for i = 1:row
     for k = 1:3
%          result1 = W1(k,:)*teD_z(i,:)'+B1(k);
%          result2 = W2(k,:)*teD_z(i,:)'+B2(k);
           result1 = rbf(teD_z(i,:),D',0.5)*W1(k,:)'+B1(k);
           result2 = rbf(teD_z(i,:),D',0.5)*W2(k,:)'+B2(k);

         result1_(i,k) = result1;
         result2_(i,k) = result2; 
         if(result1>(ekcl-1))
             y(i,k) = flq(k,1);
         else if(result1<(1-ekcl))
                 y(i,k) =flq(k,2);
             else
                 y(i,k) = flq(k,3);
             end
         end
     end
end
for i =1:row
     table = tabulate(y(i,:));
     [~,idx] = max(table(:,2));
     vote_y(i,1) = idx;
end         
 Accuracy = 1-length(find((vote_y(:,1)-biaoqian)~=0))/row;