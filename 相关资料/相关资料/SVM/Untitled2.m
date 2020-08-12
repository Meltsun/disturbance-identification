clc
clear al
close all

%% ��ʼ������
m = 5;                     %�����
n = 6;                     %������
t = 15;                    %����ģ��������ѹ���ĸ���
T = 7;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = 35;           %��һ���ѵ��������
n2 = 35;           %�ڶ����ѵ��������
s_ = 15;        %ĳһ��Ĳ���������
k = 0;                     %��ֹ������������ֵ�ĳ���
s = 1;                     %��˹�˺����еķ���
c1 = 8.9944;               %��������
c2 = 7.5212;               %��������
c3 = 0;                    %��������
c4 = 0;                    %��������
lambda1 = 4;               %��������                     
lambda2 = 1;               %��������
DD = 1;                    %ѵ������
%% ��ȡ����
load alldatapot1_lable.mat
Data = alldatapot1_lable;
Data1 = feature_end(Data);           %�����¼ӵ�һ����������ӳ��ɢ������ɢ�̶�
qian = Data(:,end);                  %���һ��
data1 = mapminmax(Data(:,1:44),0,1); %������һ��
data = [data1,Data1,qian];           %�������

tz = randi(44,1,5);                  %������5������
tz = [tz,45];                        %�����һ���������ȥ
%tz = [40,8,32,6,21,45]; %pot1 76%  
%tz = [25,43,9,2,20,45]; %pot2 90%
%tz = [1,35,18,28,43,45];%pot2 84% ����������ȫ��
tz = [3,25,7,17,9,45];  %pot2 96%
%tz = [35,9,14,28,11,45];%pot2 94% 0.026s
%tz = [22,23,42,17,7,45];%pot3 89% 0.025s
%tz = [39,43,11,34,36,45];%pot1 78% 0.024s
A1 = data(1:50,tz);                  %��ȡ��һ��
B2 = data(51:100,tz);                %��ȡ�ڶ���
C3 = data(101:150,tz);               %��ȡ������
D4 = data(151:200,tz);               %��ȡ������
E5 = data(201:250,tz);               %��ȡ������
SA = lsd(A1,0);
SB = lsd(B2,0);
SC = lsd(C3,0);
SD = lsd(D4,0);
SE = lsd(E5,0);
[~,pA] = sort(SA,'descend');%��С����
[~,pB] = sort(SB,'descend');
[~,pC] = sort(SC,'descend');
[~,pD] = sort(SD,'descend');
[~,pE] = sort(SE,'descend');
A_ = A1(pA(1:end-s),:);
B_ = B2(pB(1:end-s),:);
C_ = C3(pC(1:end-s),:);
D_ = D4(pD(1:end-s),:);
E_ = E5(pE(1:end-s),:);
trD = [A_;B_;C_;D_;E_];
A_1 = A1(pA(end-s+1:end),:);
B_1 = B2(pB(end-s+1:end),:);
C_1 = C3(pC(end-s+1:end),:);
D_1 = D4(pD(end-s+1:end),:);
E_1 = E5(pE(end-s+1:end),:);
teD_z = [A_1;B_1;C_1;D_1;E_1];
%% ѵ�����Ͳ��Լ�
% [trDA,teDA,tr1] = F_CV(A,T);            %A��ѵ�����Ͳ��Լ����±�
% [trDB,teDB,tr2] = F_CV(B,T);            %B��ѵ�����Ͳ��Լ����±�
% [trDC,teDC,tr3] = F_CV(C,T);            %C��ѵ�����Ͳ��Լ����±�
% [trDD,teDD,tr4] = F_CV(D,T);            %D��ѵ�����Ͳ��Լ����±�
% [trDE,teDE,tr5] = F_CV(E,T);            %E��ѵ�����Ͳ��Լ����±�
% 
% trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%ѵ����ѹ������
% trD = [trDA;trDB;trDC;trDD;trDE];
%% ѵ��
for g = 1:DD
    c = 0;                                 %����
    for i = 1:m-1
        for j = i+1:m
            trD1 = trD((i-1)*n1+1:i*n1,:); %��һ�� 
            trD2 = trD((j-1)*n1+1:j*n1,:); %�ڶ���

            %����TLDM����
%             X  = [trD1;trD2];
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             e  = ones(n1+n2,1);
%             Y = [ones(n1,1);-ones(n2,1)];
%             M  = [X,e];
%             D  = M'*(ones(n1+n2,n1+n2)-Y*Y');
%             H  = [trD1,e1];
%             G  = [trD2,e2];
%             Q  = lambda2/(n1+n2)*M'*Y;
%             P  = lambda1/(n1+n2).^2*D*M+H'*H; 
%             S  = lambda1/(n1+n2).^2*D*M+G'*G; 
%             [w1,b1,w2,b2] = TLDMtrain(H,Q,P,G,S,e1,e2,n1,n2,c1,c2);

            %����twsvm���Ժ˺�������
            e1 = ones(n1,1);
            e2 = ones(n2,1);
            H = [trD1,e1];
            G = [trD2,e2];
            P = [trD1,e1];
            Q = [trD2,e2];
            [w1,b1,w2,b2] = twsvmtrain(H,G,P,Q,e1,e2,c1,c2,n1,n2,k,T);

            %����twsvm��˹�˺�������
%             A = trD1;
%             B = trD2;
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             C = [A',B']';
%             S = [rbf(A,C',s),e1];
%             R = [rbf(B,C',s),e2];
%             L = [rbf(A,C',s),e1];
%             N = [rbf(B,C',s),e2];
%             [w1,b1,w2,b2] = twsvmtrainRBF(S,R,L,N,e1,e2,c1,c2,50,50,k,T);

            %����LSTWSVM
%             e1 = ones(n1,1);
%             e2 = ones(n2,1);
%             H = [trD1,e1];
%             G = [trD2,e2];
%             I = ones(size(H'*H,1),size(H'*H,1));
%             u1 = -c1*inv(H'*H+c1*(G'*G)+c3*I)*G'*e2;
%             u2 = -c2*inv(G'*G+c2*(H'*H)+c4*I)*H'*e1;
%             w1 = u1(1:end-1);
%             b1 = u1(end);
%             w2 = u2(1:end-1);
%             b2 = u2(end);

            %�������ĸ���
            c = c+1;
            %Ȩֵ����
            W1(c,:) = w1';
            W2(c,:) = w2';
            B1(c)   = b1;
            B2(c)   = b2;
        end
    end
    %% ����������
    flq = [1,2;1,3;1,4;1,5;2,3;2,4;2,5;3,4;3,5;4,5];
    %% ���Լ�����ǩ
%     teD_z = [teDA;teDB;teDC;teDD;teDE];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;
                ones(s_,1)*5];
    %% ����
    tic;
    [row,col] = size(teD_z);
    % ��һ��
    %���������
    trA_mean = mean(trDA,1);
    trB_mean = mean(trDB,1);
    trC_mean = mean(trDC,1);
    trD_mean = mean(trDD,1);
    trE_mean = mean(trDE,1);
    for i = 1:row
         S1 = zeros(n,n);
         S2 = zeros(n,n);
         S3 = zeros(n,n);
         S4 = zeros(n,n);
         S5 = zeros(n,n);
         for k = 1:n1
             S1 = S1+(trDA(k,:)-trA_mean)'*(trDA(k,:)-trA_mean);
         end
         for k = 1:n1
             S2 = S2+(trDB(k,:)-trB_mean)'*(trDB(k,:)-trB_mean);
         end
         for k = 1:n1
             S3 = S3+(trDC(k,:)-trC_mean)'*(trDC(k,:)-trC_mean);
         end

         for k = 1:n1
             S4 = S4+(trDD(k,:)-trD_mean)'*(trDD(k,:)-trD_mean);
         end

         for k = 1:n1
             S5 = S5+(trDE(k,:)-trE_mean)'*(trDE(k,:)-trE_mean);
         end
         d1 = sqrt((teD_z(i,:)-trA_mean)*inv(S1)*(teD_z(i,:)-trA_mean)');
         d2 = sqrt((teD_z(i,:)-trB_mean)*inv(S2)*(teD_z(i,:)-trB_mean)');
         d3 = sqrt((teD_z(i,:)-trC_mean)*inv(S3)*(teD_z(i,:)-trC_mean)');
         d4 = sqrt((teD_z(i,:)-trD_mean)*inv(S4)*(teD_z(i,:)-trD_mean)');
         d5 = sqrt((teD_z(i,:)-trE_mean)*inv(S5)*(teD_z(i,:)-trE_mean)');
         [~,px] = sort([d1,d2,d3,d4,d5]);
         pxjg(i,:) = sort(px(1:2));
         pxjg_(i,:) = px;
    end 
     for i = 1:row
         for j = 1:c
             if((flq(j,1) == pxjg(i,1))&&(flq(j,2) == pxjg(i,2)))
                jg_(i,1) = j;
             end
         end
     end
    %�ڶ�������
    for i = 1:row
        for k = 1:1
            %�޺�
            result1 = (W1(jg_(i,k),:)*teD_z(i,:)'+B1(jg_(i,k)))/(W1(jg_(i,k),:)*W1(jg_(i,k),:)');
            result2 = (W2(jg_(i,k),:)*teD_z(i,:)'+B2(jg_(i,k)))/(W2(jg_(i,k),:)*W2(jg_(i,k),:)');
            %�к�
    %         result1 = W1(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B1(jg_(i,k));
    %         result2 = W2(jg_(i,k),:)*rbf(teD_z(i,:),C',s)'+B2(jg_(i,k));
            if(abs(result1) > abs(result2))
               te_(i,k) = flq(jg_(i,k),2);
            else
               te_(i,k) = flq(jg_(i,k),1);
            end 
            result(i,1) = abs(result1);
            result(i,2) = abs(result2);
        end
    end
    %ͶƱ
     for i =1:row
         table = tabulate(te_(i,:));
         [~,idx] = max(table(:,2));
         vote_y(i,g) = idx;
     end
     Accuracy(g,1) = 1-length(find((vote_y(:,g)-biaoqian)~=0))/row;
end 
toc;
%% ��ͼ
figure(1)
plot(vote_y,'og')
hold on
plot(biaoqian,'r*');
legend('Ԥ���ǩ','ʵ�ʱ�ǩ')
title('twsvmԤ�������ʵ�����ȶ�','fontsize',12)
ylabel('����ǩ','fontsize',12)
xlabel('������Ŀ','fontsize',12)
%% ������ȷ��
A_j = length(find(vote_y(1:s_,:) == 1))/(DD*s_);             %������ȷ��
A_q = length(find(vote_y(s_+1:2*s_,:) == 2))/(DD*s_);        %�õ���ȷ��
A_y = length(find(vote_y(2*s_+1:3*s_,:) == 3))/(DD*s_);      %ѹ����ȷ��
A_p = length(find(vote_y(3*s_+1:4*s_,:) == 4))/(DD*s_);      %������ȷ��
A_n = length(find(vote_y(4*s_+1:5*s_,:) == 5))/(DD*s_);      %�޵���ȷ��
A_z = sum(Accuracy)/DD;                                      %ƽ����ȷ��


sprintf('���Ĳ���׼ȷ��=%0.2f',A_j)
sprintf('�õĲ���׼ȷ��=%0.2f',A_q)
sprintf('ѹ�Ĳ���׼ȷ��=%0.2f',A_y)
sprintf('���Ĳ���׼ȷ��=%0.2f',A_p)
sprintf('�޵Ĳ���׼ȷ��=%0.2f',A_n)
sprintf('�ܵĲ���׼ȷ��=%0.2f',A_z)
