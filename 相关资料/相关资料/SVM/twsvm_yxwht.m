%�����޻�ͼ
clc
clear all
close all

%% ��ʼ������
m = 5;                     %�����
T = 6;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = n1_*T/10;             %��һ���ѵ��������
n2 = n2_*T/10;             %�ڶ����ѵ��������
s_ = n1_*(10-T)/10;        %ĳһ��Ĳ���������
k = 0.0001;                %��ֹ������������ֵ�ĳ���
s = 1;                     %��˹�˺����еķ���
c1 = 8.9944;               %��������
c2 = 7.5212;               %��������
c3 = 0;                    %��������
c4 = 0;                    %��������
lambda1 = 4;               %��������                     
lambda2 = 1;               %��������
DD = 1;                    %ѵ������
%% ��ȡ����
data = xlsread('D:\PyCharm Edu 2018.3\ML\selfCodes\new\p1z.xlsx');

A1 = data(1:50,1:end-1);                  %��ȡ��һ��
B2 = data(51:100,1:end-1);                %��ȡ�ڶ���
C3 = data(101:150,1:end-1);               %��ȡ������
D4 = data(151:200,1:end-1);               %��ȡ������
E5 = data(201:250,1:end-1);               %��ȡ������
%% ѵ�����Ͳ��Լ�
for g = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A��ѵ�����Ͳ��Լ����±�
[trDB,teDB,tr2] = F_CV(B2,T);            %B��ѵ�����Ͳ��Լ����±�
[trDC,teDC,tr3] = F_CV(C3,T);            %C��ѵ�����Ͳ��Լ����±�
[trDD,teDD,tr4] = F_CV(D4,T);            %D��ѵ�����Ͳ��Լ����±�
[trDE,teDE,tr5] = F_CV(E5,T);            %E��ѵ�����Ͳ��Լ����±�

trD = [trDA;trDB;trDC;trDD;trDE];
%% ѵ��

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
%             e1 =ones(n1,1);             
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
    yxwht = [1,5;2,5;1,4;3,5;2,4;1,3;4,5;3,4;2,3;1,2];
    for i = 1:c
        for j = 1:c
           if(yxwht(i,:) == flq(j,:))
               lib(i) = j;
           end
        end
    end       
    teD_z = [teDA;teDB;teDC;teDD;teDE];
    biaoqian = [ones(s_,1)*1;
                ones(s_,1)*2;
                ones(s_,1)*3;
                ones(s_,1)*4;
                ones(s_,1)*5];
    [row,col] = size(teD_z);   
    
    tic;
% for i = 1:row
%     jl1 = sqrt(sum((repmat(teD_z(i,:),n1*5,1)-trD).^2,2));
%     jl(i,1) = sum(jl1(1:n1));
%     jl(i,2) = sum(jl1(n1+1:2*n1));
%     jl(i,3) = sum(jl1(2*n1+1:3*n1));
%     jl(i,4) = sum(jl1(3*n1+1:4*n1));
%     jl(i,5) = sum(jl1(4*n1+1:5*n1));
%     
% end  
%     jl = mapminmax(jl,0,1);
%     
    
    
    
    
    
    
    % ���������޻�ͼ 10��������
    for i = 1:row
        result1 = (W1(lib(1),:)*teD_z(i,:)'+B1(lib(1)))/(W1(lib(1),:)*W1(lib(1),:)');
        result2 = (W2(lib(1),:)*teD_z(i,:)'+B2(lib(1)))/(W2(lib(1),:)*W2(lib(1),:)');
        if(abs(result1) > abs(result2))
%                te_(i,k) = flq(lib(1),2);
            result1 = (W1(lib(2),:)*teD_z(i,:)'+B1(lib(2)))/(W1(lib(2),:)*W1(lib(2),:)');
            result2 = (W2(lib(2),:)*teD_z(i,:)'+B2(lib(2)))/(W2(lib(2),:)*W2(lib(2),:)'); 
            if(abs(result1) > abs(result2))   
               result1 = (W1(lib(4),:)*teD_z(i,:)'+B1(lib(4)))/(W1(lib(4),:)*W1(lib(4),:)');
               result2 = (W2(lib(4),:)*teD_z(i,:)'+B2(lib(4)))/(W2(lib(4),:)*W2(lib(4),:)');        
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(7),:)*teD_z(i,:)'+B1(lib(7)))/(W1(lib(7),:)*W1(lib(7),:)');
                  result2 = (W2(lib(7),:)*teD_z(i,:)'+B2(lib(7)))/(W2(lib(7),:)*W2(lib(7),:)');  
%                     result1 = jl(i,flq(lib(7),1));
%                     result2 = jl(i,flq(lib(7),2));
                  if(abs(result1) > abs(result2))
                      jg(g,i) = flq(lib(7),2);
                  else
                      jg(g,i) = flq(lib(7),1);
                  end
               else 
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');      
%                     result1 = jl(i,flq(lib(8),1));
%                     result2 = jl(i,flq(lib(8),2));

                  if(abs(result1) > abs(result2))
                      jg(g,i) = flq(lib(8),2);
                  else
                      jg(g,i) = flq(lib(8),1);
                  end
               end  
            else 
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
%                result1 = jl(i,flq(lib(5),1));
%                result2 = jl(i,flq(lib(5),2));
               
               
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
%                     result1 = jl(i,flq(lib(8),1));
%                     result2 = jl(i,flq(lib(8),2));
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(8),2);
                  else
                       jg(g,i) = flq(lib(8),1);
                  end
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');  
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(9),2);
                  else
                       jg(g,i) = flq(lib(9),1);
                  end
               end
            end
        else
            result1 = (W1(lib(3),:)*teD_z(i,:)'+B1(lib(3)))/(W1(lib(3),:)*W1(lib(3),:)');
            result2 = (W2(lib(3),:)*teD_z(i,:)'+B2(lib(3)))/(W2(lib(3),:)*W2(lib(3),:)'); 
%               result1 = jl(i,flq(lib(3),1));
%               result2 = jl(i,flq(lib(3),2));
            
            if(abs(result1) > abs(result2))
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)');  
               
%                result1 = jl(i,flq(lib(5),1));
%                result2 = jl(i,flq(lib(5),2));
               
               
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  
                  
%                   result1 = jl(i,flq(lib(8),1));
%                   result2 = jl(i,flq(lib(8),2));
                  
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(8),2);
                  else
                       jg(g,i) = flq(lib(8),1);
                  end 
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(9),2);
                  else
                       jg(g,i) = flq(lib(9),1);
                  end  
               end
            else
               result1 = (W1(lib(6),:)*teD_z(i,:)'+B1(lib(6)))/(W1(lib(6),:)*W1(lib(6),:)');
               result2 = (W2(lib(6),:)*teD_z(i,:)'+B2(lib(6)))/(W2(lib(6),:)*W2(lib(6),:)');   
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(9),2);
                  else
                       jg(g,i) = flq(lib(9),1);
                  end  
               else
                  result1 = (W1(lib(10),:)*teD_z(i,:)'+B1(lib(10)))/(W1(lib(10),:)*W1(lib(10),:)');
                  result2 = (W2(lib(10),:)*teD_z(i,:)'+B2(lib(10)))/(W2(lib(10),:)*W2(lib(10),:)');     
                  if(abs(result1) > abs(result2))
                       jg(g,i) = flq(lib(10),2);
                  else
                       jg(g,i) = flq(lib(10),1);
                  end     
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
A_y = length(find(jg(2*s_+1:3*s_,:) == 3))/(DD*s_);      %������ȷ��
A_p = length(find(jg(3*s_+1:4*s_,:) == 4))/(DD*s_);      %ѹ����ȷ��
A_n = length(find(jg(4*s_+1:5*s_,:) == 5))/(DD*s_);      %�޵���ȷ��
A_z = sum(Accuracy)/DD;                                  %ƽ����ȷ��


sprintf('���Ĳ���׼ȷ��=%0.2f',A_j)
sprintf('�õĲ���׼ȷ��=%0.2f',A_q)
sprintf('���Ĳ���׼ȷ��=%0.2f',A_y)
sprintf('ѹ�Ĳ���׼ȷ��=%0.2f',A_p)
sprintf('�޵Ĳ���׼ȷ��=%0.2f',A_n)
sprintf('�ܵĲ���׼ȷ��=%0.2f',A_z)
    