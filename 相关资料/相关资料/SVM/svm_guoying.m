clc
clear all
close all

%% ��ʼ������
m = 5;                     %�����
n = 6;                     %������
t = 10;                    %����ģ��������ѹ���ĸ���
T = 7;                     %������֤�ı���������6��4
n1_ = 50;                  %��һ���������
n2_ = 50;                  %��һ���������
n1 = n1_*T/10-t;           %��һ���ѵ��������
n2 = n2_*T/10-t;           %�ڶ����ѵ��������
s_ = n1_*(10-T)/10;        %ĳһ��Ĳ���������
k = 0;                     %��ֹ������������ֵ�ĳ���
s = 1;                     %��˹�˺����еķ���
DD = 100;
%% ��ȡ����
load alldatapot2_lable.mat
Data = alldatapot2_lable;
Data1 = feature_end(Data);           %�����¼ӵ�һ����������ӳ��ɢ������ɢ�̶�
qian = Data(:,end);                  %���һ��
data1 = mapminmax(Data(:,1:44),0,1); %������һ��
data = [data1,Data1,qian];           %�������
tz = [3,25,7,17,9,45];  %pot2 96%
A1 = data(1:50,tz);                  %��ȡ��һ��
B2 = data(51:100,tz);                %��ȡ�ڶ���
C3 = data(101:150,tz);               %��ȡ������
D4 = data(151:200,tz);               %��ȡ������
E5 = data(201:250,tz);               %��ȡ������
for h = 1:DD
[trDA,teDA,tr1] = F_CV(A1,T);            %A��ѵ�����Ͳ��Լ����±�
[trDB,teDB,tr2] = F_CV(B2,T);            %B��ѵ�����Ͳ��Լ����±�
[trDC,teDC,tr3] = F_CV(C3,T);            %C��ѵ�����Ͳ��Լ����±�
[trDD,teDD,tr4] = F_CV(D4,T);            %D��ѵ�����Ͳ��Լ����±�
[trDE,teDE,tr5] = F_CV(E5,T);            %E��ѵ�����Ͳ��Լ����±�

trD = yasuo(trDA,trDB,trDC,trDD,trDE,t);%ѵ����ѹ������
teD_z = [teDA;teDB;teDC;teDD;teDE];

%*** ���ò���
maxgen = 50; %��������
sizepop = 50; %��Ⱥ��ģ

%[ yy, Xbest, Ybest  ] = FOA( maxgen, sizepop );
X_axis = 5 * rand()+5;
Y_axis = 5 * rand()+5;

    %*** ��ӬѰ�ſ�ʼ���������Ѱ��ʳ��
for i=1 : sizepop

    %*** �����Ӭ�������������Ѱʳ��֮������������
    X(i) = X_axis + 2 * rand() - 1;
    Y(i) = Y_axis + 2 * rand() - 1;

    %*** �����޷���֪ʳ��λ�ã�����ȹ�����ԭ��ľ��루Dist�����ټ���ζ��Ũ���ж�ֵ��S������ֵΪ����ĵ���
    D(i) = (X(i)^2 + Y(i)^2)^0.5;
    S(i) = 1 / D(i);

    %*** ζ��Ũ���ж�ֵ��S������ζ��Ũ���ж����������ΪFitness function����������ù�Ӭ����λ�õ�ζ��Ũ�ȣ�Smell(i))
   % Smell(i) = Fitness(S(i));
    Smell(i) = PSO_twsvm_fitness(trDA,trDB,trDC,trDD,trDE,X(i),Y(i));

end

%*** �ҳ��˹�ӬȺ����ζ��Ũ����͵Ĺ�Ӭ����Сֵ��
% [bestSmell ,bestindex] = min(Smell);
[bestSmell ,bestindex] = max(Smell);
%*** �������ζ��Ũ��ֵ��x��y�����꣬��ʱ��ӬȺ�������Ӿ�����λ�÷�ȥ
X_axis = X(bestindex);
Y_axis = Y(bestindex);
Smellbest = bestSmell;

%*** ��Ӭ����Ѱ�ſ�ʼ
for g=1 : maxgen

    %*** �����Ӭ�������������Ѱʳ����������;���
    for i=1 : sizepop
        X(i) = X_axis + 2 * rand() - 1;
        Y(i) = Y_axis + 2 * rand() - 1;

        %*** �����޷���֪ʳ��λ�ã�����ȹ�����ԭ��ľ��루Dist�����ټ���ζ��Ũ���ж�ֵ��S������ֵΪ����ĵ���
        D(i) = (X(i)^2 + Y(i)^2)^0.5;
        S(i) = 1 / D(i);

        %*** ζ��Ũ���ж�ֵ��S������ζ��Ũ���ж�������������ù�Ӭ����λ�õ�ζ��Ũ�ȣ�Smell(i))
        Smell(i) = PSO_twsvm_fitness(trDA,trDB,trDC,trDD,trDE,X(i),Y(i));

    end;

    %*** �ҳ��˹�ӬȺ����ζ��Ũ����͵Ĺ�Ӭ����Сֵ��
    [bestSmell bestindex] = max(Smell);

    %*** �ж�ζ��Ũ���Ƿ�����ǰһ�ε���ζ��Ũ�ȣ������������ζ��Ũ��ֵ��x��y�����꣬��ʱ��ӬȺ�������Ӿ�����λ�÷�ȥ
    if bestSmell > Smellbest
        X_axis = X(bestindex);
        Y_axis = Y(bestindex);
        Smellbest = bestSmell;
    end;

    %*** ÿ������Semllֵ��¼��yy�����У�����¼���ŵ�������
    yy(g) = Smellbest;
    Xbest(g) = X_axis;
    Ybest(g) = Y_axis;

end;
Accuracy_train(h,1) = Smellbest;
Xbest(h) = X_axis;
Ybest(h) = Y_axis;
c1 = X_axis;
c2 = Y_axis;
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
                  if(abs(result1) > abs(result2))
                      jg(h,i) = flq(lib(7),2);
                  else
                      jg(h,i) = flq(lib(7),1);
                  end
               else 
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');      
                  if(abs(result1) > abs(result2))
                      jg(h,i) = flq(lib(8),2);
                  else
                      jg(h,i) = flq(lib(8),1);
                  end
               end  
            else 
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)'); 
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(8),2);
                  else
                       jg(h,i) = flq(lib(8),1);
                  end
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');  
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(9),2);
                  else
                       jg(h,i) = flq(lib(9),1);
                  end
               end
            end
        else
            result1 = (W1(lib(3),:)*teD_z(i,:)'+B1(lib(3)))/(W1(lib(3),:)*W1(lib(3),:)');
            result2 = (W2(lib(3),:)*teD_z(i,:)'+B2(lib(3)))/(W2(lib(3),:)*W2(lib(3),:)'); 
            if(abs(result1) > abs(result2))
               result1 = (W1(lib(5),:)*teD_z(i,:)'+B1(lib(5)))/(W1(lib(5),:)*W1(lib(5),:)');
               result2 = (W2(lib(5),:)*teD_z(i,:)'+B2(lib(5)))/(W2(lib(5),:)*W2(lib(5),:)');  
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(8),:)*teD_z(i,:)'+B1(lib(8)))/(W1(lib(8),:)*W1(lib(8),:)');
                  result2 = (W2(lib(8),:)*teD_z(i,:)'+B2(lib(8)))/(W2(lib(8),:)*W2(lib(8),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(8),2);
                  else
                       jg(h,i) = flq(lib(8),1);
                  end 
               else
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(9),2);
                  else
                       jg(h,i) = flq(lib(9),1);
                  end  
               end
            else
               result1 = (W1(lib(6),:)*teD_z(i,:)'+B1(lib(6)))/(W1(lib(6),:)*W1(lib(6),:)');
               result2 = (W2(lib(6),:)*teD_z(i,:)'+B2(lib(6)))/(W2(lib(6),:)*W2(lib(6),:)');   
               if(abs(result1) > abs(result2))
                  result1 = (W1(lib(9),:)*teD_z(i,:)'+B1(lib(9)))/(W1(lib(9),:)*W1(lib(9),:)');
                  result2 = (W2(lib(9),:)*teD_z(i,:)'+B2(lib(9)))/(W2(lib(9),:)*W2(lib(9),:)');   
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(9),2);
                  else
                       jg(h,i) = flq(lib(9),1);
                  end  
               else
                  result1 = (W1(lib(10),:)*teD_z(i,:)'+B1(lib(10)))/(W1(lib(10),:)*W1(lib(10),:)');
                  result2 = (W2(lib(10),:)*teD_z(i,:)'+B2(lib(10)))/(W2(lib(10),:)*W2(lib(10),:)');     
                  if(abs(result1) > abs(result2))
                       jg(h,i) = flq(lib(10),2);
                  else
                       jg(h,i) = flq(lib(10),1);
                  end     
               end
            end
        end
    end
jg = jg';
Accuracy_test(h,1) = 1-length(find((jg(:,h)-biaoqian)~=0))/row;
end


% %*** ���Ƶ���ζ��Ũ�����Ӭ����·������ͼ
% figure(1);
% plot(yy);
% title('Optimization process', 'fontsize', 12);
% xlabel('Iteration Number', 'fontsize', 12);
% ylabel('Smell', 'fontsize', 12);
% figure(2);
% plot(Xbest, Ybest, 'b.');
% title('Fruit fly flying route', 'fontsize', 14);
% xlabel('X-axis', 'fontsize', 12);
% ylabel('Y-axis', 'fontsize', 12);
%% ������ȷ��
A_j = length(find(jg(1:s_,:) == 1))/(DD*s_);             %������ȷ��
A_q = length(find(jg(s_+1:2*s_,:) == 2))/(DD*s_);        %�õ���ȷ��
A_y = length(find(jg(2*s_+1:3*s_,:) == 3))/(DD*s_);      %ѹ����ȷ��
A_p = length(find(jg(3*s_+1:4*s_,:) == 4))/(DD*s_);      %������ȷ��
A_n = length(find(jg(4*s_+1:5*s_,:) == 5))/(DD*s_);      %�޵���ȷ��
A_z = sum(Accuracy_test)/DD;                                      %ƽ����ȷ��


sprintf('���Ĳ���׼ȷ��=%0.2f',A_j)
sprintf('�õĲ���׼ȷ��=%0.2f',A_q)
sprintf('ѹ�Ĳ���׼ȷ��=%0.2f',A_y)
sprintf('���Ĳ���׼ȷ��=%0.2f',A_p)
sprintf('�޵Ĳ���׼ȷ��=%0.2f',A_n)
sprintf('�ܵĲ���׼ȷ��=%0.2f',A_z)