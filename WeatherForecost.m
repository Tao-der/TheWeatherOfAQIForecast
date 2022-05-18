clc
clear
%%  ����ĳ��2021��Ŀ�����Ⱦ���ݴ��BP������Ԥ��ģ�ͣ�
%���ֶ�����������Ӱ��AQIֵ���������Կ�����������ģ������  
          
%ԭʼ��������
    X_Oringin = [xlsread('C:\Users\HP\Desktop\yiyang2021aqi.xlsx','Sheet1')]';
%Ŀ�꣨��������ݾ���
    Y_Oringin = [xlsread('C:\Users\HP\Desktop\yiyang2021aqi.xlsx','Sheet2')]';
%     X_Oringin_test = [xlsread('C:\Users\HP\Desktop\AQIW.csv')]';
%     Y_Oringin_test = [xlsread('C:\Users\HP\Desktop\AQI.csv')]';

%% ��һ��
 [X_train,PS] = mapminmax(X_Oringin,-1,1);%-1 1֮������
 [Y_train,PS] = mapminmax(Y_Oringin,-1,1);
    %% �������������
    m=0;  
    In=length(X_train(1,:));                                         %�������Ԫ��
    H1=In-5; 
    H2=H1-4;
    Out=1;                                          %�������Ԫ��                       
    w1=unifrnd(-1,1,In,H1);      %��ʼ��
    w2=unifrnd(-1,1,H1,H2);         %�����ʼ��
    w3=unifrnd(-1,1,H2,Out);  
    %}  
    Epoch=100;   %���ݼ����ظ�������ѵ��
    batch=1;%һ��ѵ�������ݸ���%
    Iteration=10;%һ��batchѵ����������
    Lr=0.001/Iteration+0.0001;    
    
    %% ѵ��
    while (m<=Epoch)                                   %����5000�ֵ���
    
        m=m+1;
        
       for  i=1:length(Y_train)
         for j=1:Iteration
           %% ǰ�򴫲�
           In=X_train(1,:) ;   
           H_1i=In*w1;
           H_1o=tanh(H_1i);
           
           H_2i=H_1o*w2;
           H_2o=tanh(H_2i);         
           out=H_2o*w3;
         
           %% ���򴫲�
         %  e= 0.5*(Y_train(i)-out).^2;                     %�������=����-���
           
          Error=(Y_train(i)-out);

           D3=H_2o.*Error;
           D2=(((1-H_2o.^2))'.*(w3.*Error))*H_1o;
           D1=(((1-H_1o.^2))'.*w2*(( (1-H_2o.^2))'.*w3.*Error))*In;
           
           w1=w1+Lr*D1';%-a.*Lr.*w1;                                 %�����Ȩֵ����
           w2=w2+Lr*D2';%-a.*Lr.*w2;   
           w3=w3+Lr*D3';%-a.*Lr.*w3;%������Ȩֵ����
        
%          if j==1
%       
%            Err_squ = [Err_squ Error*Error];
% 
%          end 
%      end
      
       end%����һ�ֵĵ���
       end
    end
    


