Input=load('train_sp2017_v19'); %loading training data in Matrix 'Input'
Test=load('test_sp2017_v19');%loading test data in Matrix 'Test'
Test=(Test.');
H=(Input');
%Histogram Plot of Training data and test data to find out what distribution it is
subplot(2,1,1)
histogram(H);
title('Histogram of Input data')
subplot(2,1,2)
histogram(Test);
title('Histogram of Test data')

%Finding d (no. of features) and N(no. of feature vectors)
[d,N]=size(H);
class1=ones(5000,d);
class2=ones(5000,d);
class3=ones(5000,d);
%JB test for checking if feature vectors belong to normal distribution
counter=0;
for i=1:N
    chk=H(1:d,i);
    check=jbtest(chk);
    if(check==0)
        counter = counter+1;
    end
end

% Separating the classes
for i = 1: 5000
    class1(i,1:d)=Input(i,1:d);
    class2(i,1:d)=Input(i+5000,1:d);
    class3(i,1:d)=Input(i+10000,1:d);   
end
%finding mean and covariance of class1,class2,class3
% since the given data is transposed,mean is transposed
mu1=mean((class1));  
mu1=(mu1.');
mu2=mean((class2));  
mu2=(mu2.');
mu3=mean((class3));  
mu3=(mu3.');
covar1=cov(class1);
covar2=cov(class2);
covar3=cov(class3);
%Calculating inverse of covariance matrix
inverse_cov1=inv(covar1);
inverse_cov2=inv(covar2);
inverse_cov3=inv(covar3);
%Initialising "store" matrix
store=ones(N,1);
% Calculating gaussian discriminant function for each  training feature vector
for j=1: N
    g1x= (-0.5)*((H(1:d,j) - mu1).')* (inverse_cov1)* (( H(1:d,j) - mu1)) - (.5 * log(det(covar1))) ;
    g2x= (-0.5)*((H(1:d,j) - mu2).')* (inverse_cov2)* (( H(1:d,j) - mu2)) - (.5 * log(det(covar2))) ;
    g3x= (-0.5)*((H(1:d,j) - mu3).')* (inverse_cov3)* (( H(1:d,j) - mu3)) - (.5 * log(det(covar3))) ;
    if(g1x > g2x && g1x > g3x)
        store(j,1)=1;
    elseif(g2x > g1x && g2x > g3x)
          store(j,1)=2;
    else
         store(j,1)=3;        
    end        
end
%Calculating Error Matrix for visualization of the performance
err1_2=0;err1_3=0;tp1=0;
err2_1=0;err2_3=0;tp2=0;
err3_1=0;err3_2=0;tp3=0;
err_probability=0;
for i=1:N
    if (i <= 5000) && (store(i,1) == 2)
            err1_2 = err1_2+1;
    elseif(i <= 5000) && (store(i,1) == 3)
            err1_3 = err1_3+1;
    elseif(i <= 5000) && (store(i,1) == 1)
            tp1 =tp1+1;
    elseif ((i > 5000) && (i<=10000)) && (store(i,1) == 1) 
            err2_1 = err2_1 + 1;
    elseif ((i > 5000) && (i<=10000)) && (store(i,1) == 3) 
            err2_3 = err2_3 + 1;
    elseif ((i > 5000) && (i<=10000)) && (store(i,1) == 2) 
            tp2 = tp2 + 1;        
    elseif ((i > 10000) &&(i <= 15000)) && (store(i,1) == 1) 
            err3_1 = err3_1+1;
    elseif ((i > 10000) &&(i <= 15000)) && (store(i,1) == 2) 
            err3_2 = err3_2+1;
    elseif ((i > 10000) &&(i <= 15000)) && (store(i,1) == 3) 
            tp3 = tp3+1;         
    end
end
%Calculating Probability of Error for Training data
err_probability=((err1_2+err1_3+err2_1+err2_3+err3_1+err3_2)/N)*100;
%Error matrix
err_mat=[tp1 err1_2 err1_3;err2_1 tp2 err2_3;err3_1 err3_2 tp3];
%Initialising "result" matrix
result=ones(N,1);
% Calculating gaussian discriminant function for each test feature vector
for j=1: N
    g1_1x= (-0.5)*((Test(1:d,j) - mu1).')* (inverse_cov1)* (( Test(1:4,j) - mu1)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar1))) ;
    g2_1x= (-0.5)*((Test(1:d,j) - mu2).')* (inverse_cov2)* (( Test(1:4,j) - mu2)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar2))) ;
    g3_1x= (-0.5)*((Test(1:d,j) - mu3).')* (inverse_cov3)* (( Test(1:4,j) - mu3)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar3))) ;
    if(g1_1x > g2_1x && g1_1x > g3_1x)
        result(j,1)=1;
    elseif(g2_1x > g1_1x && g2_1x > g3_1x)
          result(j,1)=2;
    else
         result(j,1)=3;        
    end        
end
%Writing Classification result into text-file
fileID = fopen('akoluma-classified-takehome1.txt','w');
fprintf(fileID,'%d\r\n',result);
fclose(fileID);








