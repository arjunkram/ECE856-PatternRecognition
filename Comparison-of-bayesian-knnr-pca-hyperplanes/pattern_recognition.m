Input=load('train_sp2017_v19'); %loading training data in Matrix 'Input'
Test=load('test_sp2017_v19');%loading test data in Matrix 'Test'
%Part1
Classified=load('akoluma-classified-takehome1.txt');    %loading our classified text file
sze_classified=size(Classified,1);%Finding no.of rows in text file
true_pattern=[3 1 2 3 2 1]; %The actual pattern
sze_pattern=size(true_pattern,2);%finding size of vector
val=sze_classified/sze_pattern;
Actual_value=repmat(true_pattern,1,val);   %repeating the pattern 15000/6 times and taking transpose to get column vector
Actual_value=Actual_value';
%Creating confusion Matrix for test data
confusion_matrix=confusion(Actual_value,Classified);
%Finding sum of diagonal elements
correct_classified=trace(confusion_matrix);
%finding probability of error
prob_error_1=(15000-correct_classified)/15000;
%--------------------------------------------------------------------------
%-------------------------------K-NN Algorithm-----------------------------
%--------------------------------------------------------------------------
ind_closest=ones(15000,5);
 for i=1:15000
     testpoint=Test(i,:);
        dists = sqrt(sum(bsxfun(@minus, Input, testpoint).^2, 2));
%          distance=sum((repmat(Test(i,:),15000,1) - H).^2,2);
         [Distance ,Index]=sort(dists,1);
         ind_closest(i, :) = Index(1 : 5).';
 end
 % 1-NNR
 store_1=zeros(15000,1);
 for i=1:15000
    if(ind_closest(i,1) <=5000)
        store_1(i,1)=1;
    elseif(ind_closest(i,1)>5000 && ind_closest(i,1)<=10000)
        store_1(i,1)=2;
    else
        store_1(i,1)=3;
    end 
 end
 %store the data
 fileID = fopen('akoluma-classified-1NNR-takehome2.txt','w');
fprintf(fileID,'%d\r\n',store_1);
fclose(fileID);
 % 3-NNR
  store_3=zeros(15000,3);
  store_3_final=zeros(15000,1);
 for i=1:15000
     for j=1:3
         if(ind_closest(i,j) <=5000)
            store_3(i,j)=1;
         elseif(ind_closest(i,j)>5000 && ind_closest(i,j)<=10000)
            store_3(i,j)=2;
         else
            store_3(i,j)=3;
         end
     end
      store_3_final(i,1)=mode(store_3(i,:));
 end
 %store the data
 fileID1 = fopen('akoluma-classified-3NNR-takehome2.txt','w');
fprintf(fileID1,'%d\r\n',store_3_final);
fclose(fileID1);
 % 5-NNR
  store_5=zeros(15000,3);
  store_5_final=zeros(15000,1);
 for i=1:15000
     for j=1:5
         if(ind_closest(i,j) <=5000)
            store_5(i,j)=1;
         elseif(ind_closest(i,j)>5000 && ind_closest(i,j)<=10000)
            store_5(i,j)=2;
         else
            store_5(i,j)=3;
         end
     end
      store_5_final(i,1)=mode(store_5(i,:));
 end   
  %store the data
 fileID2 = fopen('akoluma-classified-5NNR-takehome2.txt','w');
fprintf(fileID2,'%d\r\n',store_5_final);
fclose(fileID2);
 %Creating confusion Matrix for test data for 5-NNR
 confusion_matrix_knn5=confusion(Actual_value,store_5_final);
%Finding sum of diagonal elements
correct_classified_knn5=trace(confusion_matrix_knn5);
%finding probability of error
prob_error_knn5=(15000-correct_classified_knn5)/15000; 
 %Creating confusion Matrix for test data for 3-NNR
  confusion_matrix_knn3=confusion(Actual_value,store_3_final);
%Finding sum of diagonal elements
correct_classified_knn3=trace(confusion_matrix_knn3);
%finding probability of error
prob_error_knn3=(15000-correct_classified_knn3)/15000;
%Creating confusion Matrix for test data for 1-NNR
%  confusion_matrix_knn1=zeros(3,3);
 confusion_matrix_knn1=confusion(Actual_value,store_1);
correct_classified_knn1=trace(confusion_matrix_knn1);
%finding probability of error
prob_error_knn1=(15000-correct_classified_knn1)/15000;
%--------------------------------------------------------------------------
%---------------------------------PCA--------------------------------------
%--------------------------------------------------------------------------

%Finding d (no. of features) and N(no. of feature vectors)
[d,N]=size(Input');
k=2; % the no.of dimensions we need to reduce to
mean_training=mean(Input);
%Mean normalized Input
mean_norm_input=Input-mean_training;
%Covariance of mean normalized input
cov_norm_input=cov(mean_norm_input);
% finding eigenvectors(u1) and eigen values(s1) of covariance matrices
[u1,s1,v1]=svd(cov_norm_input);
%taking two eigenvectors corresponding to eigenvalues with largest value
u1_red=u1(:,1:k);
%Alternate method to find eignevalue and eigen vector
%[eigve,eigval]=eig(cov_norm_input);
% %Reduced training set and test set which is given by formula z=(u_red)' * x
reduced_train=(u1_red')*Input' ;
reduced_test=(u1_red')*Test';
[d1,N1]=size(reduced_train);
%-------------------Classification-----------------------------------------
%Splitting Input data into three parts
class1=ones(2,5000);class2=ones(2,5000);class3=ones(2,5000);
for i = 1: 5000
    class1(1:d1,i)=reduced_train(1:d1,i);
    class2(1:d1,i)=reduced_train(1:d1,i+5000);
    class3(1:d1,i)=reduced_train(1:d1,i+10000);   
end
%finding mean and covariance of class1,class2,class3
% since the given data is transposed,mean is transposed
mu1=mean(class1,2);  
% mu1=(mu1.');
mu2=mean(class2,2);  
% mu2=(mu2.');
mu3=mean(class3,2);  
% mu3=(mu3.');
covar1=cov(class1');
covar2=cov(class2');
covar3=cov(class3');
%Calculating inverse of covariance matrix
inverse_cov1=inv(covar1);
inverse_cov2=inv(covar2);
inverse_cov3=inv(covar3);
%Initialising "store" matrix
store_1=ones(N1,1);
% Calculating gaussian discriminant function for each  training feature vector
for j=1: N1
    g1x= (-0.5)*((reduced_train(1:d1,j) - mu1).')* (inverse_cov1)* (( reduced_train(1:d1,j) - mu1)) - (.5 * log(det(covar1))) ;
    g2x= (-0.5)*((reduced_train(1:d1,j) - mu2).')* (inverse_cov2)* (( reduced_train(1:d1,j) - mu2)) - (.5 * log(det(covar2))) ;
    g3x= (-0.5)*((reduced_train(1:d1,j) - mu3).')* (inverse_cov3)* (( reduced_train(1:d1,j) - mu3)) - (.5 * log(det(covar3))) ;
    if(g1x > g2x && g1x > g3x)
        store_1(j,1)=1;
    elseif(g2x > g3x)
          store_1(j,1)=2;
    else
         store_1(j,1)=3;        
    end        
end
%Re-creating the correct sequence for input data
h1=[1 2 3];
Nrepeat = 5000;  % The number of times each element will repeat 
train_pattern = reshape(repmat(h1,Nrepeat,1),1,Nrepeat*length(h1));
train_pattern=train_pattern';
%Creating confusion Matrix for training data
 confusion_matrix_3=confusion(train_pattern,store_1);
%Finding sum of diagonal elements
correct_classified_3=trace(confusion_matrix_3);
%finding probability of error
prob_error_3=(15000-correct_classified_3)/15000;
%---------------Classification of test data--------------------------------
%Initialising "result" matrix
result=ones(N1,1);
% Calculating gaussian discriminant function for each test feature vector
for j=1: N
    g1_1x= (-0.5)*((reduced_test(1:d1,j) - mu1).')* (inverse_cov1)* ((reduced_test(1:d1,j) - mu1)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar1))) ;
    g2_1x= (-0.5)*((reduced_test(1:d1,j) - mu2).')* (inverse_cov2)* (( reduced_test(1:d1,j) - mu2)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar2))) ;
    g3_1x= (-0.5)*((reduced_test(1:d1,j) - mu3).')* (inverse_cov3)* (( reduced_test(1:d1,j) - mu3)) - ((d/2) * log(2*pi)) - (.5 * log(det(covar3))) ;
    if(g1_1x > g2_1x && g1_1x > g3_1x)
        result(j,1)=1;
    elseif(g2_1x > g3_1x)
          result(j,1)=2;
    else
         result(j,1)=3;        
    end        
end
%store the data
 fileID4 = fopen('akoluma-classified-PCA-takehome2.txt','w');
fprintf(fileID4,'%d\r\n',result);
fclose(fileID4);
%Creating confusion Matrix for test data
confusion_matrix_3_test=confusion(Actual_value,result);
%Finding sum of diagonal elements
correct_classified_3_test=trace(confusion_matrix_3_test);
%finding probability of error
prob_error_3_test=(15000-correct_classified_3_test)/15000;
rt=reduced_test';
%--------------------------------------------------------------------------
%--------------------------Ho-Kayshap procedure----------------------------
%--------------------------------------------------------------------------
%initialising the three classes
class1_hk=ones(5000,4);
class2_hk=ones(5000,4);
class3_hk=ones(5000,4);
% Separating the classes
for i = 1: 5000
    class1_hk(i,1:4)=Input(i,1:4);
    class2_hk(i,1:4)=Input(i+5000,1:4);
    class3_hk(i,1:4)=Input(i+10000,1:4);   
end
%Between the training data of class1 and class2,class 3
Ytrain1 = [ones(5000,1) class1_hk ; -ones(5000,1) -class2_hk; -ones(5000,1) -class3_hk];
b_k1 = zeros(15000,1);
b_k = ones(15000,1);
a12=pinv(Ytrain1)*b_k1;
eta = 0.9;
flag=true;
while(flag)
    e = Ytrain1*a12-b_k;
    b_k1 = b_k + ((eta)*(e+abs(e)));
   a12=pinv(Ytrain1)*b_k1;
    error_train1=sum(b_k1-b_k);
    b_k=b_k1;
    
   if(error_train1 == 0)
       flag=false;
   end
end
new_test=[ones(15000,1) Test];
new_train=[ones(15000,1) Input];
new_class12 = new_test*a12;
new_class12_train = new_train*a12;
%Between the training data of class2 and class3,class 1
Ytrain23 = [ones(5000,1) class2_hk ; -ones(5000,1) -class3_hk; -ones(5000,1) -class1_hk];
b_k1_23 = zeros(15000,1);
b_k_23 = ones(15000,1);
 a23=pinv(Ytrain23)*b_k1_23;
flag=true;
while(flag)
    e = Ytrain23*a23-b_k_23;
    b_k1_23 = b_k_23 + ((eta)*(e+abs(e)));
    a23=pinv(Ytrain23)*b_k1_23;
    error_train_23=sum(b_k1_23-b_k_23);
    b_k_23=b_k1_23;
   if(error_train_23 == 0)
       flag=false;
   end
end
new_class23 = new_test*a23;
new_class23_train = new_train*a23;
%Between the training data of class3 and class1,class 2
Ytrain31 = [ones(5000,1) class3_hk ; -ones(5000,1) -class1_hk;-ones(5000,1) -class2_hk];
b_k1_31 = zeros(15000,1);
b_k_31 = ones(15000,1);
a31=pinv(Ytrain31)*b_k1_31;
flag=true;
while(flag)
    e = Ytrain31*a31-b_k_31;
    b_k1_31 = b_k_31 + ((eta)*(e+abs(e)));
    a31=pinv(Ytrain31)*b_k1_31;
    error_train_31=sum(b_k1_31-b_k_31);
    b_k_31=b_k1_31;
   if(error_train_31 == 0)
       flag=false;
   end
end
new_class31= new_test*a31;
new_class31_train = new_train*a31;
%--------------------------------------------------------------------------
%Classifying training data
hk_train=zeros(15000,1);
for i=1:15000
    if (new_class12_train(i,1)>new_class23_train(i,1) && new_class12_train(i,1)>new_class31_train(i,1) )
        hk_train(i,1)=1;
    elseif (new_class23_train(i,1)>new_class31_train(i,1) )
        hk_train(i,1)=2;
    else
        hk_train(i,1)=3;
    end
end
fileID6 = fopen('akoluma-classified-ho-kayshap-train-takehome2.txt','w');
fprintf(fileID6,'%d\r\n',hk_train);
fclose(fileID6);
%Creating confusion Matrix for train data
confusion_matrix_hk_train=confusion(train_pattern,hk_train);
%Finding sum of diagonal elements
correct_classified_hktrain=trace(confusion_matrix_hk_train);
%finding probability of error
prob_error_hk_train=(15000-correct_classified_hktrain)/15000;
%--------------------------------------------------------------------------
%Classifying test data
hk_test=zeros(15000,1);
for i=1:15000
    if (new_class12(i,1)>new_class23(i,1) && new_class12(i,1)>new_class31(i,1) )
        hk_test(i,1)=1;
    elseif (new_class23(i,1)>new_class31(i,1) )
        hk_test(i,1)=2;
    else
        hk_test(i,1)=3;
    end
end
%store the data
 fileID5 = fopen('akoluma-classified-ho-kayshap-takehome2.txt','w');
fprintf(fileID5,'%d\r\n',hk_test);
fclose(fileID5);
%Creating confusion Matrix for test data
confusion_matrix_hk_test=confusion(Actual_value,hk_test);
%Finding sum of diagonal elements
correct_classified_hktest=trace(confusion_matrix_hk_test);
%finding probability of error
prob_error_hk_test=(15000-correct_classified_hktest)/15000;
