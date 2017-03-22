%Function for creating confusion matrix
function confusion_matrix_knn5 = confusion(Actual_value,store_5_final)
    confusion_matrix_knn5=zeros(3,3);
    for i=1:15000
        if(Actual_value(i,1) == store_5_final(i,1))
            confusion_matrix_knn5(Actual_value(i,1),store_5_final(i,1))=confusion_matrix_knn5(Actual_value(i,1),store_5_final(i,1))+1;
        elseif(Actual_value(i,1) ~= store_5_final(i,1))
            confusion_matrix_knn5(Actual_value(i,1),store_5_final(i,1))=confusion_matrix_knn5(Actual_value(i,1),store_5_final(i,1))+1;
        end
    end
end