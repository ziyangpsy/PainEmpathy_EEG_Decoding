clear all;
%%% pca�޷�����Ȩ��wֵ

load gray_matter_aal1024.mat

% ��һ��������֤
permut     = 200;%����׼ȷ�ʼ����û�����
h = waitbar(0,'please wait..');
for i = 1:size(data_all,1)
    waitbar(i/size(data_all,1),h,[num2str(i),'/',num2str(size(data_all,1))])
    new_DATA = data_all;
    new_label  = label;
    test_data   = data_all(i,:); new_DATA(i,:) = []; train_data = new_DATA;
    test_label   = label(i,:);new_label(i,:) = [];train_label = new_label;
    
    % ���ݹ�һ��
    [train_data,PS] = mapminmax(train_data',0,1);
    test_data          = mapminmax('apply',test_data',PS);
    train_data = train_data';
    test_data   = test_data';
    
    % PCA����ѡ��
    xdm = train_data - repmat(mean(train_data,1),size(train_data,1),1);% ȥ��ֵ
    [pc,score,latent,tsquare] = princomp(xdm); % PCA����
    latent        = latent./sum(latent);
    latent        = cumsum(latent);%����ֵ�ۼƱ���
    laten_ind  = find(latent >= 0.9);%ѡ���ۼƹ�����90%���ϵĳɷ�
    train_data = train_data*pc(:,1:laten_ind(1));% ԭʼ������kά�ռ��ͶӰ��laten_ind���ۼƹ��׸մﵽ0.9�ĳɷֱ��
    test_data  = test_data*pc(:,1:laten_ind(1));
    
    %%% linear kernal
    [bestacc,bestc] = SVMcgForClass_NoDisplay_linear(train_label,train_data,-10,10,5,0.2);
    cmd = ['-t 0 ', ' -c ',num2str(bestc)];
    
    %%% gauss kernal
    %     [bestacc,bestc,bestg] = SVMcgForClass_NoDisplay_gauss(train_label,train_data,-10,10,-10,10,5,0.2,0.2);
    %     cmd = ['-t 2 ', ' -c ',num2str(bestc),' -g ', num2str(bestg)];
    
    model = svmtrain(train_label,train_data, cmd);
    [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
    acc(i,1) = accuracy(1);
    deci_value(i,1) = deci;
    clear  test_data  train_data test_label train_label model
end
acc_final = mean(acc);
disp(['accuracy - ',num2str(acc_final)]);

% ROC����
[X,Y,T,AUC] = perfcurve(label,deci_value,1);
figure;plot(X,Y);hold on;plot(X,X,'-');
xlabel('False positive rate'); ylabel('True positive rate');

for i=1:length(X)
    Cut_off(i,1) = (1-X(i))*Y(i);
end
[~,maxind] = max(Cut_off);
Specificity = 1-X(maxind);
Sensitivty = Y(maxind);
disp(['Specificity= ', num2str(Specificity)]);
disp(['Sensitivty= ', num2str(Sensitivty)]);


% permutation test
acc_final_rand     = zeros(permut,1);
h = waitbar(0,'please wait..');
for i=1:permut
    waitbar(i/permut,h,['permutation:',num2str(i),'/',num2str(permut)]);
    randlabel = randperm(length(label));
    label_r  = label(randlabel);
    for j=1:size(data_all,1)
        new_DATA = data_all;
        new_label  = label_r;
        test_data   = new_DATA(j,:);  new_DATA(j,:) = []; train_data = new_DATA;
        test_label  = new_label(j,:);    new_label(j,:)  = [];  train_label = new_label;
        
        
        % ���ݹ�һ��
        [train_data,PS] = mapminmax(train_data',0,1);
        test_data          = mapminmax('apply',test_data',PS);
        train_data = train_data';
        test_data   = test_data';
        
        % PCA����ѡ��
        xdm = train_data - repmat(mean(train_data,1),size(train_data,1),1);% ȥ��ֵ
        [pc,score,latent,tsquare] = princomp(xdm); % PCA����
        latent        = latent./sum(latent);
        latent        = cumsum(latent);%����ֵ�ۼƱ���
        laten_ind  = find(latent >= 0.9);%ѡ���ۼƹ�����90%���ϵĳɷ�
        train_data = train_data*pc(:,1:laten_ind(1));% ԭʼ������kά�ռ��ͶӰ��
        test_data  = test_data*pc(:,1:laten_ind(1));
        
        %%% linear kernal
        [bestacc,bestc] = SVMcgForClass_NoDisplay_linear(train_label,train_data,-10,10,5,0.2);
        cmd = ['-t 0 ', ' -c ',num2str(bestc)];
        
        %%% gauss kernal
        %     [bestacc,bestc,bestg] = SVMcgForClass_NoDisplay_gauss(train_label,train_data,-10,10,-10,10,5,0.2,0.2);
        %     cmd = ['-t 2 ', ' -c ',num2str(bestc),' -g ', num2str(bestg)];
        
        model = svmtrain(train_label,train_data, cmd);
        [predicted_label, accuracy, deci] = svmpredict(test_label,test_data,model);
        acc_r(i) = accuracy(1);
        clear B IX order strength  cmd bestacc bestc F
    end
    acc_final_rand(i) = mean(acc_r);
    clear randlabel  acc_r label_r
end
close(h);
pvalue_final     = mean(abs(acc_final_rand) >= abs(acc_final));

save('svm_PCA_results.mat','acc_final','pvalue_final','AUC','Specificity','Sensitivty','p_auc','auc_rand');
