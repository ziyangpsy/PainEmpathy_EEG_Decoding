%--------------------------------------------------------------------------
% Copyright (C) 2024 @RacLab (Reward and Cognition Laboratory)
% Intellectual Property of Yang Ziyang yzypsy2001@gmail.com
% Decoding for EEG/ERP
%
% ===================
%     NOTE
% ===================
% This Scripts is recommended to be used with MATLAB version R2022b or later for full compatibility.
% Ensure that the following content is included in the plugin folder:
% 1.LibSVM  https://www.csie.ntu.edu.tw/~cjlin/libsvm/ (recommended version 3.33 or later)
% 2.eeglab2021.1  https://sccn.ucsd.edu/eeglab (recommended version 2021.1 or later)
% 3.Decodingcheck
% 4.shadedErrorBar
%
% ===================
%     INPUT
% ===================
% ERP        - Sample*(Feature*Timepoints) -> [data_all]
% Label      - SampleLabel -> [label]
% **EEG/ERP data Processed after S3_Epoching_MVPA and MVPA_Decoding_ExtractTimecourseFeatures_YZY
%
% ===================
%     SETTING
% ===================
% [kfold]          - K-Fold cross-validator
% [nsample]        - Sampling rate
% [plottimewindow] - Timewindow
% [C]              - Mechine learning Classifier
% [FS]             - Feature selection
% **See [Decoding Setting] for complete parameter settings
%
% ===================
%     OUTPUT
% ===================
% [all_acc_final]  - Classification accuracy at each time point
% [pvalue_acc]     - Pvalue with permutation test
% [...Sen/Spe/Auc] - Depends on what you need
%
%--------------------------------------------------------------------------
% Written by Yang Ziyang,08-31-2024

%%
clc;clear;close all
rootpath='C:\Users\yang\Desktop\RacLab3.0_YZY\';
rdpath=[rootpath,'MVPA\data\'];
datapath=[rdpath,'Data4MVPA\No2\'];
savepath=[rdpath,'ProData\'];
cd(datapath)
load 'cond_erp_all_F.mat'
load 'cond_label_F.mat'
data_all = double(erp_all);
label=label_all;

%% Decoding Setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic setting
kfold = 5 ; %外层折数 外层经验值一般大于4，且为样本数的合数   内层经验值：5
% kfold_ncv = 5 ; %内层折数
nsample = 100;
plottimewindow = [-0.200 +1.000];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
LuckyNumber = 13;  % Enter your Lucky Number %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Classifier Setting
C = 2
% 1 -
% 2 - SVM (recommended)
        %%Hyper-parameter optimization - Kernel function ONLY FOR SVM(FI = 2)
        KF = 2
        % 1 - default
        % 2 - linear kernal
        % 3 - gauss kernal (Warning:If your computer configuration is too low, it may take too long to complete the calculation)
% 3 - Later version update...

%%Feature selection Setting
FS = 2
% 1 - Filtering
        %%Filtering method ONLY FOR Filtering(FS = 1)
        FM = 2
        % 1 - Spearman (no recommended)
        % 2 - T (recommended)
        % 3 - F
        %%Filtering percent
        FP = 0.05  %[0-100] recommended 5-10%
% 2 - PCA (recommended)
        PCA_Contribution = 0.99  %[80-95] recommended 90%a
% 3 - RFE (no recommended) % RFE ONLY FOR SVM （default -c = 1）
        RFE_FP = 0.05 %[0-100] recommended 5-10%
        RFE_Step = 2 %[1-5] recommended 2 *Composite number of features
% 4 - Later version update...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
label(label(:, 1) == 2, 1) = -1;
subj=max(label(:,2));
nbchan = length(chanlocs);

for i = 1:subj
    % 找到label中第二列等于当前样本号i的所有行索引
    idx = find(label(:, 2) == i);
    data_cell{i} = data_all(idx, :);
end

for i = 1:subj
    % 找到label中第二列等于当前样本号i的所有行索引
    idx = find(label(:, 2) == i);
    label_cell{i} = label(idx, :);
end

twlength = plottimewindow(2) - plottimewindow(1);
nbchan = length(chanlocs);
pnts = nsample * twlength; % 采样率*采样时间
%空间预分配
% all_predicted_labels = zeros(length(label), pnts);
% all_decision_values = zeros(length(label), pnts);
% all_acc_kfold = zeros(kfold, pnts);
% all_weights = zeros(kfold, size(data_all,2), pnts); % 分类权重 线性核不适用 % KF = 2 - linear kernal
all_acc = zeros(subj, pnts);pvalue_acc = zeros(subj, pnts);
% all_sen = zeros(subj, pnts);all_pvalue_sen = zeros(subj, pnts);
% all_spe = zeros(subj, pnts);all_pvalue_spe = zeros(subj, pnts);
all_auc_values = zeros(subj, pnts);pvalue_AUC = zeros(subj, pnts);

%%
%将
M = waitbar(0,'please  wait..');
for y = 1:subj  %每个被试水平

    waitbar(y/subj,M,[num2str(y),'/',num2str(subj)])

    data_all=data_cell{y};
    label=label_cell{y};label = label(:, 1);

    timescheck=length(times);
    Decodingcheck(pnts,timescheck,'1')

    start_indices1 = 1:nbchan:(nbchan * pnts);
    matrices4decoding = cell(1, pnts);
    for t = 1:pnts
        start_idx = start_indices1(t);
        end_idx = start_idx + nbchan - 1;
        matrices4decoding{t} = data_all(:, start_idx:end_idx);
    end

    H = waitbar(0,'please wait..');
    for z=1:length(matrices4decoding) %最外层循环 i=timepoint

        data_all = matrices4decoding{z};

        waitbar(z/length(matrices4decoding),H,[num2str(z),'/',num2str(length(matrices4decoding))])

        n_sample = length(label);
        n_div = ceil(n_sample / kfold);
        indices = repmat(1:kfold, 1, n_div)';
        indices = indices(1:n_sample);
        rng(LuckyNumber); % Lucky number
        indices  = indices(randperm(n_sample));

        % 创建空数组，用于保存每一折的预测结果
        predicted_label = zeros(size(label));
        deci = zeros(size(label));
        %用于保存每一折的特征权重，k*特征数
        w = zeros(kfold, size(data_all,2)); % 分类权重

        acc_fold = zeros(kfold, 1);
        sen_fold = zeros(kfold, 1);
        spe_fold = zeros(kfold, 1);
        auc_fold = zeros(kfold, 1);

        %% PCA
        h = waitbar(0,'please wait..');
        %对于K折交叉验证中的每一折
        for i=1:kfold
            waitbar(i/kfold,h,[num2str(i),'/',num2str(kfold)])
            % 数据集划分
            train_idx = find(indices ~= i); test_idx = find(indices == i); %获取本折训练集、测试集的位置（索引）
            train_data = data_all(train_idx, :); test_data = data_all(test_idx, :); %提取 训练特征集、测试特征集
            train_label = label(train_idx, :); test_label = label(test_idx,:); %提取 训练集标签、 测试集标签

%             [train_data, mu, sigma] = zscore(train_data);
%             test_data = (test_data - mu) ./ sigma;
            [train_data,PS] = mapminmax(train_data',-1,1);
            test_data          = mapminmax('apply',test_data',PS);
            train_data = train_data';
            test_data   = test_data';
% %             % PCA特征降维
%             [pc,score,latent1] = pca(train_data); % PCA分析
%             latent2        = latent1./sum(latent1);
%             latent3        = cumsum(latent2);%特征值累计比例
%             %  laten_ind  = find(latent3 >= 0.90, 1, 'first');%选择累计贡献在90%以上的成分 %[80-95]  MAX主成分特征-1
%             laten_ind  = find(latent3 >= PCA_Contribution, 1, 'first');%选择累计贡献在90%以上的成分 %[80-95]  MAX主成分特征-1
%             train_data = train_data*pc(:,1:laten_ind);% 原始数据向k维空间的投影。laten_ind，累计贡献刚达到0.8的成分编号
%             test_data  = test_data*pc(:,1:laten_ind); %
% 
%             % 数据标准化（减均值、除标准差）
%             % 降维后每个主成分的量纲不一致，需要再做一次标准化
%             [train_data, mu, sigma] = zscore(train_data);
%             test_data = (test_data - mu) ./ sigma;

            %
%             [bestacc,bestc] = SVMcgForClass_NoDisplay_linear(train_label,train_data,-10,10,3,1);
%             cmd = ['-s 0 -t 0', ' -c ',num2str(bestc)];
            cmd = ['-s 0 -t 0 -c 1'];


            % 使用筛选出的特征以及最优超参数做训练和预测
            model = svmtrain(train_label,train_data, cmd);
            test_label = test_label(:, 1);
            [predicted_label(test_idx), ~, deci(test_idx)] = svmpredict(test_label,test_data,model);


%         acc_final = mean(label==predicted_label);% 准确率

        % 混淆矩阵
%         Decodingcheck(length(label),length(predicted_label),'2')
%         cmat=confusionmat(label,predicted_label);
%         sen_final=cmat(2,2)/sum(cmat(2,:));
%         spe_final=cmat(1,1)/sum(cmat(1,:));
% 
%         % ROC曲线
%         deci = deci(:, 1);
%         [X,Y,T,AUC_final] = perfcurve(label,deci,1);
%         figure;plot(X,Y,'b','linewidth',1);hold on;plot(X,X,'k--','linewidth',1);
%         xlabel('False positive rate'); ylabel('True positive rate');
%         close gcf
            Decodingcheck(length(label),length(predicted_label),'2')
            cmat=confusionmat(label(test_idx),predicted_label(test_idx));
%             sen_fold(i) = cmat(2,2) / sum(cmat(2,:)); % 敏感度
%             spe_fold(i) = cmat(1,1) / sum(cmat(1,:)); % 特异性
            acc_fold(i) = mean(label(test_idx) == predicted_label(test_idx)); % 基于测试集的准确率
            [X, Y, T, auc_fold(i)] = perfcurve(label(test_idx), deci(test_idx), 1);
        end

        acc_final = mean(acc_fold); % 平均准确率
        sen_final = mean(sen_fold); % 平均敏感度
        spe_final = mean(spe_fold); % 平均特异性
        AUC_final = mean(auc_fold); % 平均AUC

        disp(['Time Point ', num2str(z)]);
        disp(['accuracy= ', num2str(acc_final)]);
        disp(['Sensitivty= ', num2str(sen_final)]);
        disp(['Specificity= ', num2str(spe_final)]);
        disp(['AUC= ', num2str(AUC_final)]);

        %       all_predicted_labels(:, z) = predicted_label;
        %       all_decision_values(:, z) = deci;
        %       all_weights(:, :, z) = w;
        all_acc(y, z) = acc_final;        
        all_auc_values(y, z) = AUC_final;
%         all_sen(y, z) = sen_final;
%         all_spe(y, z) = spe_final;


        %     w_msk = double(sum(w~=0,1)==size(w,1)); % 一致性特征
        %     w = mean(w,1).*w_msk; % 每一次交叉验证都被选择到的特征权重

        close all ;

        disp('分类完成！');

        close(h)
    end
    close(H)
end
close (M)

%% filter averger
window_size = 5;
for y = 1:size(all_acc, 1)  % 遍历每个被试
    for z = window_size:size(all_acc, 2)  % 从第5个时间点开始滑动
        % 对每个时间点，计算该时间点前5个时间点的平均值
        all_acc(y, z) = mean(all_acc(y, z-window_size+1:z));
        all_auc_values(y, z) = mean(all_auc_values(y, z-window_size+1:z));
    end
end

%% --------------------------------------------------------------------------
%  Permutation test
mean_acc = mean(all_acc, 1);
mean_auc = mean(all_auc_values, 1);


permut     = 1000 ; 
%分类准确率检验置换次数，建议10000次，不少于5000次，最最最最少不少于1000次
acc_final_rand = zeros(permut, subj, pnts);%ii,yy,zz
% sen_final_rand = zeros(permut, subj, pnts);
% spe_final_rand = zeros(permut, subj, pnts);
AUC_final_rand = zeros(permut, subj, pnts);
acc_fold_r =  zeros(kfold, 1);
auc_fold_r = zeros(kfold, 1);
P = waitbar(0,'please wait..');disp(['Running permutation test']);

for ii=1:permut

    waitbar(ii/permut,P,['permutation:',num2str(ii),'/',num2str(permut)]);

    M = waitbar(0,'please  wait..');
    for yy = 1:subj

        waitbar(yy/subj,M,[num2str(yy),'/',num2str(subj)])

        data_all=data_cell{yy};
        label=label_cell{yy};label = label(:, 1);

        timescheck=length(times);
        Decodingcheck(pnts,timescheck,'1')

        start_indices1 = 1:nbchan:(nbchan * pnts);
        matrices4decoding = cell(1, pnts);
        for t = 1:pnts
            start_idx = start_indices1(t);
            end_idx = start_idx + nbchan - 1;
            matrices4decoding{t} = data_all(:, start_idx:end_idx);
        end

        HH = waitbar(0,'please wait..');
        for zz=1:length(matrices4decoding) %最外层循环 i=timepoint

            data_all = matrices4decoding{zz};

            waitbar(zz/length(matrices4decoding),HH,[num2str(zz),'/',num2str(length(matrices4decoding))])

            %随机打乱标签 code of permutation
            randlabel = randperm(length(label));
            label_r  = label(randlabel);

            %划分每一折的样本
            n_sample = length(label);
            n_div = ceil(n_sample / kfold);
            indices = repmat(1:kfold, 1, n_div)';
            indices = indices(1:n_sample);
            rng(ii);
            indices  = indices(randperm(n_sample));

            %创建空数组，用于保存置换后与所有样本的预测结果
            predicted_label_r = zeros(size(label));
            deci_r = zeros(size(label));

            %% PCA
            for j=1:kfold
                % 数据集划分
                train_idx = find(indices ~= j); test_idx = find(indices == j); %获取本折训练集、测试集的位置（索引）
                train_data = data_all(train_idx, :); test_data = data_all(test_idx, :); %提取 训练特征集、测试特征集
                train_label = label_r(train_idx, :); test_label = label_r(test_idx,:); %提取 训练集标签、 测试集标签
                
                
%                 [train_data, mu, sigma] = zscore(train_data);
%                 test_data = (test_data - mu) ./ sigma;
            [train_data,PS] = mapminmax(train_data',-1,1);
            test_data          = mapminmax('apply',test_data',PS);
            train_data = train_data';
            test_data   = test_data';
                % PCA特征选择
                %xdm = train_data - repmat(mean(train_data,1),size(train_data,1),1);% 去均值
%                 [pc,score,latent1] = pca(train_data); % PCA分析
%                 latent2        = latent1./sum(latent1);
%                 latent3        = cumsum(latent2);%特征值累计比例
%                 laten_ind  = find(latent3 >= PCA_Contribution, 1, 'first');%选择累计贡献在90%以上的成分
%                 train_data = train_data*pc(:,1:laten_ind);% 原始数据向k维空间的投影。
%                 test_data  = test_data*pc(:,1:laten_ind);

                % 数据标准化（减均值、除标准差）
                % 降维后每个主成分的量纲不一致，需要再做一次标准化
%                 [train_data, mu, sigma] = zscore(train_data);
%                 test_data = (test_data - mu) ./ sigma;


                %%% linear kernal
%             [bestacc,bestc] = SVMcgForClass_NoDisplay_linear(train_label,train_data,-10,10,3,1);
%             cmd = ['-s 0 -t 0', ' -c ',num2str(bestc)];
            cmd = ['-s 0 -t 0 -c 1'];


               if numel(unique(test_label)) < 2
                    disp(['Fold ', num2str(j), ' 测试集仅有一种类别，跳过AUC和ACC计算。']);
                    acc_fold_r(j) = NaN;
                    auc_fold_r(j) = NaN;
                else
                    % 训练SVM模型
                    model = svmtrain(train_label, train_data, cmd);
                    % 在测试集上预测
                    [predicted_label_r(test_idx), ~, deci_r(test_idx)] = svmpredict(test_label, test_data, model);
                    
                    % 计算fold内的准确率
                    acc_fold_r(j) = mean(test_label == predicted_label_r(test_idx));
                    
                    % 计算fold内的AUC
                    [~, ~, ~, auc_fold_r(j)] = perfcurve(test_label, deci_r(test_idx), 1);
                end

            
            end

            acc_final_rand(ii,yy,zz) = nanmean(acc_fold_r);% 准确率
            AUC_final_rand(ii,yy,zz) = nanmean(auc_fold_r);
            % 混淆矩阵
%             cmat_r=confusionmat(label_r(test_idx),predicted_label_r(test_idx));


%             sen_final_rand(ii,yy,zz)=cmat_r(2,2)/sum(cmat_r(2,:));
%             spe_final_rand(ii,yy,zz)=cmat_r(1,1)/sum(cmat_r(1,:));
            % ROC曲线
%             [~,~,~,AUC_final_rand(ii,yy,zz)] = perfcurve(label_r,deci_r,1);
%             clear predicted_label_r deci_r
        end
        close(HH)
    end
    close(M)
end
close(P)

for i = 1:size(acc_final_rand, 1)  % 遍历每个置换
    for y = 1:size(acc_final_rand, 2)  % 遍历每个被试
        for z = window_size:size(acc_final_rand, 3)  % 从第5个时间点开始滑动
            % 对每个置换和被试，计算时间点前5个时间点的平均值
            acc_final_rand(i, y, z) = mean(acc_final_rand(i, y, z-window_size+1:z));
            AUC_final_rand(i, y, z) = mean(AUC_final_rand(i, y, z-window_size+1:z));
        end
    end
end
% acc_final_rand   =  squeeze(mean(acc_final_rand, 2));
% sen_final_rand  =   squeeze(mean(sen_final_rand, 2));
% spe_final_rand   =  squeeze(mean(spe_final_rand, 2));
% AUC_final_rand   =  squeeze(mean(AUC_final_rand, 2));

% for zzz=1:length(matrices4decoding)
%     % 计算 p 值
%     pvalue_acc = mean([acc_final_rand; mean_acc] >= mean_acc,1);
%     %             pvalue_sen = mean([sen_final_rand; sen_final] >= sen_final);
%     %             pvalue_spe = mean([spe_final_rand; spe_final] >= spe_final);
%     pvalue_AUC = mean([AUC_final_rand; mean_auc] >= mean_auc,1);
% 
% end

t_observed = zeros(1, pnts);
for t = 1:pnts
    [~, ~, ~, stats] = ttest(all_acc(:, t) - 0.5, 0, 'Tail', 'right');
    t_observed(t) = stats.tstat;
end

% 计算置换t值 - 准确率
t_permuted = zeros(permut, pnts);
for ii = 1:permut
    for t = 1:pnts
        acc_perm = squeeze(acc_final_rand(ii, :, t));
        [~, ~, ~, stats_perm] = ttest(acc_perm - 0.5, 0, 'Tail', 'right');
        t_permuted(ii, t) = stats_perm.tstat;
    end
end

% 计算t值 - AUC
t_observed_auc = zeros(1, pnts);
for t = 1:pnts
    [~, ~, ~, stats] = ttest(all_auc_values(:, t) - 0.5, 0, 'Tail', 'right');
    t_observed_auc(t) = stats.tstat;
end

% 计算置换t值 - AUC
t_permuted_auc = zeros(permut, pnts);
for ii = 1:permut
    for t = 1:pnts
        auc_perm = squeeze(AUC_final_rand(ii, :, t));
        [~, ~, ~, stats_perm] = ttest(auc_perm - 0.5, 0, 'Tail', 'right');
        t_permuted_auc(ii, t) = stats_perm.tstat;
    end
end

% 定义t值阈值
df = subj - 1;
t_threshold = tinv(0.95, df); % 单尾检验，p = 0.05

significant_timepoints = t_observed > t_threshold;
clusters = bwconncomp(significant_timepoints);
cluster_stats_observed = zeros(1, length(clusters.PixelIdxList));
for i = 1:length(clusters.PixelIdxList)
    cluster_indices = clusters.PixelIdxList{i};
    cluster_stats_observed(i) = sum(t_observed(cluster_indices));
end

% 构建置换分布
max_cluster_stats_permuted = zeros(permut, 1);
for ii = 1:permut
    t_perm = t_permuted(ii, :);
    significant_timepoints_perm = t_perm > t_threshold;
    clusters_perm = bwconncomp(significant_timepoints_perm);
    cluster_stats_perm = zeros(1, length(clusters_perm.PixelIdxList));
    for j = 1:length(clusters_perm.PixelIdxList)
        cluster_indices_perm = clusters_perm.PixelIdxList{j};
        cluster_stats_perm(j) = sum(t_perm(cluster_indices_perm));
    end
    if ~isempty(cluster_stats_perm)
        max_cluster_stats_permuted(ii) = max(cluster_stats_perm);
    else
        max_cluster_stats_permuted(ii) = 0;
    end
end

% 计算簇级别的p值 
cluster_pvals = zeros(1, length(cluster_stats_observed));
for i = 1:length(cluster_stats_observed)
    cluster_pvals(i) = mean(max_cluster_stats_permuted >= cluster_stats_observed(i));
end

% 标记显著的簇 - 准确率
alpha = 0.05;
significant_clusters = find(cluster_pvals < alpha);

% 观测数据中超过阈值的时间点 - AUC
significant_timepoints_auc = t_observed_auc > t_threshold;

% 识别观测数据中的簇 - AUC
clusters_auc = bwconncomp(significant_timepoints_auc);

% 计算观测数据的簇统计量 - AUC
cluster_stats_observed_auc = zeros(1, length(clusters_auc.PixelIdxList));
for i = 1:length(clusters_auc.PixelIdxList)
    cluster_indices = clusters_auc.PixelIdxList{i};
    cluster_stats_observed_auc(i) = sum(t_observed_auc(cluster_indices));
end

% 构建置换分布 - AUC
max_cluster_stats_permuted_auc = zeros(permut, 1);
for ii = 1:permut
    t_perm_auc = t_permuted_auc(ii, :);
    significant_timepoints_perm_auc = t_perm_auc > t_threshold;
    clusters_perm_auc = bwconncomp(significant_timepoints_perm_auc);
    cluster_stats_perm_auc = zeros(1, length(clusters_perm_auc.PixelIdxList));
    for j = 1:length(clusters_perm_auc.PixelIdxList)
        cluster_indices_perm_auc = clusters_perm_auc.PixelIdxList{j};
        cluster_stats_perm_auc(j) = sum(t_perm_auc(cluster_indices_perm_auc));
    end
    if ~isempty(cluster_stats_perm_auc)
        max_cluster_stats_permuted_auc(ii) = max(cluster_stats_perm_auc);
    else
        max_cluster_stats_permuted_auc(ii) = 0;
    end
end

% 计算簇级别的p值 - AUC
cluster_pvals_auc = zeros(1, length(cluster_stats_observed_auc));
for i = 1:length(cluster_stats_observed_auc)
    cluster_pvals_auc(i) = mean(max_cluster_stats_permuted_auc >= ...
        cluster_stats_observed_auc(i));
end

% 标记显著的簇 - AUC
significant_clusters_auc = find(cluster_pvals_auc < alpha);

% 绘图 - 准确率
mean_acc = mean(all_acc, 1);
std_err_acc = std(all_acc, 0, 1) / sqrt(subj);
timelist = (-200:1000/nsample:1500) ./ 1000;
timelist(1) = [];
itime = timelist >= plottimewindow(1) & timelist <= plottimewindow(2);
time_points = timelist(itime);

smooth_mean_acc = smooth(mean_acc(itime), 0.15, 'loess');
smooth_std_err_acc = smooth(std_err_acc(itime), 0.15, 'loess');

figure();
hold on
shadedErrorBar(time_points, smooth_mean_acc, smooth_std_err_acc, ...
    'lineprops', {'-', 'LineWidth', 1.5, 'Color', [0/255, 114/255, 178/255]});

ylims = [0.44, 0.65]; % 根据实际数据调整y轴范围
for i = significant_clusters
    cluster_indices = clusters.PixelIdxList{i};
    cluster_times = time_points(cluster_indices);
    if length(cluster_times)  < 3
        continue; % 跳过空簇
    end
    if length(cluster_times) >= 3
        x_start = min(cluster_times);
        x_end = max(cluster_times);
    end
    y_value = ylims(1) + 0.01; % 线的位置，可根据需要调整
    plot([x_start, x_end], [y_value, y_value], '-', 'LineWidth', 2, 'Color', [0/255, 114/255, 178/255]);
end


% 其他绘图设置
xlims = [plottimewindow(1), plottimewindow(2)];
ylims = [0.44, 0.65];
plot(xlims, [0.5, 0.5], 'k-', 'linewidth', 1.5, 'color', [0.25, 0.25, 0.25]);
plot([0, 0], ylims, 'k--', 'linewidth', 1.5, 'color', [0.25, 0.25, 0.25]);
xlim(xlims); ylim(ylims);
set(gca, 'xtick', plottimewindow(1):0.1:plottimewindow(2), ...
    'ytick', 0:0.1:1, 'box', 'off', 'FontSize', 17, 'FontName', 'Arial', ...
    'FontWeight', 'Bold', 'LineWidth', 1, 'TickDir', 'out');
set(gcf, 'units', 'centimeters', 'position', [0, 0, 25, 20], 'color', 'w');
xlabel('Time (s)'); ylabel('Classification Accuracy');
title('Classification Accuracy Over Time with Cluster-based Permutation');
set(findobj(gcf, 'type', 'axes'), 'FontName', 'Arial', 'FontSize', 14, ...
    'FontWeight', 'Bold', 'LineWidth', 1, 'TickDir', 'out');

% 绘图 - AUC
mean_auc = mean(all_auc_values, 1);
std_err_auc = std(all_auc_values, 0, 1) / sqrt(subj);
smooth_mean_auc = smooth(mean_auc(itime), 0.15, 'loess');
smooth_std_err_auc = smooth(std_err_auc(itime),0.15, 'loess');

figure();
hold on
shadedErrorBar(time_points, smooth_mean_auc, smooth_std_err_auc, ...
    'lineprops', {'-', 'LineWidth', 1.5, 'Color', [0/255, 114/255, 178/255]});

% 标记显著的簇 - AUC
ylims = [0.45, 0.70]; % 根据实际数据调整y轴范围
for i = significant_clusters_auc
    cluster_indices = clusters_auc.PixelIdxList{i};
    cluster_times = time_points(cluster_indices);
    if length(cluster_times)  < 3
        continue; % 跳过空簇
    end
    if length(cluster_times) >= 3
        x_start = min(cluster_times);
        x_end = max(cluster_times);
    end
    y_value = ylims(1) + 0.01; % 线的位置，可根据需要调整
    plot([x_start, x_end], [y_value, y_value], '-', 'LineWidth', 2, 'Color', [0/255, 114/255, 178/255]);
end

% 其他绘图设置
xlims = [plottimewindow(1), plottimewindow(2)];
ylims = [0.44, 0.70];
plot(xlims, [0.5, 0.5], 'k-', 'linewidth', 1.5, 'color', [0.25, 0.25, 0.25]);
plot([0, 0], ylims, 'k--', 'linewidth', 1.5, 'color', [0.25, 0.25, 0.25]);
xlim(xlims); ylim(ylims);
yticks([0.45 0.5 0.55 0.60 0.65 0.70]);
set(gca, 'xtick', plottimewindow(1):0.1:plottimewindow(2), ...
    'ytick', 0:0.05:1, 'box', 'off', 'FontSize', 17, 'FontName', 'Arial', ...
    'FontWeight', 'Bold', 'LineWidth', 1, 'TickDir', 'out');
set(gcf, 'units', 'centimeters', 'position', [0, 0, 25, 20], 'color', 'w');
xlabel('Time (s)'); ylabel('AUC');
title('AUC Over Time with Cluster-based Permutation');
set(findobj(gcf, 'type', 'axes'), 'FontName', 'Arial', 'FontSize', 14, ...
    'FontWeight', 'Bold', 'LineWidth', 1, 'TickDir', 'out');

save(sprintf( 'LOW.mat' , savepath),'smooth_std_err_auc', 'smooth_mean_auc','time_points','significant_clusters','significant_clusters_auc','clusters','clusters_auc','pvalue_AUC','pvalue_acc','all_auc_values','subj','all_acc');