%%
1_S3_Epoch_MVPA
第一步与单变量分析的步骤基本一致：
  1.EPOCH 
  2.INTERPOLATE
  3.Baseline Correction
不一致的地方:
  1.将 EEG.data(:,:,Out) = nan; 替换为 EEG = pop_rejepoch( EEG, find(bad_epochs(si).total) ,0);
  2.将数据保存为set格式 EEG = pop_saveset(EEG,'filename',num2str(subj(si)),'filepath',exportpath)

%%
2_MVPA_Decoding_ExtractTimecoursefeatures
第二部将ERP数据做特征提取以及标签保存
主要功能:
  1.降采样(fs_new)
  2.supertrial(根据需要)
  3.整理数据为
     erp_all 即 [(super)trial x (timepoint x channel)] 每一行为一个trial，一行内按照时间点的顺序排列所有电极点
     label_all 第一列为类别(暂时只能做二分类)，第二列为样本编号(检查数据用的，并不会在decoding中使用到)
     time

3_MVPA_Decoding_SVM_Subj_L_ClusterBased_fliter
  1.需要先在Decoding Setting设定必要参数，见注解 55-94行
  2.整理数据及空间预分配，以被试为单位一cell,检查label_cell和data_cell,两者的行数必须是一样的
    这里有一些指标暂时用不掉注释掉了(spn/spe) 95-123行
  3.解码 127行开始
    第一个循环y--被试水平  matrices4decoding是每个时间点的数据
    第二个循环z--时间点水平 
    第三个循环i--kfold水平
    在一次解码中：
      (1)首先根据kfold做随机划分 153-158
      (2)划分后根据indices对数据进一步分为train和test 177-179
      (3)归一化/标准化/PCA降维 因为我们做的是全通道，也就是某一时间点中有正有负有大有小，为了避免极值影响要做归一化或者标准化
         181-199 根据需要选择
         *特别注意：归一化/标准化一定是在train上得到map后再在test上投射，防止数据泄露
      (4)解码 svmtrain得到模型，在testlabel上验证，得到每一fold参数，最后将每一fold上做平均 解码：204-210 平均：227-238
         在all_acc 和 all_auc_values上记录:被试x时间点矩阵
      (5)滑动平均filter averger 这一步参考 Bae & Luck (2018) 268-276

正式的解码到这里就结束了，可以直接画出带cluster的图，但因为没有经过统计检验所以不一定有意义
   
   4.Permutation test (蒙特卡洛极限，同样也是参考Bae和luck的文献，此检验相对严格)
    置换检验的过程与正式解码过程完全一致(重要，必须必须必须完全一致)

    但是多了一段：
    randlabel = randperm(length(label));
    label_r  = label(randlabel);
    即shuffle
    
    permutation的结果存在：
     acc_final_rand(ii,yy,zz)
     AUC_final_rand(ii,yy,zz)
    其中ii为permut次数
    278-421 置换检验  
    422后就是cluster的计算和画图，在EffortEmpathy的方法中做了详细的计算介绍

    主要步骤:
    (1)每一时间点比较得到t
    (2)连续显著的t形成cluster并相加得到cluster_t
    (3)对shuffle后数据重复，取max_cluster_t (无cluster则取0),形成分布
    (4)将cluster_t放入max_cluster_t排序，前5%则显著



Written by Yang Ziyang,14-03-2025













