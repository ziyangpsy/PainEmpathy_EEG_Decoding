% Checking
% Written by Yang Ziyang, 2024-08-30
% 对decoding过程中重要的部分进行检查

%% 
function Decodingcheck(var1, var2, varName1)

    if var1 ~= var2
        if strcmp(varName1, '1') 
            error('采样率设置错误！');
        elseif strcmp(varName1, '2') %&& strcmp(varName2, 'predicted_label')
            error('label 和 predicted_label 的长度必须相同!');
        end
    else
        disp('ok!')
    end
end
