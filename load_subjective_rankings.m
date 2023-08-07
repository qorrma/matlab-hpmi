
if strcmp(exp_id, '19_INS_7_BGLs')
    load('./SRs/SR_19_INS_7_BGLs.mat');
    bgls = [48, 64, 128, 168, 192, 218, 235];
    
elseif strcmp(exp_id, '33_INS_7_BGLs')
    load('./SRs/SR_33_INS_7_BGLs.mat');
    bgls = [48, 64, 128, 168, 192, 218, 235];
    
elseif strcmp(exp_id, '14_INS_10_BGLs')
    load('./SRs/SR_14_INS_10_BGLs.mat');
    bgls = [48, 64, 82, 96, 128, 168, 180, 192, 218, 235];
    
elseif strcmp(exp_id, '21_INS_10_BGLs')
    load('./SRs/SR_21_INS_10_BGLs.mat');
    bgls = [48, 64, 82, 96, 128, 168, 180, 192, 218, 235];
    
elseif strcmp(exp_id, 'circular')
    load('./SRs/SR_circular_10_INS_3_BGLs.mat');
    bgls = [32 128 192];
    
elseif strcmp(exp_id, 'line' )
    load('./SRs/SR_line_10_INS_3_BGLs.mat');
    bgls = [32 128 192];
else
    disp('Experiment identification is not known...')
end
    