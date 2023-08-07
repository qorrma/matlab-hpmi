clear;clc;
rng(128)

% CAE, VAE
bg_alg      = 'REG';
data_dir    = './data/private';
ckpt        = './results_230805';
k_sz        = 3;
Od          = 50;
D           = 50*exp(-Od+50);
K           = min(D,1);
DK          = 0.2*Od.^2-0.2*Od+1;
record      = 1;

% 19_INS_7_BGLs
% 33_INS_7_BGLs
% 14_INS_10_BGLs
% 21_INS_10_BGLs

exp_id = '33_INS_7_BGLs';
load_subjective_rankings;

[n_views, n_samples, n_bgls] = size(SubjectiveTest);
GT_data = squeeze(mean(SubjectiveTest,1))';

algs = ["semu", "dsemu", "sso", "tam", "pmqs", "aemm", "hpmi"];
n_algs = length(algs);

corrs = zeros(n_algs, n_bgls);
raes = zeros(n_algs, n_bgls);
mapes = zeros(n_algs, n_bgls);

%% Main
for cnt = 1:1:n_bgls
    bgl = bgls(cnt);
    GT = arrank2(GT_data(cnt, :), 1, 'ascend');
    save_dir = sprintf('%s/%d', ckpt, bgl);
    if exist(save_dir, 'file')==0; mkdir(save_dir); end
    
    for num=1:1:n_samples
        %% Load input Mura image
        Mura = double(imread(sprintf('%s/%d/Mura_%d.png', data_dir, bgl, num)));
        MFG = double(imread(sprintf('%s/%d/MFG_%d.png', data_dir, bgl, num)));
        [h, w] = size(Mura);
        
        %% Estimate base layer with polynomial regression
        if strcmp(bg_alg, 'CAE')
            Base = double(imread(sprintf('CAE/%d/Base_%02d.png', bgl, num)));
        elseif strcmp(bg_alg, 'VAE')
            Base = double(imread(sprintf('VAE/%d/Base_%02d.png', bgl, num)));
        else
            Base = RegNlls2D(Mura, 0);
        end
        
        if record > 0; imwrite(uint8(Base), sprintf('%s/%d/Base_%d.png', ckpt, bgl, num)); end
        
        %% Background-aware contrast map extraction
        [CMAP, NMAP] = cmap_fn(Mura, Base);
        if record > 0; imwrite(NMAP, sprintf('%s/%d/CMAP_%d.png', ckpt, bgl, num)); end
        
        %% Mura detection with region of interest
        CFG = AdaptiveThreshold(CMAP, MFG, 3);
        if record > 0; imwrite(CFG, sprintf('%s/%d/CFG_%d.png', ckpt, bgl, num)); end
        
        %% Extract average contrast, size, and length of major axis and minor axis
        [Cx, Sx, Ax] = FeatsExtr(CMAP, CFG);
        
        %% SEMU
        SEMU(num) = Cx./(1.24./(Sx.^0.33)+0.72);
        
        %% DSEMU
        DSEMU(num) = Cx./((2.2*DK).*(1./Ax.^0.33));
        
        %% SSO
        SSO(num) = SSO_index(CMAP, MFG);
        
        %% TAM
        a = -0.0695;
        b = 1.695;
        if bgl < 48
            TAM(num) = (a*(bgl/255)+b).*Cx;
        else
            TAM(num) = Cx;
        end
        
        %% PMQS
        PMQS(num) = Cx./(1.97./(Sx.^0.33)+0.72);
        
        %% AEMM
        SE_th = CMAP(CFG==1);
        Area_th = sum(sum(CFG ~= 1));
        AEMM(num) = sum(SE_th.^2) / Area_th;
        
        %% HPMI
        L_BGL = bgl/255;
%         w_c = -1.3387*L_BGL^2 + 2.0845*L_BGL + 0.0655;
%         w_s = 0.3292*L_BGL^2 - 0.5124*L_BGL + 0.5266;
%         w_a = 0.2920*L_BGL^2 - 0.4192*L_BGL + 0.4749;
        
        w_c = -1.3365*L_BGL^2 + 2.0051*L_BGL + 0.1557;
        w_s = 0.4215*L_BGL^2 - 0.6156*L_BGL + 0.5495;
        w_a = 0.4031*L_BGL^2 - 0.5479*L_BGL + 0.5063;
        
        if bgl <= 48
            HPMI(num) = (Cx.*(Sx.^.7)*(Ax.^.33)) / D;
        else
            HPMI(num) = (Cx^(w_c)*Sx^(0.33*w_s)*Ax^(0.33*w_a)) / D;
        end
    end
    
    %% Objective ranking (OR)
    SEMU_rank = arrank2(SEMU, 1, 'descend');
    DSEMU_rank = arrank2(DSEMU, 1, 'descend');
    SSO_rank = arrank2(SSO, 1, 'descend');
    TAM_rank = arrank2(TAM, 1, 'descend');
    PMQS_rank = arrank2(PMQS, 1, 'descend');
    AEMM_rank = arrank2(AEMM, 1, 'descend');
    HPMI_rank = arrank2(HPMI, 1, 'descend');
    
    gt(cnt, :) = GT;
    semu(cnt, :) = SEMU_rank;
    dsemu(cnt, :) = DSEMU_rank;
    sso(cnt, :) = SSO_rank;
    tam(cnt, :) = TAM_rank;
    pmqs(cnt, :) = PMQS_rank;
    aemm(cnt, :) = AEMM_rank;
    hpmi(cnt, :) = HPMI_rank;
    
    %% Pearson correlation coefficient (PCC)
    corrs(1, cnt) = pcc(SEMU_rank, GT);
    corrs(2, cnt) = pcc(DSEMU_rank, GT);
    corrs(3, cnt) = pcc(SSO_rank, GT);
    corrs(4, cnt) = pcc(TAM_rank, GT);
    corrs(5, cnt) = pcc(PMQS_rank, GT);
    corrs(6, cnt) = pcc(AEMM_rank, GT);
    corrs(7, cnt) = pcc(HPMI_rank, GT);
    
    %% Relative absolute errors
    raes(1, cnt) = rae(SEMU_rank, GT);
    raes(2, cnt) = rae(DSEMU_rank, GT);
    raes(3, cnt) = rae(SSO_rank, GT);
    raes(4, cnt) = rae(TAM_rank, GT);
    raes(5, cnt) = rae(PMQS_rank, GT);
    raes(6, cnt) = rae(AEMM_rank, GT);
    raes(7, cnt) = rae(HPMI_rank, GT);
    
    %% Mean absolute percentage errors
    mapes(1, cnt) = mape(SEMU_rank, GT);
    mapes(2, cnt) = mape(DSEMU_rank, GT);
    mapes(3, cnt) = mape(SSO_rank, GT);
    mapes(4, cnt) = mape(TAM_rank, GT);
    mapes(5, cnt) = mape(PMQS_rank, GT);
    mapes(6, cnt) = mape(AEMM_rank, GT);
    mapes(7, cnt) = mape(HPMI_rank, GT);
    
    fprintf('------------------------------\n');
end

rae_idx = 2*(1:1:n_bgls)-1;
mape_idx = 2*(1:1:n_bgls);

rae_mapes(:, rae_idx) = raes;
rae_mapes(:, mape_idx) = mapes;

function rst = rae(x, y)
lv = sum(abs(y-x));
tv = sum(abs(y-mean(y)));
rst = lv / tv;
end

function rst = mape(x, y)
rst = mean(abs((y-x)/y));
end

function rst = pcc(x,y)
corrs = zeros(size(y,1), 1);
for i=1:1:size(y,1)
    target = y(i,:);
    numerator = cov(x,target);
    denominator = std(x)*std(target);
    corr_mat = numerator./denominator;
    corrs(i) = corr_mat(1,2);
end
rst = mean(corrs);
end