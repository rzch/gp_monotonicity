%This script performs the same function as demo.m except on scale with
%1000 Monte-Carlo simulations
%Statistics are computed at the end

clc
clear
close all

%true utility as a multivariate Gaussian CDF
u = @(x) mvncdf([x(1), x(2)], [0.2, 0.4], diag([0.07, 0.05]));

%%

%points to sample from a grid
gridsize = 5;
[X1, X2] = meshgrid(linspace(0, 1, gridsize), linspace(0, 1, gridsize));

X1 = reshape(X1, length(X1)^2, 1);
X2 = reshape(X2, length(X2)^2, 1);

%create test grid for use in evaluation
tgridsize = 9;
[T1, T2] = meshgrid(linspace(0, 1, tgridsize), linspace(0, 1, tgridsize));
T1 = reshape(T1, length(T1)^2, 1);
T2 = reshape(T2, length(T2)^2, 1);
%generate all combinations of pairs
tcomb1 = combnk(T1, 2);
tcomb2 = combnk(T2, 2);
t1 = [tcomb1(:, 1), tcomb2(:, 1)]';
t2 = [tcomb1(:, 2), tcomb2(:, 2)]';
tpref = zeros(1, length(t1));
%calculate a vector all the pairwise comparisons
for i = 1:length(t1)
    tpref(i) = (u(t1(:, i)) >= u(t2(:, i)));
end

%generate combinations of pairs for the sample data
comb1 = combnk(X1, 2);
comb2 = combnk(X2, 2);
x1comb = [comb1(:, 1), comb2(:, 1)]';
x2comb = [comb1(:, 2), comb2(:, 2)]';

%indices of combinations
combind_master = combnk(1:length(X1), 2);

%data matrix made from all pairs
X = [X1, X2];

factor = 0.3; %proportion of points to sample
Nsample = round(factor*length(x1comb)); %number of points to sample

Nsim = 1000; %number of simulations

alpha_results = zeros(1, Nsim);
score_MLE_results = zeros(1, Nsim);
score_MLE_mono_results = zeros(1, Nsim);
score_MLE_nonmono_results = zeros(1, Nsim);
score_GP_results = zeros(1, Nsim);
score_GP_mono_results = zeros(1, Nsim);
score_GP_nonmono_results = zeros(1, Nsim);
score_MAP_results = zeros(1, Nsim);
score_MAP_mono_results = zeros(1, Nsim);
score_MAP_nonmono_results = zeros(1, Nsim);

alphares = 21;
score_alpha_sweep_results = zeros(Nsim, alphares);

parfor N = 1:Nsim
    
    %take a random subset of the pairs to use in learning
    sampled = 0;
    while (~sampled)
        factor = 0.3;
        rs = randsample(1:length(x1comb), Nsample);
        x1 = x1comb(:, rs);
        x2 = x2comb(:, rs);
        combind = combind_master(rs, :);
        %only exit loop when every point from the grid is represented
        sampled = (length(unique(combind)) == gridsize^2);
    end
    
    %number of comparisons
    M = length(x1);
    
    %standard deviation of rating noise
    sig_noise_real = 0.1;
    
    indA = zeros(1, M);
    indB = zeros(1, M);
    xA = zeros(2, M);
    xB = zeros(2, M);
    for i = 1:M
        %compute 'corrupted utilities' with gaussian noise
        v1 = u(x1(:, i)) + sig_noise_real*randn;
        v2 = u(x2(:, i)) + sig_noise_real*randn;
        
        %compare corrupted utilities to generate data and record indices
        if v1 > v2
            indB(i) = combind(i, 1);
            indA(i) = combind(i, 2);
            xB(:, i) = x1(:, i);
            xA(:, i) = x2(:, i);
        else
            indA(i) = combind(i, 1);
            indB(i) = combind(i, 2);
            xA(:, i) = x1(:, i);
            xB(:, i) = x2(:, i);
        end
    end
    
    %preference learning
    [f_MLE, y, f_MAP, alpha, K, sig, ells, beta_star_normalised] = pref_learn(X, xA, xB, indA, indB);
    
    
    %% Evaluate the results
    
    u1 = pred_GP(t1', y, K, X, sig, ells, beta_star_normalised);
    u2 = pred_GP(t2', y, K, X, sig, ells, beta_star_normalised);
    test_GP = (u1 >= u2)';

    mono_indices = [];
    nonmono_indices = [];
    for i = 1:length(t1)
        if (sum(t2(:, i) >= t1(:, i)) == 2 || sum(t1(:, i) >= t2(:, i)) == 2)
            mono_indices = [mono_indices, i];
        else
            nonmono_indices = [nonmono_indices, i];
        end
    end
    score_GP = sum(test_GP == tpref)/length(tpref);
    score_GP_mono = sum(test_GP(mono_indices) == tpref(mono_indices))/length(mono_indices);
    score_GP_nonmono = sum(test_GP(nonmono_indices) == tpref(nonmono_indices))/length(nonmono_indices);
    
    test_MLE = (t1'*beta_star_normalised >= t2'*beta_star_normalised)';
    score_MLE = sum(test_MLE == tpref)/length(tpref);
    score_MLE_mono = sum(test_MLE(mono_indices) == tpref(mono_indices))/length(mono_indices);
    score_MLE_nonmono = sum(test_MLE(nonmono_indices) == tpref(nonmono_indices))/length(nonmono_indices);
    
    u1 = pred_GP(t1', f_MAP, K, X, sig, ells, beta_star_normalised);
    u2 = pred_GP(t2', f_MAP, K, X, sig, ells, beta_star_normalised);
    test_MAP = (u1 >= u2)';
    score_MAP = sum(test_MAP == tpref)/length(tpref);
    score_MAP_mono = sum(test_MAP(mono_indices) == tpref(mono_indices))/length(mono_indices);
    score_MAP_nonmono = sum(test_MAP(nonmono_indices) == tpref(nonmono_indices))/length(nonmono_indices);
    
    
    %Sweep across alpha and compute prediction accuracy
    alpha_results(1, N) = alpha;
    score_MLE_results(1, N) = score_MLE;
    score_MLE_mono_results(1, N) = score_MLE_mono;
    score_MLE_nonmono_results(1, N) = score_MLE_nonmono;
    score_MAP_results(1, N) = score_MAP;
    score_MAP_mono_results(1, N) = score_MAP_mono;
    score_MAP_nonmono_results(1, N) = score_MAP_nonmono;
    score_GP_results(1, N) = score_GP;
    score_GP_mono_results(1, N) = score_GP_mono;
    score_GP_nonmono_results(1, N) = score_GP_nonmono;
    
    alphas = linspace(0, 1, alphares);
    
    for nsweep = 1:alphares
        y_sweep = alphas(nsweep)*f_MLE + (1 - alphas(nsweep))*f_MAP;
        u1 = pred_GP(t1', y_sweep, K, X, sig, ells, beta_star_normalised);
        u2 = pred_GP(t2', y_sweep, K, X, sig, ells, beta_star_normalised);
        test_temp = (u1 >= u2)';
        score_alpha_sweep_results(N, nsweep) = sum(test_temp == tpref)/length(tpref);
    end
    
    disp(N)
end

%%

%Create boxplot of prediction accuracy on alpha
plotflag = 1;
saveflag = 0;
if (plotflag)
    figure(1)
    boxplot(100*score_alpha_sweep_results, 'Colors', 'k', 'BoxStyle', 'filled');
    xlabel('$\alpha$', 'FontSize', 14, 'Interpreter', 'latex')
    xt = get(gca, 'XTick');
    set(gca, 'FontSize', 14)
    ylabel('Prediction Accuracy (%)', 'FontSize', 10)
    set(gca,'TickLabelInterpreter','latex')
    xticklabels({'0','','','','0.2','','', '', '0.4', '','', '', '0.6', '', '', '', '0.8', '', '', '', '1'})
    set(gcf,'units','points','position',[100,100,250,150])
    if (saveflag)
        saveas(gcf,'figures\boxplot','eps')
    end
    
    figure(2)
    set(gcf,'units','points','position',[100,100,250,150])
    histogram(alpha_results)
    xlabel('$\alpha^{*} + \epsilon$', 'FontSize', 14, 'Interpreter', 'latex')
    ylabel('Frequency', 'FontSize', 10)

    if (saveflag)
        saveas(gcf,'figures\histogram','eps')
    end    
end

mean_GP_mono = mean(score_GP_mono_results)
mean_GP_nonmono = mean(score_GP_nonmono_results)
mean_GP = mean(score_GP_results);
mean_MAP_mono = mean(score_MAP_mono_results)
mean_MAP_nonmono = mean(score_MAP_nonmono_results)
mean_MAP = mean(score_MAP_results);
mean_MLE_mono = mean(score_MLE_mono_results);
mean_MLE_nonmono = mean(score_MLE_nonmono_results);
mean_MLE = mean(score_MLE_results);