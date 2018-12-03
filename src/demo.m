%This script performs a single demonstration of preference learning using
%Gaussian processes with monotonicity constraints from pairwise comparisons
%on a 2d example

clc
clear
close all

saveflag = 0; %flag whether to save plots throughout script
plotflag = 1; %flag whether to plot plots throughout script

%% True utility

%true utility as a multivariate Gaussian CDF
u = @(x) mvncdf([x(1), x(2)], [0.2, 0.4], diag([0.07, 0.05]));

%compute and plot true utility surface
[X1, X2] = meshgrid(linspace(0, 1), linspace(0, 1));

%Plot true utility surface
for i = 1:length(X1)
    for j = 1:length(X1)
        U(i, j) = u([X1(i, j); X2(i, j)]);
    end
end

if (plotflag)
    figure(1)
    mesh(X1, X2, U)
    colormap copper
    xlabel('x1')
    ylabel('x2')
end

%contour plot of true utility
if (plotflag)
    figure(3)
    contour(X1, X2, U, 0:0.05:1, 'ShowText', 'on', 'LabelSpacing', 500, 'TextList', 0.1:0.1:0.9)
    colormap([0 0 0])
    xlabel('$x_{1}$', 'Interpreter', 'latex', 'FontSize', 14)
    ylabel('$x_{2}$', 'Interpreter', 'latex', 'FontSize', 14)
    set(get(gca,'ylabel'), 'Rotation', 0)
    set(gcf,'units','points','position',[100,100,250,150])
    hold on
end

%% Sampling and Learning

%points to sample from a grid for data generation
gridsize = 5;
[X1, X2] = meshgrid(linspace(0, 1, gridsize), linspace(0, 1, gridsize));

X1 = reshape(X1, length(X1)^2, 1);
X2 = reshape(X2, length(X2)^2, 1);

if (plotflag)
    scatter(X1, X2, 'k', 'filled')
    if (saveflag)
        saveas(gcf,'figures\utility_function','eps')
    end
end

%create test grid for use in evaluation/validation
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
x1 = [comb1(:, 1), comb2(:, 1)]';
x2 = [comb1(:, 2), comb2(:, 2)]';

%indices of combinations
combind = combnk(1:length(X1), 2);

%take a random subset of the pairs to use in learning
sampled = 0;
while (~sampled)
    factor = 0.3; %fraction to sample
    rs = randsample(1:length(x1), round(factor*length(x1)), 0);
    x1 = x1(:, rs);
    x2 = x2(:, rs);
    combind = combind(rs, :);
    %only exit loop when every point from the grid is represented
    sampled = (length(unique(combind)) == gridsize^2);
end

%number of comparisons
M = length(x1);

%standard deviation of true rating noise
sig_noise_real = 0.1;

%generate comparisons using rating model
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

%data matrix made from all pairs
X = [X1, X2];
n = length(X);

%perform preference learning
[f_MLE, y, f_MAP, alpha, K, sig, ells, beta_star_normalised] = pref_learn(X, xA, xB, indA, indB);

%% Check Results

%grid to check monotonicity over
[X1, X2] = meshgrid(linspace(0, 1), linspace(0, 1));

if (plotflag)
    %plot true utility
    figure(1)
    hold on
    scatter3(X(:, 1), X(:, 2), f_MLE, 'm', 'filled')
    title('Learned MLE Utility Function')
    legend('True Utility Function', 'MLE Estimates', 'Location','northeast')
end


%true utilities
for i = 1:length(X)
    f_true(i) = u(X(i, :)');
end


%compute and plot prediction mean surface with other estimates

%monotonic constrained posterior
f_pred_GP = zeros(length(X1), length(X2));
for i = 1:length(X1)
    for j = 1:length(X1)
        f_pred_GP(i, j) = pred_GP([X1(i, j), X2(i, j)], y, K, X, sig, ells, beta_star_normalised);
    end
end

%not monotonicity constrained posterior (f_MAP)
f_pred_GP_old = zeros(length(X1), length(X2));
for i = 1:length(X1)
    for j = 1:length(X1)
        f_pred_GP_old(i, j) = pred_GP([X1(i, j), X2(i, j)], f_MAP, K, X, sig, ells, beta_star_normalised);
    end
end

%plotting
if (plotflag)
    figure(4)
    contour(X1, X2, f_pred_GP, 0:0.05:1, 'ShowText', 'on', 'LabelSpacing', 500, 'TextList', 0.1:0.1:0.9, 'LineWidth', 1.5)
    colormap([0 0 0])
    hold on
    contour(X1, X2, U, 0:0.05:1, 'ShowText', 'off', 'LineStyle', '--')
    colormap([0 0 0])
    xlabel('$x_{1}$', 'Interpreter', 'latex', 'FontSize', 14)
    ylabel('$x_{2}$', 'Interpreter', 'latex', 'FontSize', 14)
    set(get(gca,'ylabel'), 'Rotation', 0)
    set(gcf,'units','points','position',[100,100,250,150])
    if (saveflag)
        saveas(gcf,'figures\pred_contour','eps')
    end

    figure(5)
    contour(X1, X2, X1*beta_star_normalised(1) + X2*beta_star_normalised(2), 0:0.05:1, 'ShowText', 'on', 'LabelSpacing', 500, 'TextList', 0.1:0.1:0.9, 'LineWidth', 1.5)
    colormap([0 0 0])
    hold on
    contour(X1, X2, U, 0:0.05:1, 'ShowText', 'off', 'LineStyle', '--')
    colormap([0 0 0])
    xlabel('$x_{1}$', 'Interpreter', 'latex', 'FontSize', 14)
    ylabel('$x_{2}$', 'Interpreter', 'latex', 'FontSize', 14)
    set(get(gca,'ylabel'), 'Rotation', 0)
    set(gcf,'units','points','position',[100,100,250,150])
    if (saveflag)
        saveas(gcf,'figures\mle_contour','eps')
    end
    
    figure(6)
    contour(X1, X2, f_pred_GP_old, 0:0.05:1, 'ShowText', 'on', 'LabelSpacing', 500, 'TextList', 0.1:0.1:0.9, 'LineWidth', 1.5)
    colormap([0 0 0])
    hold on
    contour(X1, X2, X1*beta_star_normalised(1) + X2*beta_star_normalised(2), 0:0.05:1, 'ShowText', 'off', 'LineStyle', '--')
    colormap([0 0 0])
    xlabel('$x_{1}$', 'Interpreter', 'latex', 'FontSize', 14)
    ylabel('$x_{2}$', 'Interpreter', 'latex', 'FontSize', 14)
    set(get(gca,'ylabel'), 'Rotation', 0)
    set(gcf,'units','points','position',[100,100,250,150])
    if (saveflag)
        saveas(gcf,'figures\map_contour','eps')
    end
    
    
    figure(8)
    contour(X1, X2, f_pred_GP_old, 0:0.05:1, 'ShowText', 'on', 'LabelSpacing', 500, 'TextList', 0.1:0.1:0.9, 'LineWidth', 1.5)
    colormap([0 0 0])
    hold on
    contour(X1, X2, U, 0:0.05:1, 'ShowText', 'off', 'LineStyle', '--')
    colormap([0 0 0])
    xlabel('$x_{1}$', 'Interpreter', 'latex', 'FontSize', 14)
    ylabel('$x_{2}$', 'Interpreter', 'latex', 'FontSize', 14)
    set(get(gca,'ylabel'), 'Rotation', 0)
    set(gcf,'units','points','position',[100,100,250,150])
    if (saveflag)
        saveas(gcf,'figures\map_contour','eps')
    end
    
    figure(2)
    mesh(X1, X2, f_pred_GP)
    hold on
    title('Learned GP Utility Function')
    xlabel('x1')
    ylabel('x2')
    
    hold on
    scatter3(X(:, 1), X(:, 2), f_true, 'm', 'filled')
    scatter3(X(:, 1), X(:, 2), f_MAP, 'b', 'filled')
    scatter3(X(:, 1), X(:, 2), f_MLE, 'r', 'filled')
    legend('Posterior GP Mean', 'True Utility', 'GP Estimates', 'MLE Estimates', 'Location','northeast')
    axis([0 1 0 1 0 1])
end


%% Evaluate the results 

%predict utilities from test set 
u1 = pred_GP(t1', y, K, X, sig, ells, beta_star_normalised);
u2 = pred_GP(t2', y, K, X, sig, ells, beta_star_normalised);
test_GP = (u1 >= u2)';

%generate the indices of the pairs for dominated and non-dominated
%comparisons
mono_indices = [];
nonmono_indices = [];
for i = 1:length(t1)
    if (sum(t2(:, i) >= t1(:, i)) == 2 || sum(t1(:, i) >= t2(:, i)) == 2)
        mono_indices = [mono_indices, i];
    else
        nonmono_indices = [nonmono_indices, i];
    end
end
%score the comparisons using monotonic latent vector
score_GP = sum(test_GP == tpref)/length(tpref);
score_GP_mono = sum(test_GP(mono_indices) == tpref(mono_indices))/length(mono_indices);
score_GP_nonmono = sum(test_GP(nonmono_indices) == tpref(nonmono_indices))/length(nonmono_indices);

%score the comparisons using linear latent vector
test_MLE = (t1'*beta_star_normalised >= t2'*beta_star_normalised)';
score_MLE = sum(test_MLE == tpref)/length(tpref);
score_MLE_nonmono = sum(test_MLE(nonmono_indices) == tpref(nonmono_indices))/length(nonmono_indices);

%score the comparisons using MAP latent vector
u1 = pred_GP(t1', f_MAP, K, X, sig, ells, beta_star_normalised);
u2 = pred_GP(t2', f_MAP, K, X, sig, ells, beta_star_normalised);
test_MAP = (u1 >= u2)';
score_MAP = sum(test_MAP == tpref)/length(tpref);
score_MAP_mono = sum(test_MAP(mono_indices) == tpref(mono_indices))/length(mono_indices);
score_MAP_nonmono = sum(test_MAP(nonmono_indices) == tpref(nonmono_indices))/length(nonmono_indices);

%Sweep across alpha and compute prediction accuracy
alphas = sort([linspace(0, 1, 21), alpha]);

scores = zeros(1, length(alphas));
for nsweep = 1:length(alphas)
    y_sweep = alphas(nsweep)*f_MLE + (1 - alphas(nsweep))*f_MAP;
    u1 = pred_GP(t1', y_sweep, K, X, sig, ells, beta_star_normalised);
    u2 = pred_GP(t2', y_sweep, K, X, sig, ells, beta_star_normalised);
    test_temp = (u1 >= u2)';
    scores(nsweep) = sum(test_temp == tpref)/length(tpref);
end

figure(9)
plot(alphas, scores)
xlabel('Accuracy')
ylabel('\alpha')