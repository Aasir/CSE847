% Importing data
load('data/alzheimers/ad_data.mat');

% Adding bias
X_train = [ones(size(X_train, 1), 1), X_train];
X_test = [ones(size(X_test, 1), 1), X_test];

reg_params = [1e-10, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

% Accuracy
roc_acc = zeros(size(reg_params));
feature_vals = zeros(size(reg_params));

for i=1:size(reg_params,2)
    reg = reg_params(i);
    % Train
    [w, c] = logistic_l1_train(X_train, y_train, reg);
    
    % Test
    y = sigmf(X_test * w, [1,0]);

    % features = Non zero values
    feature_vals(i) = sum(w ~= 0);
    [~, ~, ~, roc_acc(i)] = perfcurve(y_test, y, 1);
end

figure
plot(reg_params, feature_vals, '-o')
title('{\bf Features vs. Regularization}')
xlabel('Regularization')
ylabel('Features (Count of non-zero weight values)')

figure
plot(reg_params, roc_acc, '-o')
title('{\bf AUC vs. Regularization}')
xlabel('Regularization')
ylabel('Area Under Curve (AUC)')