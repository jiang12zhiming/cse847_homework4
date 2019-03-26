% load data
load('ad_data.mat');
[N,D] = size(X_train);
X_train = [ones(N,1) X_train];
[N,D] = size(X_test);
X_test = [ones(N,1) X_test];

% options and paramters
opts.rFlag = 1;
opts.tol = 1e-6;
opts.tFlag = 4;
opts.maxIter = 5000;
par = [0.01,0.03,0.04,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];

% regression
num_of_features = zeros(length(par),1);
AUC = zeros(length(par),1);
for i = 1:length(par)
    %call l1 logistic regression
    [w, c] = LogisticR(X_train, y_train, par(i), opts);
    %number of none zeros 
    n = nnz(w);
    num_of_features(i, 1) = n;
    
    y = X_test * w + c;
    predict = 1 ./ (1+exp(-y));
    [X,Y,T,AUC(i,1)] = perfcurve(y_test, predict, 1);
end

% plot
figure(1);
plot(par, num_of_features, '--o',...
    'LineWidth', 2,...
    'MarkerFaceColor', 'b');
xlabel('L_1 Regularization Parameter');
ylabel('# features');
grid on;
figure(2);
plot(par, AUC, '--*',...
    'LineWidth', 2);
xlabel('L_1 Regularization Parameter');
ylabel('AUC');
grid on;