% load data
data = importdata('data.txt');
[N,dim] = size(data);
data = [data ones(N,1)]; % add an column at end with ones 
dim = dim+2;
labels = importdata('labels.txt');

% split data to train and test data
train_data = data(1:2000,:);
train_labels = labels(1:2000,:);
test_data = data(2001:4601,:);
test_labels = labels(2001:4601,:);

% train with different number of training samples
num_of_train_samples = [200,500,800,1000,1500,2000];
test_acc = zeros(length(num_of_train_samples),1);% zero matrix (6,1)
train_acc = zeros(length(num_of_train_samples),1);
number_of_iter = 5000;
epsilon = 1e-6;
for i=1:length(num_of_train_samples)
    w0 = zeros(dim,1);
    w = logistic_train_weight(train_data(1:num_of_train_samples(i),:),...
        train_labels(1:num_of_train_samples(i),:), w0,number_of_iter,epsilon);
    predicts = logistic_prediction(test_data, w);
    train_predicts = logistic_prediction(train_data(1:num_of_train_samples(i),:), w);
    test_acc(i) = 1 - sum(abs(predicts-test_labels)) / size(test_data,1);
    train_acc(i) = 1-sum(abs(train_predicts-train_labels(1:num_of_train_samples(i)))) / num_of_train_samples(i);
end

% plot the results
figure(1)
plot(num_of_train_samples, test_acc, '--o', 'LineWidth', 2,...
    'MarkerFaceColor','b');
grid on;
xlabel('# Training Samples');
ylabel('Accuracy');

figure(2)
plot(num_of_train_samples, train_acc, '--o', 'LineWidth', 2,...
    'MarkerFaceColor','b')

% sigmoid
function output = sigmoid(input)
    output = 1/(1+exp(- input));
end
