% Importing data
data = load('data/spam_email/data.txt');
labels = load('data/spam_email/labels.txt');
labels(labels==-1.0) = 0.0;

% Add bias
data = [ones(size(data, 1), 1), data];
training_sizes = [200, 500, 800, 1000, 1500, 2000];

% Testing Split
test_data = data(2001:4601,:);
test_labels = labels(2001:4601,:);

% Accuracy
roc_acc = zeros(size(training_sizes));
accuracy = zeros(size(training_sizes));

for i = 1:size(training_sizes,2)
    train_size = training_sizes(i);
    train_data = data(1:train_size,:);
    train_labels = labels(1:train_size,:);
    
    % Train
    w = logistic_train(train_data, train_labels);
    % Test
    y = sigmf(test_data * w, [1,0]);

    % Classifying values based on weights and determining accuracy
    y(y >= 0.5) = 1;
    y(y < 0.5) = 0;
    accuracy(i) = sum(y==test_labels)/length(test_data);    
end

% Plot
figure;
plot(training_sizes, accuracy, 'o-');
title('{\bf Prob 1: Logistic Regression (Spam Emails)}');
xlabel('Number of training data points');
ylabel('Accuracy');
