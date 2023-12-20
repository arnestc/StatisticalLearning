clear; clc;
%Loading data
data = load("synthetic.mat");
training = data.knnClassify2dTrain;
test = data.knnClassify2dTest;
clear("data")

kMax = 100;
k = round(linspace(1, length(training) - 1, kMax));
Error = ones(2, size(k, 2));
Accuracy = ones(2, size(k, 2));

for i = 1:length(k)
   [Error(1, i), Accuracy(1, i)] = kNNclassifier(k(i), training, test);
   [Error(2, i), Accuracy(2, i)] = kNNclassifier(k(i), training, training);
end

figure();
% subplot(2,1,1);
plot(k,Error(1, :),'-*',k, Error(2, :),'-o');
grid minor;
xlabel('k');ylabel('Misclassification rate');
legend('Test Data','Training Data','Location','southoutside','Orientation','horizontal');
ylim([0 2*max(Error(1,i))]);
axis auto;title('Misclassification rate');
% subplot(2,1,2);
% plot(k, Accuracy(1, :), '-x', k, Accuracy(2, :), '-s');
% grid minor; xlabel('k'); ylabel('Accuracy(%)');
% legend('Test Data', 'Training Data', 'Location','best'); ylim([0 2*max(Error(1,i))]);
% axis auto; title('Accuracy');

% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%kNNclassifier where Nc is the classes number, k is the NNs-number and
%training and test are the trainingset and testset
%kNNclassifier returns the misclassification rate and the accuracy
function [Misclass, Accuracy] = kNNclassifier(k, training, test)
    Indices = kNN(k, test, training);
    TrainingClass = training(:,end);
    IndexClass = TrainingClass(Indices);
    TestClass = mode(IndexClass, 2);
    Misclass = sum(TestClass ~= test(:,end)) / length(test);
    Accuracy = (sum(test(:,end) == TestClass) / length(test))*100;
end
%
%kNN(k,test,training) where k is the k-parameter, test is the test s and
%training is the training data
%kNN(k,test,training) returns the indices of k-NN from training set for
%each test set element
function I = kNN(k, test, training)
    %D is the distances matrix:
    %rows: test elements
    %columns: training elements
    D = zeros(length(test),length(training));
    %I is the k-NN indices (from training elements) for each test element
    I = zeros(length(test),k);
    for i = 1:length(test)
        for j = 1:length(training)
            D(i,j) = norm(test(i,:)-training(j,:));
        end
        [~,I(i,:)] = mink(D(i,:),k);
    end
end
%
%classP(Nc, I, test, training) where Nc is the classes number, I is the
%k-NN indices (from training elements) for each test element, test is the
%test s and training is the training data
%classP(Indices) returns the probability for each class for each test
%element
% function P = classP(Nc, I, test, training)
%     P = zeros(length(test),Nc);
%     for i = 1:Nc
%         IndFun = sum(reshape(training(I,width(training))==i,length(test),[]),2);
%         P(:,i) = IndFun / width(I);
%     end
% end