clear; clc;
%Loading data
data = load("speech_dataset.mat");
data = data.dataset;
%Splitting dataset into training set and test set
%Remark: 3818-signals of class-1
%        1586-signals of class-2
%Using the 70-30 rule (70% of data for training set and 30% of data for
%test set)
%In order to have the same proportionality between class1 and class2 in the
%training set and test set, I choose 70% of class1 and 70% of class2 for
%the training set and the remaining data for the test set.
r = 0.70;
iClass1 = find(data(:,width(data))==1);
iClass2 = find(data(:,width(data))==2);
training = cat(1, data(iClass1(1:floor(length(iClass1)*r)),:),...
               data(iClass2(1:floor(length(iClass2)*r)),:));
%training = training(randperm(length(training)),:);
test = setdiff(data, training,'rows');
%test = test(randperm(length(test)),:);
clear("data","iClass1","iClass2","r");

kMax = 50;
k = round(linspace(1, length(training) - 1, kMax));

Error1 = ones(2, size(k, 2));
Accuracy = ones(2, size(k, 2));

for i = 1:length(k)
   [Error1(1, i), Accuracy(1, i)] = kNNclassifier(k(i), training, test);
   [Error1(2, i), Accuracy(2, i)] = kNNclassifier(k(i), training, training);
end

figure();
% subplot(2,1,1);
plot(k, Error1(1, :),'-*');
hold on;
plot(k,Error1(2, :),'-o');
grid minor;
xlabel('k');ylabel('Misclassification rate');
legend('Test Data','Training Data','Location','southoutside','Orientation','horizontal');
ylim([0 2*max(Error1(1,i))]);
axis auto; title('Misclassification rate');

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