%External source used
%http://uc-r.github.io/discriminant_analysis#prep
%https://github.com/farnazgh/Linear-discriminant-analysis/blob/master/LDA.m

function LDA()
%reading data from files
T = readtable('inputdata.txt','Format','%f%f%f%s');
Main_table = T(:,1:4);
Table2 = readtable('sampledata.txt','Format','%f%f%f');

table_size = size(Main_table);
rows_count = table_size(1);

X = T(:,1:3);
X = table2array(X);
Xtest = table2array(Table2);


%Normalizing dataset against each feature
m = length(X);
[X_norm] = normalization(X);
[Xtrain_norm] = normalization(Xtest);

%Converting class labels for 0 and 1
y = T(:,4);
yt = table2array(y);
ytrain = zeros(length(yt),1);
N = length(ytrain);
countM = 0;
countW = 0;
for i=1:length(yt)
    j = char(T{i,4});
    if(j == 'W')
        ytrain(i,:) = 0;
        countM = countM + 1;
    else
        ytrain(i,:) = 1;
        countW = countW + 1;
    end
end


%calculate prior probability for class M and W

piW = countW/N;
piM = countM/N;

%calculate mean for class 'W'
meanW = mean(X_norm(ytrain(:,1) == 0,:));
%meanW = mean(X_norm(ytest(:,1) == 0),1:3);

%calculate mean for class 'M'
%meanM = mean(X_norm(ytest(:,1) == 1),1:3);
meanM = mean(X_norm(ytrain(:,1) == 1,:));

% computing single covariance matrix used for all class type
cov_main = zeros(length(X_norm),3);
for p=1:N
    class_type = char(Main_table{p,4});
    if(class_type == 'W')
        cov_main(p,:) = X_norm(p,:) - meanW;
    else
        cov_main(p,:) = X_norm(p,:) - meanM;
    end
end

cov_main = (cov_main'*cov_main)./N;

%prediction using the formula for training data
for s = 1:N
    prob_classM = X_norm(s,:)*inv(cov_main)*meanM' - (0.5).*meanM*inv(cov_main)*meanM' + log(piM);
    prob_classW = X_norm(s,:)*inv(cov_main)*meanW' - (0.5).*meanW*inv(cov_main)*meanW' + log(piW);
    
    if(prob_classM > prob_classW)
        result = 1;
    else
        result = 0;
    end
        
    fprintf("\n Predicted %d , Actual %d",result,ytrain(s,1));
end

fprintf("\n");

% prediction of test data
testlabel = [0,1,0,1];

for s = 1:length(Xtrain_norm)
    prob_classM = Xtrain_norm(s,:)*inv(cov_main)*meanM' - (0.5).*meanM*inv(cov_main)*meanM' + log(piM);
    prob_classW = Xtrain_norm(s,:)*inv(cov_main)*meanW' - (0.5).*meanW*inv(cov_main)*meanW' + log(piW);
    
    if(prob_classM - prob_classW < 0)
        result = 0;
    else
        result = 1;
    end
        
    fprintf("\n Predicted %d , Actual %d",result,testlabel(s));
end

%decision boundary for Training data
first = 0;
second = 0;
trainresult = 0

for p = 1:length(X_norm)
    first = first + (X_norm(s,:)- meanM)*inv(cov_main)*(X_norm(s,:)- meanM)';
    second = second + (X_norm(s,:)- meanW)*inv(cov_main)*(X_norm(s,:)- meanW)'; 
    if(first < second)
        break;
    end
end

trainresult = first - second;
fprintf("\t decision boundary for training data %f",trainresult);

%decision boundary for Test data
firstTerm = 0;
secondTerm = 0;
result = 0

for p = 1:length(Xtrain_norm)
    firstTerm = firstTerm + (Xtrain_norm(s,:)- meanM)*inv(cov_main)*(Xtrain_norm(s,:)- meanM)';
    secondTerm = secondTerm + (Xtrain_norm(s,:)- meanW)*inv(cov_main)*(Xtrain_norm(s,:)- meanW)'; 
    if(firstTerm < secondTerm)
        break;
    end
end

result = firstTerm - secondTerm;
fprintf("\t decision boundary for test data %f",result);

%generating datapoints and visualization

detCov = det(cov_main);
detCovSqrt = power(detCov ,0.5);
pterm = (2*3.14);
pt = power(pterm,N/2);

%dataPointsForClassM = rand([50,3]);
dataPointsForClassM = randi([20,200],50,3);
dataPointsForClassW = randi([20,200],50,3);
[dataPointsForClassM_Norm, mu, sigma] = normalization(dataPointsForClassM);
[dataPointsForClassW_Norm, mu, ~] = normalization(dataPointsForClassW);

for m = 1:length(dataPointsForClassM)
    term = -(1/2).*(dataPointsForClassM(m,:)-meanM)*inv(cov_main)*(dataPointsForClassM(m,:)-meanM)';
    %fprintf("\t%f",exp(term));
    dataPointsForClassM(m,:) = (exp(term))/(detCovSqrt*pt);
end

for m = 1:length(dataPointsForClassW)
    term = -(1/2).*(dataPointsForClassW(m,:)-meanW)*inv(cov_main)*(dataPointsForClassW(m,:)-meanW)';
    dataPointsForClassW(m,:) = (exp(term))./(detCovSqrt.*pt);
end

Xterm = X(:,1);
yterm = X(:,2);

%Plotting this point
figure;
plot(Xterm,yterm,'*');
xlabel('Height (from Training Set)'); ylabel('Weight(from Training Set)');
hold on;

Xterm1 = dataPointsForClassM_Norm(:,1);
yterm1 = dataPointsForClassM_Norm(:,2);

figure;
plot(Xterm1,yterm1,'*');
xlabel('Height (Genereated data set from model for class M)'); ylabel('Weight (Genereated data set from model for class M)')
Xterm2 = dataPointsForClassW_Norm(:,1);
yterm2 = dataPointsForClassW_Norm(:,2);

figure;
plot(Xterm2,yterm2,'*');
xlabel('Height (Genereated data set from model for class W)'); ylabel('Weight (Genereated data set from model for class W)');
hold off;

end

