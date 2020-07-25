
format compact
clear 
clc

%% Load data
data=load('../Datasets/haberman.data');


class2=data(find(data(:,4)==2),:);
class1=data(find(data(:,4)==1),:);

class1=class1(1:length(class2),:);
data=[class1',class2'];
data=data';


%% init arrays
Acc=zeros(2,1);
numOfRules=zeros(2,1);
Perf=zeros(1,6);
%% split the data to 60 training,20 val and 20 test.
preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);

%% ANFIS - Scatter Partition

%%Clustering Per Class
radius=0.9;
[c1,sig1]=subclust(trnData(trnData(:,end)==1,:),radius);
[c2,sig2]=subclust(trnData(trnData(:,end)==2,:),radius);
num_rules=size(c1,1)+size(c2,1);

%Build FIS From Scratch
fis=newfis('FIS_SC','sugeno');

%Add Input-Output Variables
names_in={'in1','in2','in3','in4','in5'};
for i=1:size(trnData,2)-1
    fis=addvar(fis,'input',names_in{i},[0 1]);
end
fis=addvar(fis,'output','out1',[0 1]);

%Add Input Membership Functions
name='sth';
for i=1:size(trnData,2)-1
    for j=1:size(c1,1)
        fis=addmf(fis,names_in{i},'gaussmf',[sig1(j) c1(j,i)]);
    end
    for j=1:size(c2,1)
        fis=addmf(fis,names_in{i},'gaussmf',[sig2(j) c2(j,i)]);
    end
end

%Add Output Membership Functions
%params=[zeros(1,size(c1,1)) ones(1,size(c2,1))];
params=[ones(1,size(c1,1)) 2*ones(1,size(c2,1))];
for i=1:num_rules
    fis=addmf(fis,'out1','constant',params(i));
end

%Add FIS Rule Base
ruleList=zeros(num_rules,size(trnData,2));
for i=1:size(ruleList,1)
    ruleList(i,:)=i;
end

ruleList=[ruleList ones(num_rules,2)];
fis=addrule(fis,ruleList);

fis=genfis2(trnData(:,1:end-1),trnData(:,end),radius);
    
%get the num of rules
numOfRules(i)=length(fis.Rules());
[trnFis,trnError,~,valFis,valError]=anfis(trnData,fis,[100 0 0.01 0.9 1.1],[],chkData);
    
%evaluation of model
Y=evalfis(tstData(:,1:end-1),valFis);
Y=round(Y);
diff=tstData(:,end)-Y;
Acc=(length(diff)-nnz(diff))/length(Y)*100;
    
%% Error Matrix
error_matrix = confusionmat(tstData(:,end), Y);
pa = zeros(1, 2);
ua = zeros(1, 2);
%% confusion matrix
figure;
cm = confusionchart(tstData(:,end),Y);
%% Producer’s accuracy – User’s accuracy
N = length(tstData);
for i = 1 : 2
    pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end
%% Overall accuracy
overall_acc = 0;
for i = 1 : 2
    overall_acc = overall_acc + error_matrix(i, i);
end
overall_acc = overall_acc / N;
%% k
p1 = sum(error_matrix(1, :)) * sum(error_matrix(:, 1)) / N ^ 2;
p2 = sum(error_matrix(2, :)) * sum(error_matrix(:, 2)) / N ^ 2;
pe = p1 + p2;
k = (overall_acc - pe) / (1 - pe);
%% Plot some input membership functions
figure;
plotMFs(fis,size(trnData,2)-1);
suptitle('TSK model : some membership functions before training');
%% Plot the input membership functions after training
figure;
plotMFs(valFis,size(trnData,2)-1);
suptitle('TSK model : some membership functions after training');
%% Learning curve
figure;
plot(1:length(trnError), trnError, 1:length(valError), valError);
title('TSK model : Learning Curve');
xlabel('iterations');
ylabel('Error');
legend('Training Set', 'Check Set');
