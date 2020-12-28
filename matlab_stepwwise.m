X_data = [LOAN,	MORTDUE, VALUE,	REASON,	JOB, YOJ, DEROG, DELINQ, CLAGE, NINQ, CLNO, DEBTINC];

CVO = cvpartition(BAD, 'k', 50);

err = zeros(CVO.NumTestSets,1);

finalmodel = logical('NoneNoneNone');

for i = 1:CVO.NumTestSets
    trIdx = CVO.training(i);
    teIdx = CVO.test(i);
    testset = X_data(teIdx,:);
    trainset = X_data(trIdx,:);
    trainlable = BAD(trIdx,:);
    %sequentials
    %c = cvpartition(trainlable, 'k', 2);
    opts = statset('Display','iter');
    fun = @(XT,yT,Xt,yt)loss(fitcecoc(XT,yT),Xt,yt);
    %[fs,history] = sequentialfs(fun,trainset,trainlable,'options',opts);
    [fs,history] = sequentialfs(fun,trainset,trainlable,'options',opts);
    %stepwisefit
    [betahat_in,se_in,pval_in,finalmodel_in,stats_in] = stepwisefit(trainset,trainlable,'InModel',fs, 'PEnter', 0.03, 'PRemove', 0.07,'Display','off');
    finalmodel(i,:) = finalmodel_in;
end


%testset = X_data(teIdx,:);
%trainset= X_data(trIdx,:);
%trainlable= BAD(trIdx,:);
%cp = cellstr(num2str(X_data));

%sequentials
%c = cvpartition(BAD, 'k', 5);
%opts = statset('Display','iter');
%fun = @(XT,yT,Xt,yt)loss(fitcecoc(XT,yT),Xt,yt);
%[fs,history] = sequentialfs(fun,X_data,BAD,'cv',c,'options',opts)


%stepwisefit
%[betahat_in,se_in,pval_in,finalmodel_in,stats_in] = stepwisefit(X_data,BAD,'InModel',fs, 'PEnter', 0.03, 'PRemove', 0.07,'Display','off');



