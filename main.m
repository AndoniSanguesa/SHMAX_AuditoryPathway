%% Set parameters
setParameters;

status = [0, 0, 0, 0, 0, 1];
disp(dir())
dataDirList = ["SHMAX_AuditoryPathway/matdata" "S1Result" "S2Result" "S3Result" "S4Result" "S5Result"];
resultDirList = ["S1Result" "S2Result" "S3Result" "S4Result" "S5Result" "S6Result"];

for i = 1:6
    if ~param.sS(i)
    %     tic;
        SHMAX_2dS(1, status, param.sT, param.sI, param.Bn, param.Bs, param.Sn, param.s1, param.SC, dataDirList, resultDirList, param.Pr, param.s2);
        status(i) = 1;
    %     toc;
    end
end