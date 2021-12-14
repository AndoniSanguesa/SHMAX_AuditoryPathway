function base = SHMAX_3dS(numLayer, status, skipTrainingList, skipInferenceList, nBaseList,  baseSizeList, sampleSizeList, sstrideList, paramSC, dataDirList, resultDirList, poolSizeList, cstrideList, file_name, y)
    skipTraining = skipTrainingList(numLayer);
    skipInference = skipInferenceList(numLayer);
    nBase = nBaseList(numLayer);
    baseSize = baseSizeList(numLayer);
    sampleSize = sampleSizeList(numLayer);
    stride = sstrideList(numLayer);
    dataDir = convertStringsToChars(dataDirList(numLayer));
    resultDir = convertStringsToChars(resultDirList(numLayer));
    poolSize = poolSizeList(numLayer);
    cstride = cstrideList(numLayer);

    if ~(numLayer == 6)
        nextBaseSize = baseSizeList(numLayer+1);
        nextSampleSize = sampleSizeList(numLayer+1);
    end
    % SHMAX: S layer, 3d data
    % Learn bases
    fprintf('Learning bases...\n');
    tic;
    if skipTraining
        load([resultDir, filesep, 'base_', num2str(nBase), '_', num2str(sampleSize), '_', num2str(baseSize), '.mat'], 'base');
    else
        full_sample = zeros(baseSize.^2 * size(y, 3), sampleSize);
        files = dir(fullfile(dataDir, 'sample_*.mat') );
        for i = 1:length(files)
            load(fullfile(dataDir, files(i).name), 'sample');
            full_sample(:, i) = sample;
        end
        base = mexTrainDL(sample, paramSC);
        if ~exist(resultDir, 'dir')
            mkdir(resultDir);
        end
        save([resultDir, filesep, 'base_', num2str(nBase), '_', num2str(sampleSize), '_', num2str(baseSize), '.mat'], 'base');
    end
    toc;

    % Inference++++++++
    fprintf('Calculating responses...\n');
    tic;
    if ~skipInference
        if ~status(numLayer)
            n = min(length(files), 5000);
            samplePerImage = nextSampleSize / n;
        end
        tic;
        disp(i/length(files))
        y = gpuArray(y);
        w = zeros([size(y, 1) - baseSize + 1, size(y, 2) - baseSize + 1, nBase]);
        for j = 1:nBase
            kernel = gpuArray(reshape(base(:, j), baseSize, baseSize, size(y, 3)));
            w(:, :, j) = convn(y, kernel, 'valid');
        end
        
        w = w(1:stride:end, 1:stride:end, :);
        y = SHMAX_C(poolSize, cstride, w);

        if ~status(numLayer) && i <= n
            xx = 1 + floor(rand(samplePerImage, 1) * (size(w, 1) - nextBaseSize));
            yy = 1 + floor(rand(samplePerImage, 1) * (size(w, 2) - nextBaseSize));
            for j = 1:samplePerImage
                sample = reshape(y(xx:xx+nextBaseSize-1, yy:yy+nextBaseSize-1, :), [nextBaseSize.^2 * size(y, 3), 1]);
                save([resultDir, filesep, 'sample_', int2str((i-1)*samplePerImage+j), '.mat'], 'sample')
            end
        end
        toc;
        if status(numLayer)
            if ~(numLayer == 6)
                SHMAX_3dS(numLayer + 1, status, skipTrainingList, skipInferenceList, nBaseList, baseSizeList, sampleSizeList, sstrideList, paramSC, dataDirList, resultDirList, poolSizeList, cstrideList, file_name, y)
            else
                save([resultDir, filesep, 'y_', file_name, '.mat'], 'y')
            end
        end
    end
    toc;
end