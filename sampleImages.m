function X = sampleImages(sampleSize, baseSize, dataDir)
    % INPUT variables:
    % sampleSize            total number of patches to take
    % patchSize             patch width in pixels
    %
    % OUTPUT variables:
    % X                  the patches as column vectors

    files = dir(fullfile(dataDir, '*.mat') );
    n = min(length(files), 5000);
    samplePerImage = sampleSize / n;
    
    X = zeros(baseSize.^2, sampleSize);

    for i = 1:n
        disp(i/n)
        load(fullfile(dataDir, files(i).name), "data");
        x = data;
        xx = floor(rand(samplePerImage, 1) * (size(x, 1) - baseSize - 1)) + 1;
        yy = floor(rand(samplePerImage, 1) * (size(x, 2) - baseSize - 1)) + 1;
        for j = 1:samplePerImage
            X(:, (i-1)*samplePerImage+j) = reshape(x(xx:xx+baseSize-1, yy:yy+baseSize-1), [baseSize.^2, 1]);
        end
    end
end
