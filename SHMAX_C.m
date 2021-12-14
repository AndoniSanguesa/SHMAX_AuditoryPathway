function y = SHMAX_C(poolSize, stride, w)
% SHMAX: C layer, 2d data
    fprintf('Calculating responses...\n');
    w = gpuArray(w);
    y = zeros(size(w) - [1, 1, 0]);
    for j = 1:size(w, 3)
        y(:, :, j) = col2im(max(im2col(w(:, :, j), [poolSize, poolSize]), [], 1), [poolSize, poolSize], size(w(:, :, j)), 'sliding');
    end
    y = y(1:stride:end, 1:stride:end, :);
end