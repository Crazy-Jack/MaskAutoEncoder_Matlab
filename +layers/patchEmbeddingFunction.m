function [patchEmbeddingActivation] = patchEmbeddingFunction(inputImage, weight, bias, patchSize)
    patchEmbeddingActivation = layers.conv2d_forward(inputImage, weight, ...
        bias, patchSize, 0); % b, co, h, w
    
    [b, co, h, w] = size(patchEmbeddingActivation);
    patchEmbeddingActivation = permute(patchEmbeddingActivation, [1 2 4 3]); % b, co, h, w but row order
    patchEmbeddingActivation = reshape(patchEmbeddingActivation, [b, co, h * w]);
    patchEmbeddingActivation = permute(patchEmbeddingActivation, [2 3 1]); % L, numFeatures, b
end 