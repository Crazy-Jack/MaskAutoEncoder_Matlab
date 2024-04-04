
%% Hyper Parameters
randomMaskingRatio = 0.0; % modify this to generate mask randomly


inputSize = [224 224 3];
numFeature = 1024;
mlpRatio = 4;
numFeatureMLPHidden = mlpRatio * numFeature;
patchSize = 16;
maxNumTokens = (inputSize(1) / patchSize) * (inputSize(2) / patchSize);
gridSize_h = inputSize(1) / patchSize;
gridSize_w = inputSize(2) / patchSize;
numLayers = 24;
numHeads = 16;

numLayersDecoder = 8;
numHeadsDecoder = 16;
numFeatureDecoder = 512;
numFeatureMLPHiddenDecoder = mlpRatio * numFeatureDecoder;

%% I/O -- Load image
% img = imread('./img/fox.jpg'); % Replace with the path to your image
% img = imread('./img/grouping2.png');
% img = imread('./img/rings.png');
% img = imread('./img/Ktriangle.png');
% img = imread('./img/grouping.png');
img = imread('./img/pepper_tomato.png');
img = imresize(img, [224 224]); % Resize the image to fit the input layer
img = single(img) / 255;

%%% Normalize
imagenetMean = [0.485, 0.456, 0.406];
imagenetStd = [0.229, 0.224, 0.225];

%%%% normalize by ImageNet mean and std
% Initialize a container for the normalized image
normalizedImage = zeros(size(img));
for channel = 1:3
    % Extract one channel
    channelData = img(:,:,channel);

    % Normalize the current channel
    channelMean = imagenetMean(channel);
    channelStd = imagenetStd(channel);
    
    normalizedImage(:,:,channel) = (channelData - channelMean) / channelStd;
end
inputImage = dlarray(single(normalizedImage));

%% Load Pytorch MAE weights
path_to_weights = "./weights/mae_visualize_vit_large_ganloss_weight.mat";
mae_weight = load(path_to_weights);


%% Network Architecture - (1) - PatchEmbedding

inputImage_transpose = permute(inputImage, [4, 3, 1, 2]);
patchEmbeddingActivation = layers.patchEmbeddingFunction(inputImage_transpose, mae_weight.patch_embed_proj_weight, ...
    mae_weight.patch_embed_proj_bias, patchSize); 

%% Network Architecture - (2) - PositionalEmbedding
has_cls_token = false;
activationAfterPosEmb = layers.positionalEmbeddingFunction(patchEmbeddingActivation, has_cls_token);

%% Operation - (3) - Random Masking
rng("default"); % set random seed for random masking selection
[actEmbeddingSelect, selectedTokenIndices] = layers.randomMasking(activationAfterPosEmb, img, patchSize, randomMaskingRatio);


%% Network Architecture - (4) CLS token 
cls_token = reshape(mae_weight.cls_token, [size(mae_weight.cls_token, 3)], 1);
actEmbeddingSelectwithCLS = cat(2, cls_token, actEmbeddingSelect);


%% Attention Blocks
X = dlarray(actEmbeddingSelectwithCLS);

clear AttentionScores;

for layerIdx = 1:numLayers
    fprintf('-- Encoder block %d --\n', layerIdx)
    Block_Input = X;

    % (5) Layernorm1
    layernorm1_weight = eval(sprintf('mae_weight.blocks_%d_norm1_weight', layerIdx-1));
    layernorm1_bias = eval(sprintf('mae_weight.blocks_%d_norm1_bias', layerIdx-1));
    X = layers.normalization(X, layernorm1_weight.', layernorm1_bias.');
    

    % (6) MHA
    init_struct = struct('name', sprintf('encoder_block_%d', layerIdx));
    eval(sprintf("AttentionScores.block_%d = init_struct;", layerIdx)); % init a struct to store attention score

    MHA_weights.attn_c_attn_w_0 = dlarray(eval(sprintf('mae_weight.blocks_%d_attn_qkv_weight', layerIdx-1)));
    mae_attn_qkv_bias = dlarray(eval(sprintf('mae_weight.blocks_%d_attn_qkv_bias', layerIdx-1)));
    MHA_weights.attn_c_attn_b_0 = dlarray(mae_attn_qkv_bias.');
    
    MHA_weights.attn_c_proj_w_0 = dlarray(eval(sprintf('mae_weight.blocks_%d_attn_proj_weight', layerIdx-1))); 
    mae_attn_proj_bias = dlarray(eval(sprintf('mae_weight.blocks_%d_attn_proj_bias', layerIdx-1)));
    MHA_weights.attn_c_proj_b_0 = dlarray(mae_attn_proj_bias.');
    
    [MHAOutput, Score] = layers.attention(X, MHA_weights, numHeads);
    eval(sprintf("AttentionScores.block_%d.score = Score;", layerIdx)); % store attn scores
    
    %% (7) Addition
    MHAOutput = Block_Input + MHAOutput;
    
    
    %% (8) Layernorm2
    layernorm2_weight = eval(sprintf('mae_weight.blocks_%d_norm2_weight', layerIdx-1));
    layernorm2_bias = eval(sprintf('mae_weight.blocks_%d_norm2_bias', layerIdx-1));
    X = layers.normalization(MHAOutput, layernorm2_weight.', layernorm2_bias.');
    
    %% (9) MLP 
    
    
    block_mlp_weights.mlp_c_fc_w_0 = dlarray(eval(sprintf('mae_weight.blocks_%d_mlp_fc1_weight', layerIdx-1)));
    block_mlp_fc1_bias = dlarray(eval(sprintf('mae_weight.blocks_%d_mlp_fc1_bias', layerIdx-1)));
    block_mlp_weights.mlp_c_fc_b_0 = block_mlp_fc1_bias.';

    block_mlp_weights.mlp_c_proj_w_0 = dlarray(eval(sprintf('mae_weight.blocks_%d_mlp_fc2_weight', layerIdx-1)));
    block_mlp_fc2_bias = dlarray(eval(sprintf('mae_weight.blocks_%d_mlp_fc2_bias', layerIdx-1)));
    block_mlp_weights.mlp_c_proj_b_0 = block_mlp_fc2_bias.';
    
    X = layers.multiLayerPerceptron(X, block_mlp_weights);
    
    
    %% (13) Addition
    X = X + MHAOutput;
    
end 

% Encoder Norm
disp('-- Encoder Layernorm --');
Encoder_norm = layers.normalization(X, mae_weight.norm_weight.', mae_weight.norm_bias.');

%% Decoder 

% Decoder embedding projection
DecoderEmb = layers.convolution1d(Encoder_norm, mae_weight.decoder_embed_weight, mae_weight.decoder_embed_bias.');

% Decoder restore mask tokens 
mask_token = dlarray(squeeze(mae_weight.mask_token));
X = single(zeros(size(DecoderEmb, 1), maxNumTokens));

% init as mask emb
for i = 1:maxNumTokens
    X(:, i) = mask_token;
end

%%% change it to input mask
for positionIdx = 1:size(selectedTokenIndices, 2)
    X(:, selectedTokenIndices(positionIdx)) = DecoderEmb(:, positionIdx+1);
end 


% Concate CLS
X = cat(2, DecoderEmb(:, 1), X);

% Positional Embedding 
has_cls_token = true;
DecoderBeforeAttentionBlocks = layers.positionalEmbeddingFunction(X, has_cls_token);

%% Decoder Attention Blocks

X = DecoderBeforeAttentionBlocks;

clear DecoderAttentionScores;

for layerIdx = 1:numLayersDecoder
    fprintf('-- Decoder block %d --\n', layerIdx)
    Block_Input = X;

    % (5) Layernorm1
    layernorm1_weight = eval(sprintf('mae_weight.decoder_blocks_%d_norm1_weight', layerIdx-1));
    layernorm1_bias = eval(sprintf('mae_weight.decoder_blocks_%d_norm1_bias', layerIdx-1));
    X = layers.normalization(X, layernorm1_weight.', layernorm1_bias.');
 

    % (6) MHA
    init_struct = struct('name', sprintf('decoder_block_%d', layerIdx));
    eval(sprintf("DecoderAttentionScores.block_%d = init_struct;", layerIdx)); % init a struct to store attention score

    MHA_weights.attn_c_attn_w_0 = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_attn_qkv_weight', layerIdx-1)));
    mae_attn_qkv_bias = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_attn_qkv_bias', layerIdx-1)));
    MHA_weights.attn_c_attn_b_0 = dlarray(mae_attn_qkv_bias.');
    
    MHA_weights.attn_c_proj_w_0 = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_attn_proj_weight', layerIdx-1))); 
    mae_attn_proj_bias = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_attn_proj_bias', layerIdx-1)));
    MHA_weights.attn_c_proj_b_0 = dlarray(mae_attn_proj_bias.');
    
    [MHAOutput, Score] = layers.attention(X, MHA_weights, numHeads);
    eval(sprintf("DecoderAttentionScores.block_%d.score = Score;", layerIdx)); % store attn scores
    
    % (7) Addition
    MHAOutput = Block_Input + MHAOutput;
    
    % (8) Layernorm2
    layernorm2_weight = eval(sprintf('mae_weight.decoder_blocks_%d_norm2_weight', layerIdx-1));
    layernorm2_bias = eval(sprintf('mae_weight.decoder_blocks_%d_norm2_bias', layerIdx-1));
    X = layers.normalization(MHAOutput, layernorm2_weight.', layernorm2_bias.');
    
    % (9) MLP 
    block_mlp_weights.mlp_c_fc_w_0 = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_mlp_fc1_weight', layerIdx-1)));
    block_mlp_fc1_bias = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_mlp_fc1_bias', layerIdx-1)));
    block_mlp_weights.mlp_c_fc_b_0 = block_mlp_fc1_bias.';

    block_mlp_weights.mlp_c_proj_w_0 = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_mlp_fc2_weight', layerIdx-1)));
    block_mlp_fc2_bias = dlarray(eval(sprintf('mae_weight.decoder_blocks_%d_mlp_fc2_bias', layerIdx-1)));
    block_mlp_weights.mlp_c_proj_b_0 = block_mlp_fc2_bias.';
    
    X = layers.multiLayerPerceptron(X, block_mlp_weights);
    
    
    % (13) Addition
    X = X + MHAOutput;
    
end 

% Encoder Norm
disp('-- Decoder Layernorm --');
Decoder_norm = layers.normalization(X, mae_weight.decoder_norm_weight.', mae_weight.decoder_norm_bias.');

%% Decoder Prediction and map back to pixel domain

Decoder_output = layers.convolution1d(Decoder_norm, mae_weight.decoder_pred_weight, mae_weight.decoder_pred_bias.');  

Decoder_pixels = Decoder_output(:, 2:size(Decoder_output, 2));

%% Map back to image 
Decoder_pixels_reshape = reshape(Decoder_pixels, 3, patchSize, patchSize, gridSize_w, gridSize_h); 
Decoder_pixels_transpose = permute(Decoder_pixels_reshape, [2 4 3 5 1]); % % c(1) q(2) p(3) w(4) h(5) -> q(2) w(4) p(3) h(5) c(1)
Decoder_img = reshape(Decoder_pixels_transpose, patchSize * gridSize_w, patchSize * gridSize_h, 3);
Decoder_img = permute(Decoder_img, [2 1 3]);
Decoder_img = extractdata(Decoder_img);

%% Normalize
imagenetMean = [0.485, 0.456, 0.406];
imagenetStd = [0.229, 0.224, 0.225];

%%%% normalize by ImageNet mean and std
% Initialize a container for the normalized image
denormalizedDecodingImage = zeros(size(Decoder_img));
for channel = 1:3
    % Extract one channel
    channelData = Decoder_img(:,:,channel);

    % Normalize the current channel
    channelMean = imagenetMean(channel);
    channelStd = imagenetStd(channel);
    channel_img = channelData .* channelStd + channelMean;
    denormalizedDecodingImage(:,:,channel) = channel_img;
end


%% show original, masked, reconstructed, and togethered
% imshow(denormalizedDecodingImage, 'InitialMagnification', 'fit')

figure('Units', 'pixels', 'Position', [100, 100, 800, 200]); % Creates a new figure window

% Display the first image
subplot(1, 4, 1); % This means 1 row, 4 columns, and this is the 1st position
imshow(img, 'InitialMagnification', 'fit');
title('Original'); % Adds a title above the image

% Display the second image

image_vis = zeros(size(img));
for positionIdx = 1:size(selectedTokenIndices, 2)
    position = selectedTokenIndices(positionIdx);
    row = floor((position-1) / gridSize_h)+1;
    col = mod(position-1, gridSize_w)+1;
    image_vis((row-1)*patchSize+1:row*patchSize, (col-1)*patchSize+1:col*patchSize, :) = 1;
end

subplot(1, 4, 2); % This means 1 row, 4 columns, and this is the 2nd position
imshow(img .* image_vis, 'InitialMagnification', 'fit');

title('Mask');

% Display the third image
subplot(1, 4, 3); % This means 1 row, 4 columns, and this is the 3rd position
imshow(denormalizedDecodingImage, 'InitialMagnification', 'fit');
title('MAE Output');

% Display the fourth image

DecodingPlusVisible = img .* image_vis + denormalizedDecodingImage .* (1 - image_vis);
subplot(1, 4, 4); % This means 1 row, 4 columns, and this is the 4th position
imshow(DecodingPlusVisible, 'InitialMagnification', 'fit');
title('MAE Output + Visible');


