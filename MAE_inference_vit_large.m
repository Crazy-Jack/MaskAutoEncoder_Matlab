
%% Hyper Parameters
inputSize = [384 384 3];
numFeature = 1024;
mlpRatio = 4;
numFeatureMLPHidden = mlpRatio * numFeature;
patchSize = 16;
maxNumTokens = (inputSize(1) / patchSize) * (inputSize(2) / patchSize);
gridSize_h = inputSize(1) / patchSize;
gridSize_w = inputSize(2) / patchSize;
numLayers = 24;
numHeads = 16;

%% I/O -- Load image
% img = imread('./img/fox.jpg'); % Replace with the path to your image
% img = imread('./img/continuity.png');
% img = imread('./img/similarity1.png');
img = imread('./img/Ktriangle.png');
img = imresize(img, [inputSize(1), inputSize(2)]); % Resize the image to fit the input layer
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

%% Load Pytorch MAE weights and testCase (numpy version) 
path_to_weights = "./weights/mae_visualize_vit_large_ganloss_weight.mat";
mae_weight = load(path_to_weights);

path_to_testCase1 = "./testCase/MAE_vis_gan_activation_fox.mat";
mae_testCase1_activation = load(path_to_testCase1);

%% Network Architecture - (1) - PatchEmbedding

inputImage_transpose = permute(inputImage, [4, 3, 1, 2]);
patchEmbeddingActivation = layers.patchEmbeddingFunction(inputImage_transpose, mae_weight.patch_embed_proj_weight, ...
    mae_weight.patch_embed_proj_bias, patchSize); 

%% Network Architecture - (2) - PositionalEmbedding interpolation
has_cls_token = false;
activationAfterPosEmb = layers.positionalEmbeddingFunction(patchEmbeddingActivation, has_cls_token);



%% Operation - (3) - Random Masking
% No random masking during inference
actEmbeddingSelect = activationAfterPosEmb;

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


%% Visulaize Attention
head_id = 10;
[selectedCells] = visualizePointAttention(img, patchSize, AttentionScores.block_24.score, head_id);
