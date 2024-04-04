function [activationAfterPosEmb] = positionalEmbeddingFunction(X, cls_token)
    
    [numFeature, gridSizeSquare] = size(X);
    if (cls_token)
        gridSizeSquare = gridSizeSquare - 1;
    end

    gridSize_h = 14; % the resolution the model is trained on
    gridSize_w = 14;
    pos_embed = layers.get_2d_sincos_pos_embed(numFeature, gridSize_h, cls_token); % [gridSize_h * gridSize_w, numFeature]
    pos_embed = pos_embed.'; % [numFeature, gridSize_h * gridSize_w]
    if (gridSizeSquare ~= gridSize_h * gridSize_w)
        
        % not the training size so it will require interpolation, assume
        % square images
        w_input = round(sqrt(gridSizeSquare));
        h_input = round(sqrt(gridSizeSquare));
        pos_embed = reshape(pos_embed, numFeature, gridSize_h, gridSize_w);
        

        % Interpolate
        [Xq, Yq] = meshgrid(linspace(1, gridSize_w, w_input), linspace(1, gridSize_h, h_input));
        interpolated = zeros(numFeature, w_input, h_input);

        for i = 1:numFeature
            interpolated(i, :, :) = interp2(squeeze(pos_embed(i, :, :)), Xq, Yq, 'cubic');
        end

        % reshape back
        pos_embed = reshape(interpolated, [numFeature, w_input * h_input]);
        
    end
    
    activationAfterPosEmb = X + pos_embed;

end