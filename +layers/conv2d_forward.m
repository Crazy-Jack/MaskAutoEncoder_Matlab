function [out] = conv2d_forward(x, w, bias, stride, pad)
    % Implement convolution2d in forLoop fashion for maximial understanding
    % This implementation doesn't contain backward

    % Argument:
    %       - x: [B Ci Hi Wi] 
    %               B is batch size, Ci is input channel numbers, Hi and Wi
    %               are input spatial sizes.
    %       - w: [Co Ci Kh Kw] 
    %               Co is output channel numbers, Kh and Kw are
    %               kernel spatial sizes.
    %       - bias: [1 1024]
    %               Bias of the convolution operation
    %       - stride: int   
    %               the stride of the convolving kernel. 
    %       - pad: int
    %               the padding of the image
    
    % pad
    
    % 
    [b, ci, hi, wi] = size(x);
    [co, ci, hk, wk] = size(w);
    ho = floor(1 + (hi - hk) / stride);
    wo = floor(1 + (wi - wk) / stride);
    out = zeros([b, co, ho, wo]); % B Co Ho Wo
    
    x = reshape(x, [b, 1, ci, hi, wi]); % B 1  Ci Hi Wi
    w = reshape(w, [1, co, ci, hk, wk]); % 1 Co Ci Hk Wk

    for i = 1:ho
        for j = 1:wo
            x_windows = x(:, :, :, (i-1) * stride+1:(i-1) * stride + hk, (j-1) * stride+1: (j-1) * stride + wk); % B 1 Ci Hk Wk
            % size(x_windows)
            % size(w)
            out(:, :, i, j) = sum(x_windows .* w, [3 4 5]); % B Co
        end
    end

    if (bias)
        bias = reshape(bias(:), [1 co 1 1]);
        out = out + bias;
    end
end