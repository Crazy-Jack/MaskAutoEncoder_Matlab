function [A, Score] = attention(X, weights, numHeads)
% attention   Full Multi-head Attention
%
%   [A, present] = attention(X, past, weights, hyperParameters) computes a
%   multi-head attention block on X as outlined in Section 3.2.2 and Figure
%   2 in [1]. See below for details of inputs and outputs.
%
%   Inputs:
%       X               - A (numFeatures*numHeads)-by-numInputSubwords-by-numObs
%                         input array.
%       weights         - The weights for the full multi-head attention
%                         block stored in a struct. This includes:
%                           - attn_c_attn_w_0: A weight matrix for the
%                             first fully connected layer.
%                             A numOutputFeatures-by-numInputFeatures weight matrix.
%                           - attn_c_attn_b_0: A bias vector for the first
%                             fully connected layer.
%                             A numOutputFeatures-by-1 bias vector.
%                           - attn_c_proj_w_0: A weight matrix for the
%                             final fully connected layer.
%                           - attn_c_proj_b_0: A bias vector for the final
%                             fully connected layer.
%       numHeads        - The number of attention heads. This is a
%                         hyper-parameter.
%
%   Outputs:
%       A               - A (numFeatures*numHeads)-by-numInputSubwords-by-numObs
%                         output array.
%       
%
%   
%   References:
%
%   [1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion
%       Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, "Attention
%       Is All You Need", https://arxiv.org/abs/1706.03762
arguments
    X
    weights
    numHeads
end

% Use a fully connected layer to generate queries, keys and values from the
% input.
C = layers.convolution1d( X, ...
    weights.attn_c_attn_w_0, ...
    weights.attn_c_attn_b_0 );

% Split the results into Q (Query), K (Keys) and V (Values).
splitSize = size(C,1)/3;
Q = C(1:splitSize,:,:);
K = C((splitSize+1):(2*splitSize),:,:);
V = C((2*splitSize+1):(3*splitSize),:,:);

% Split heads
Q = iSplitHeads(Q, splitSize, numHeads);
K = iSplitHeads(K, splitSize, numHeads);
V = iSplitHeads(V, splitSize, numHeads);


[A, Score] = layers.multiheadAttention(Q,K,V);

A = iMergeHeads(A);

A = layers.convolution1d( A, ...
    weights.attn_c_proj_w_0, ...
    weights.attn_c_proj_b_0 );
end

function Z = iSplitHeads(X, splitSize, numHeads)
% We permute the data to put the dimension for the heads last, so that we
% can use batched matrix multiplication to compute attention for all of the
% heads at once.
%
% X     - A (numFeatures*numHeads)-by-numSubwords-by-numObs array.
% Z     - A numFeatures-by-numSubwords-by-numHeads-by-numObs array.
X = reshape(X, splitSize/numHeads, numHeads, [], size(X,3));
Z = permute(X,[1 3 2 4]);
end

function Z = iMergeHeads(X)
% X     - A numFeatures-by-numSubwords-by-numHeads-by-numObs array.
% Z     - A (numFeatures*numHeads)-by-numSubwords-by-numObs array.
X = permute(X, [1 3 2 4]);
Z = reshape(X, size(X,1)*size(X,2), [], size(X,4));
end