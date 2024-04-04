%% Network Architecture - (2) - PositionalEmbedding
% in python
% def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
%     """
%     grid_size: int of the grid height and width
%     return:
%     pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
%     """
%     grid_h = np.arange(grid_size, dtype=np.float32)
%     grid_w = np.arange(grid_size, dtype=np.float32)
%     grid = np.meshgrid(grid_w, grid_h)  # here w goes first
%     grid = np.stack(grid, axis=0)
% 
%     grid = grid.reshape([2, 1, grid_size, grid_size])
%     pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
%     if cls_token:
%         pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
%     return pos_embed
%% 
function pos_embed = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token)
    if nargin < 3
        cls_token = false;
    end
    
    % Create grid
    [grid_w, grid_h] = meshgrid(0:grid_size-1, 0:grid_size-1);
    grid = cat(3, grid_w, grid_h); % Concatenate along the third dimension
    
    % Reshape grid
    grid = reshape(grid, grid_size, grid_size, 1, 2);
    
    % Get positional embedding from grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
    
    % Add class token if necessary
    if cls_token
        pos_embed = [zeros(1, embed_dim); pos_embed];
    end
end


%%
% def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
%    assert embed_dim % 2 == 0
%
%    # use half of dimensions to encode grid_h
%    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
%    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
%
%    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
%    return emb




function emb = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    assert(mod(embed_dim, 2) == 0, 'Embed dimension must be even');
    
    % Encode height and width separately
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, squeeze(grid(:,:,:, 1)));
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim / 2, squeeze(grid(:,:,:, 2)));
    % Concatenate
    emb = [emb_w, emb_h];
end

%%
%def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
%    """
%    embed_dim: output dimension for each position
%    pos: a list of positions to be encoded: size (M,)
%    out: (M, D)
%    """
%    assert embed_dim % 2 == 0
%    omega = np.arange(embed_dim // 2, dtype=np.float64)
%    omega /= embed_dim / 2.
%    omega = 1. / 10000**omega  # (D/2,)
%
%    pos = pos.reshape(-1)  # (M,)
%    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
%
%    emb_sin = np.sin(out) # (M, D/2)
%    emb_cos = np.cos(out) # (M, D/2)
%
%    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
%    return emb



function emb = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    assert(mod(embed_dim, 2) == 0, 'Embed dimension must be even');
    
    omega = (0:embed_dim/2-1)';
    omega = single(1 ./ (10000 .^ (omega / (embed_dim / 2))));
    
    pos = pos(:); % Ensure pos is a column vector
    out = pos * omega'; % Outer product
    
    % Sin and Cos
    emb_sin = sin(out);
    emb_cos = cos(out);
    
    % Concatenate
    emb = [emb_sin, emb_cos];
end
