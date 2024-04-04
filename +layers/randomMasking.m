function [X_select, selectedTokenIndices] = randomMasking(X, img, patchSize, randomMaskingRatio)
    
    inputSize = size(img);
    gridSize_h = inputSize(1) / patchSize;
    gridSize_w = inputSize(2) / patchSize;
    
    % if (randomMaskingRatio > 0)
        selectedCells = layers.selectGridsOnImage(img, patchSize, randomMaskingRatio);
        
        selectedCells = selectedCells.'; % transpose due to matlab col first encoding for matrix
        selectedCellsFlat = selectedCells(:);
        selectedTokenIndices = [];
        
        for i = 1:size(selectedCellsFlat, 1)
            if (selectedCellsFlat(i) == 0)
                selectedTokenIndices = [selectedTokenIndices i];
            end
        end 
        
        X_select = X(:, selectedTokenIndices);
    % else
    %     X_select = X;
    %     selectedTokenIndices = 1:(gridSize_h * gridSize_w);
end