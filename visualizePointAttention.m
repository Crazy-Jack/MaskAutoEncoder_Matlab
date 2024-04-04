function [selectedCells] = visualizePointAttention(img, patchSize, attnMap, head_id)

    % visualize point attention

    hFig = figure; % Create a new figure
    
    % Create axes in the figure for the image
    ax1 = axes('Parent', hFig, 'Position', [0.1, 0.1, 0.4, 0.8]); % Adjust as needed
    imshow(img, 'InitialMagnification', 'fit', 'Parent', ax1); % Display the image in those axes

    hold(ax1, 'on'); % Overlay grid on the first image

    [imgHeight, imgWidth, ~] = size(img);
    numRows = imgHeight / patchSize;
    numCols = imgWidth / patchSize;
    cellWidth = patchSize;
    cellHeight = patchSize;

    % Display for count of selected indices
    textWidth = 0.4; % Width of the text box in normalized units, adjust as needed
    textHeight = 0.05; % Height of the text box, adjust as needed
    textLeft = (1 - textWidth) / 2; % Center horizontally
    textBottom = 0.85 - textHeight; % Position from the bottom, leaving some margin
    
    countDisplay = uicontrol('Style', 'text', 'String', 'Selected: 0', ...
                             'BackgroundColor', hFig.Color, 'FontSize', 12, ...
                             'Units', 'normalized', 'Position', [textLeft, textBottom, textWidth, textHeight], ...
                             'HorizontalAlignment', 'center');
    % Draw the grid
    for i = 1:(numRows-1)
        line([1, imgWidth], [i * cellHeight, i * cellHeight], 'Color', 'black'); % Horizontal lines
        line([i * cellWidth, i * cellWidth], [1, imgHeight], 'Color', 'black'); % Vertical lines
    end
    hold(ax1, 'off');
    
    overlays = gobjects(numRows, numCols); % Graphics object array for managing overlays

    selectedCells = zeros(numRows, numCols);
    % Update the display count
    numSelected = sum(selectedCells(:)); % Count how many cells are selected
    countDisplay.String = ['Attention Matrix Disp']; % Update the text object
    
    % Create axes for the second image
    ax2 = axes('Parent', hFig, 'Position', [0.55, 0.1, 0.4, 0.8]); % Adjust as needed for the second image
    placeholderImg = zeros(size(img,1), size(img,2), 'double'); % Placeholder image, adjust as needed
    imshow(placeholderImg, 'Parent', ax2); % Initial display of the second image
    
    
    % Set up the callback
    set(gcf, 'WindowButtonDownFcn', @clickCallback, 'CloseRequestFcn', @closeFigCallback);

    % attnMap
    attnMapArray = extractdata(attnMap); % L x L x heads
    
    % Nested clickCallback function
    function clickCallback(~, ~)
        pt = get(ax1, 'CurrentPoint'); % Get the position of the click
        x = ceil(pt(1,1) / cellWidth); % Calculate the grid position
        y = ceil(pt(1,2) / cellHeight);
        
        % Toggle selection state
        if (y <= numRows && y >= 1 && x <= numCols && x >= 1)


            % Deselect any previously selected cell
            previouslySelected = find(selectedCells);
            for idx = 1:length(previouslySelected)
                [prevY, prevX] = ind2sub(size(selectedCells), previouslySelected(idx));
                selectedCells(prevY, prevX) = 0; % Deselect cell
                delete(overlays(prevY, prevX)); % Remove overlay
                overlays(prevY, prevX) = gobjects(1); % Reset overlay object
            end


    
            selectedCells(y, x) = 1; % Mark as selected
            rect = rectangle('Position', [(x-1)*cellWidth, (y-1)*cellHeight, cellWidth, cellHeight], 'FaceColor', [0.8, 0.8, 0.8, 1], 'EdgeColor', 'none');
            overlays(y, x) = rect; % Store the overlay object
    
            % Update the second image based on the selected grid point
            [attnScoreImage] = vis_multihead_attn_img(attnMapArray, head_id, y, x);
            % Interpolate
            attnScoreImageInterp = zeros(size(img, 1), size(img, 2), 3);
            for row = 1:size(attnScoreImage, 1)
                for col = 1:size(attnScoreImage, 2)
                    attnScoreImageInterp((row-1)*patchSize+1:row*patchSize, (col-1)*patchSize+1:col*patchSize, :) = ones(patchSize, patchSize, 3) * attnScoreImage(row, col);
                end
            end
    
            attnScoreImageInterp = contrast_norm(attnScoreImageInterp);

            % % mark the attn position
            % red dot
            attnScoreImageInterp(((y-1)*patchSize+1:y*patchSize), ((x-1)*patchSize+1:x*patchSize), 1) = ones(patchSize, patchSize);
            attnScoreImageInterp(((y-1)*patchSize+1:y*patchSize), ((x-1)*patchSize+1:x*patchSize), 2) = zeros(patchSize, patchSize);
            attnScoreImageInterp(((y-1)*patchSize+1:y*patchSize), ((x-1)*patchSize+1:x*patchSize), 3) = zeros(patchSize, patchSize);
            % blend with images
            attnScoreImageInterp = attnScoreImageInterp * 0.7 + img * 0.3;
        
            
            imshow(attnScoreImageInterp, 'Parent', ax2); % Display the selected image
                
        end
        % Update the display count
        numSelected = sum(selectedCells(:)); % Count how many cells are selected
        countDisplay.String = [sprintf('Row:%d, Col:%d', y, x)]; % Update the text object
        % save to base workspace
        assignin('base', 'selectedCells', selectedCells); % Save to base workspace
    end

    function closeFigCallback(src, ~)
        assignin('base', 'selectedCells', selectedCells); % Save to base workspace
        uiresume(src); % Resume execution
        delete(src); % Close the figure
    end
end


function [disp_attn] = vis_multihead_attn_img(attnMapArray, head_id, pixel_y, pixel_x)
    [H, H, num_heads] = size(attnMapArray);
    img_size = round(sqrt(H-1));
    assert(head_id <= num_heads);
    
    attn_idx = (pixel_y-1) * img_size + pixel_x + 1; % the first one is cls token
    head_attn = attnMapArray(2:H, attn_idx, head_id);
    disp_attn = reshape(head_attn, img_size, img_size).'; % assume square images, transpose because the col first indexing system
                                                            % but the code
                                                            % is written in
                                                            % row first
    disp_attn = contrast_norm(disp_attn);
    
end 


function [normalized_img] = contrast_norm(img)
    max_img = max(img, [], 'all');
    min_img = min(img, [], 'all');
    normalized_img = (img - min_img) / (max_img - min_img);
end 