function [selectedCells] = selectGridsOnImage(img, patchSize, mask_ratio)
%   presetMask = [grid_size, grid_size]
%
%
%
    hFig = figure; % Create a new figure
    
    % Create axes in the figure for the image
    ax = axes('Parent', hFig, 'Position', [0.1, 0.1, 0.8, 0.8]); % Adjust as needed
    imshow(img, 'InitialMagnification', 'fit', 'Parent', ax); % Display the image in those axes


    hold on; % Overlay grid on image

    [imgHeight, imgWidth, ~] = size(img);
    numRows = imgHeight / patchSize;
    numCols = imgWidth / patchSize;
    cellWidth = patchSize;
    cellHeight = patchSize;

    % Display for count of selected indices
    textWidth = 0.4; % Width of the text box in normalized units, adjust as needed
    textHeight = 0.05; % Height of the text box, adjust as needed
    textLeft = (1 - textWidth) / 2; % Center horizontally
    textBottom = 0.95 - textHeight; % Position from the bottom, leaving some margin
    
    countDisplay = uicontrol('Style', 'text', 'String', 'Selected: 0', ...
                             'BackgroundColor', hFig.Color, 'FontSize', 12, ...
                             'Units', 'normalized', 'Position', [textLeft, textBottom, textWidth, textHeight], ...
                             'HorizontalAlignment', 'center');
    % Draw the grid
    for i = 1:(numRows-1)
        line([1, imgWidth], [i * cellHeight, i * cellHeight], 'Color', 'black'); % Horizontal lines
        line([i * cellWidth, i * cellWidth], [1, imgHeight], 'Color', 'black'); % Vertical lines
    end
    hold off;

    overlays = gobjects(numRows, numCols); % Graphics object array for managing overlays

    % randomly generate mask
    selectedCells = zeros(numRows, numCols);
    mask_len = round(mask_ratio * numRows * numCols);
    indices = randperm(numRows * numCols, mask_len);
    for index = indices
        y = ceil(index / numCols);
        x = mod(index, numCols) + 1;
        if selectedCells(y, x) == 0
            selectedCells(y, x) = 1; % Mark as selected
            rect = rectangle('Position', [(x-1)*cellWidth, (y-1)*cellHeight, cellWidth, cellHeight], 'FaceColor', [0.8, 0.8, 0.8, 1], 'EdgeColor', 'white');
            overlays(y, x) = rect; % Store the overlay object
        end
    end
  
    % Update the display count
    numSelected = sum(selectedCells(:)); % Count how many cells are selected
    countDisplay.String = [sprintf('Selected Patches: %d (%d %%)', numSelected, round(numSelected / (numRows * numCols) * 100))]; % Update the text object
    
    % Set up the callback
    set(gcf, 'WindowButtonDownFcn', @clickCallback, 'CloseRequestFcn', @closeFigCallback);
    uiwait(gcf);
    % Nested clickCallback function
    function clickCallback(~, ~)
        pt = get(ax, 'CurrentPoint'); % Get the position of the click
        x = ceil(pt(1,1) / cellWidth); % Calculate the grid position
        y = ceil(pt(1,2) / cellHeight);
        % Toggle selection state
        if selectedCells(y, x) == 0
            selectedCells(y, x) = 1; % Mark as selected
            rect = rectangle('Position', [(x-1)*cellWidth, (y-1)*cellHeight, cellWidth, cellHeight], 'FaceColor', [0.8, 0.8, 0.8, 1], 'EdgeColor', 'none');
            overlays(y, x) = rect; % Store the overlay object
        else
            selectedCells(y, x) = 0; % Mark as not selected
            delete(overlays(y, x)); % Remove the overlay
            overlays(y, x) = gobjects(1); % Reset the overlay object placeholder
        end

        % Update the display count
        numSelected = sum(selectedCells(:)); % Count how many cells are selected
        countDisplay.String = [sprintf('Selected Patches: %d (%d %%)', numSelected, round(numSelected / (numRows * numCols) * 100))]; % Update the text object
        % save to base workspace
        assignin('base', 'selectedCells', selectedCells); % Save to base workspace
    end

    function closeFigCallback(src, ~)
        assignin('base', 'selectedCells', selectedCells); % Save to base workspace
        uiresume(src); % Resume execution
        delete(src); % Close the figure
    end
end

