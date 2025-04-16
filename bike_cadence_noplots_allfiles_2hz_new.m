%% Script to process IMU data attached to the crank arm of a bike.
% Goal: detect pedalling activity and compute pedalling cadence (turn per minute)

% Created by J.I.M. Parmentier
% February 2024
% University of Twente
% contact: j.i.m.parmentier@utwente.nl

%% Select folder containing .mat files
folderPath = uigetdir('Select Folder Containing .mat Files');

if folderPath == 0
    error('No folder selected. Exiting...');
end

% Get all .mat files in the selected folder
matFiles = dir(fullfile(folderPath, '*.mat'));

if isempty(matFiles)
    error('No .mat files found in the selected folder.');
end

%% Process each .mat file
for k = 1:length(matFiles)
    % Load the current file
    fileToLoad = fullfile(folderPath, matFiles(k).name);
    fprintf('Processing file: %s\n', matFiles(k).name);
    load(fileToLoad);

    % Extract the filename without extension
    [~, fileNameNoExt, ~] = fileparts(matFiles(k).name);

    % Extract the node number from the filename (assumes the format includes '_###-')
    nodeNumber = regexp(matFiles(k).name, '_\d{3,}-', 'match');
    if isempty(nodeNumber)
        warning('Node number could not be extracted from file: %s. Skipping...', matFiles(k).name);
        continue;
    end
    nodeNumber = strrep(nodeNumber{1}, '_', ''); % Remove underscores
    nodeNumber = strrep(nodeNumber, '-', '');   % Remove dashes

    % Construct the node field name (e.g., 'node_556')
    PedalNode = ['node_', nodeNumber];

    % Verify if the inferred node exists
    if ~isfield(inertia, PedalNode)
        warning('The inferred node "%s" does not exist in file: %s. Skipping...', PedalNode, matFiles(k).name);
        continue;
    end

    % Extract data
    data = inertia.(PedalNode).data;
    accel = inertia.(PedalNode).fields.accel;
    gyro = inertia.(PedalNode).fields.gyro;
    time = inertia.(PedalNode).data(:, 1);

    % Reset time
    time = time - time(1);

    %% Prepare the data for activity detection and cadence computation
    fs = inertia.(PedalNode).samplingRate;
    [b, a] = butter(4, 8/(fs/2), 'low');

    data_filt.accel = filtfilt(b, a, data(:, accel));
    data_filt.gyro = filtfilt(b, a, data(:, gyro));

    % Compute time in minutes
    time_mn = time / 60;

    % Set the window size to compute the cadence on a moving window
    window_size = 4 * fs; % every 4 seconds
    window_for_silence = window_size / 5; % smaller window for silence detection
    threshold = 10;
    silence_activity = silence_detect_bike(data_filt.accel(:, 1), window_for_silence, threshold);

    % Initialise variables for cadence computation
    cadence_counter = [];
    var_counter = [];

    % Define the stride for 2 Hz output sampling rate
    stride = fs * 0.5; % 0.5 seconds in samples
    
    for i = window_size/2 + 1 : stride : length(data_filt.accel(:, 1)) - window_size
        datatemp = data_filt.accel(i - window_size/2 : i + window_size/2, 1);
        silence_temp = silence_activity(i - window_size/2 : i + window_size/2);
        if mean(silence_temp) > 0.9 % The bike is moving
            [pks, lcs] = findpeaks(datatemp, 'MinPeakProminence', 4, 'MinPeakHeight', 2);
            var_counter = [var_counter; var(datatemp)];
            if length(lcs) >= 2
                cd = 60 / (mean(diff(lcs)) / fs);
                cadence_counter = [cadence_counter; ...
                    median(i - window_size/2 : i + window_size/2), cd];
            else
                cadence_counter = [cadence_counter; ...
                    i, NaN];
            end
        else
            cadence_counter = [cadence_counter; ...
                i, NaN];
        end
    end

    % Save results to CSV
    Cadence_CSV = [time(cadence_counter(:, 1)), time_mn(cadence_counter(:, 1)), cadence_counter(:, 2)];
    Cadence_CSV = array2table(Cadence_CSV, "VariableNames", {'Time_s', 'Time_mn', 'Cadence_tpmn'});

    % Include the file name in the CSV output
    outputFileName = sprintf('%s-%s-cadence-2hz.csv', fileNameNoExt, PedalNode);
    writetable(Cadence_CSV, fullfile(folderPath, outputFileName));

    fprintf('Cadence data saved to: %s\n', outputFileName);
end

fprintf('Processing complete for all files in folder: %s\n', folderPath);
