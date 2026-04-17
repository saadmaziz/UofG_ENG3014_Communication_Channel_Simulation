% Define the SNR values (x-axis)
SNR = [2, 4, 6, 8, 12, 18, 24, 32];

% Define the data for each metric
% Columns correspond to speeds: 5.5km/h, 30km/h, 50km/h

% 1. Average Bit Error Rate (%)
BER_data = [
    66.62, 65.63, 65.82;
    11.97, 11.93, 12.28;
    2.75,  2.74,  2.87;
    2.39,  2.27,  2.23;
    1.04,  1.06,  1.10;
    0.0008,0.0004,0.0004;
    0,     0,     0;
    0,     0,     0
];

% 2. Average Transmission Time (ms)
Time_data = [
    2736.8, 2721.6, 2706.4;
    865.6,  879.1,  892.7;
    364.7,  364.1,  359.9;
    290.6,  292.4,  293.3;
    168.5,  163.9,  166.5;
    122.8,  122.8,  122.8;
    122.8,  122.8,  122.8;
    122.8,  122.8,  122.8
];

% 3. Average Transmission Efficiency (%)
Efficiency_data = [
    7.33,  7.37,  7.41;
    24.15, 23.72, 23.43;
    50.53, 50.79, 51.44;
    57.68, 57.29, 57.14;
    87.34, 89.05, 88.56;
    100,   100,   100;
    100,   100,   100;
    100,   100,   100
];

% Define legend labels
speeds = {'5.5 km/h', '30 km/h', '50 km/h'};
markers = {'-o', '-s', '-^'}; % Circle, Square, Triangle markers
colors = {'b', 'r', 'g'};     % Blue, Red, Green

% Create a new figure with a specific size
figure('Position', [100, 100, 1200, 400]);

% --- Subplot 1: Bit Error Rate ---
subplot(1, 3, 1);
hold on;
for i = 1:3
    plot(SNR, BER_data(:, i), markers{i}, 'Color', colors{i}, 'LineWidth', 1.5);
end
hold off;
title('Average Bit Error Rate');
xlabel('SNR (dB)');
ylabel('BER (%)');
grid on;
legend(speeds);

% --- Subplot 2: Transmission Time ---
subplot(1, 3, 2);
hold on;
for i = 1:3
    plot(SNR, Time_data(:, i), markers{i}, 'Color', colors{i}, 'LineWidth', 1.5);
end
hold off;
title('Average Transmission Time');
xlabel('SNR (dB)');
ylabel('Time (ms)');
grid on;
legend(speeds);

% --- Subplot 3: Transmission Efficiency ---
subplot(1, 3, 3);
hold on;
for i = 1:3
    plot(SNR, Efficiency_data(:, i), markers{i}, 'Color', colors{i}, 'LineWidth', 1.5);
end
hold off;
title('Average Transmission Efficiency');
xlabel('SNR (dB)');
ylabel('Efficiency (%)');
grid on;
legend(speeds, 'Location', 'SouthEast');