% 假设文件名为 'magnetometer_data.csv'

filename = 'B_buffer.csv';

% 使用 readtable 加载 CSV 数据
data = readtable(filename);


% 提取磁力计的 x, y, z 数据
M = table2array(data(:, 1:3));  % 从 table 中提取前三列，并转换为 double 类型

% 2. 使用 magcal 函数进行磁力计数据校准
[A, b, normError] = magcal(M);

% 输出校准误差
fprintf('校准误差 (RMS error): %f\n', normError);

% 3. 校准数据
% 校准后的磁场数据 = (M - b) * A
M_calibrated = (M - b) * A;

% 4. 可视化校准前后的数据
figure;
hold on;

% 校准前的磁场数据（蓝色）
scatter3(M(:,1), M(:,2), M(:,3), 20, 'b', 'filled');

% 校准后的磁场数据（红色）
scatter3(M_calibrated(:,1), M_calibrated(:,2), M_calibrated(:,3), 20, 'r', 'filled');

% 图形设置
title('磁场数据校准前后对比');
xlabel('X');
ylabel('Y');
zlabel('Z');
legend({'原始数据', '校准后数据'}, 'Location', 'best');
axis equal; % 保证坐标轴比例一致
grid on; % 打开网格

% 设置初始三维视角
view(3); % 设置为三维视角模式
rotate3d on; % 允许鼠标交互旋转

% 输出完成信息
fprintf('校准后的数据已绘制完成。\n');
hold off;

% subplot(1, 2, 1);
% scatter3(M(:,1), M(:,2), M(:,3), 10, 'filled');
% title('原始磁场数据');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% axis equal;
% 
% subplot(1, 2, 2);
% scatter3(M_calibrated(:,1), M_calibrated(:,2), M_calibrated(:,3), 10, 'filled');
% title('校准后的磁场数据');
% xlabel('X');
% ylabel('Y');
% zlabel('Z');
% axis equal;
% 
% % 5. 保存校准后的数据到新文件
% writematrix(M_calibrated, 'calibrated_data.csv');
% 
% % 输出完成信息
% fprintf('校准后的数据已保存为 "calibrated_data.csv"\n');
