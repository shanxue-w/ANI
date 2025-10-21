% 假设数据 a 已经是 1001×4 的轨迹矩阵
theta1 = a(:,1);
theta2 = a(:,3);

% 摆长
l1 = 1;
l2 = 1;

% 时间步数
n = length(theta1);

% 计算每一帧的位置
x1 = l1 * sin(theta1);
y1 = -l1 * cos(theta1);
x2 = x1 + l2 * sin(theta2);
y2 = y1 - l2 * cos(theta2);

% GIF 文件名
gif_filename = 'double_pendulum_fixed_axis.gif';

% 创建图形
fig = figure('Units', 'pixels', 'Position', [100, 100, 500, 500]); % ✅ 确保窗口尺寸固定
ax = axes('Parent', fig);
set(ax, 'XLim', [-2.2, 2.2], 'YLim', [-2.2, 2.2], 'DataAspectRatio', [1 1 1]);
axis manual; % ✅ 禁止自动缩放
grid on;
xlabel('x');
ylabel('y');
title('Double Pendulum Animation (Fixed Axis)');

for i = 1:n
    cla(ax); % 清除旧图内容但保留轴属性

    % 绘制连杆
    line([0, x1(i)], [0, y1(i)], 'Color', 'b', 'LineWidth', 2); hold on;
    line([x1(i), x2(i)], [y1(i), y2(i)], 'Color', 'r', 'LineWidth', 2);

    % 绘制质点
    plot(x1(i), y1(i), 'bo', 'MarkerSize', 6, 'MarkerFaceColor', 'b');
    plot(x2(i), y2(i), 'ro', 'MarkerSize', 6, 'MarkerFaceColor', 'r');

    % 添加时间标签
    text(-2, 2, sprintf('Frame: %d / %d', i, n));

    drawnow;

    % 获取帧
    frame = getframe(fig);  % ✅ 只截固定大小的窗口
    im = frame2im(frame);
    [imind, cm] = rgb2ind(im, 256);

    if i == 1
        imwrite(imind, cm, gif_filename, 'gif', 'Loopcount', inf, 'DelayTime', 0.01);
    else
        imwrite(imind, cm, gif_filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.01);
    end
end

disp(['✅ GIF saved as ', gif_filename]);
