function u = poison(xl,xr,yb,yt,M,N)
% 泊松方程的有限差分解法
% 
% 输入: 矩形[xl, xr]X[yb, yt], 步长M, N
% 输出: 矩阵u, 其元素为函数u(x, y)的近似解

f = @(x,y) 0;   % 定义输入函数数据
g1 = @(x) log(1 + x.^2);    % 定义边界条件
g2 = @(x) log(4 + x.^2);
g3 = @(y) 2 * log(y);
g4 = @(y) log(1 + y.^2);

m = M + 1; n = N + 1; mn = m * n;
h = (xr - xl) / M; h2 = h^2;
k = (yt - yb) / N; k2 = k^2;

% 设置网格值
x = xl : h : xr;
y = yb : k : yt;

A = zeros(mn, mn);  % 初始化系数矩阵
b = zeros(mn, 1);   % 初始化b

% 对内部点
for i = 2 : m-1
    for j = 2 : n-1
        A(i + (j-1)*m, i - 1 + (j-1)*m) = 1 / h2;
        A(i + (j-1)*m, i + 1 + (j-1)*m) = 1 / h2;
        A(i + (j-1)*m, i + (j-1)*m) = -2 / h2 - 2 / k2;
        A(i + (j-1)*m, i + j*m) = 1 / k2;
        A(i + (j-1)*m, i + (j-2)*m) = 1 / k2;
    end
end

% 底部和顶部的边界点
for i = 1 : m
   j = 1; A(i + (j-1)*m, i + (j-1)*m) = 1; b(i + (j-1)*m) = g1(x(i));
   j = n; A(i + (j-1)*m, i + (j-1)*m) = 1; b(i + (j-1)*m) = g2(x(i));
end

% 左右两侧的边界点
for j = 2 : n-1
   i = 1; A(i + (j-1)*m, i + (j-1)*m) = 1; b(i + (j-1)*m) = g3(y(j));
   i = m; A(i + (j-1)*m, i + (j-1)*m) = 1; b(i + (j-1)*m) = g4(y(j));
end

v = A\b;    % 以v标记求解
u = reshape(v(1:mn), m, n);     % 将v转化成u
% 画出近似解
mesh(x, y, u')

end

