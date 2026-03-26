#!/usr/bin/env python3
# teenygrad 最简示例：从基础操作到自动微分
import numpy as np

from teenygrad.tensor import Tensor

# -----------------------------------------------------------------------------
# 示例1：基础张量操作（x + y * 2）
# -----------------------------------------------------------------------------
print("=" * 50)
print("示例1：基础张量操作 x + y * 2")
print("=" * 50)

# 创建两个张量
x = Tensor([1, 2, 3, 4])  # 形状 (4,)
y = Tensor([5, 6, 7, 8])  # 形状 (4,)

# 执行计算：x + y * 2
result = x + y * 2

print(f"x = {x.numpy()}")
print(f"y = {y.numpy()}")
print(f"x + y * 2 = {result.numpy()}")
print()

# -----------------------------------------------------------------------------
# 示例2：自动微分演示：求函数 f(x) = 3x² + 2x + 1 在x=2处的导数
# 导数公式 f'(x) = 6x + 2，在x=2处应该是 6*2 + 2 = 14
# -----------------------------------------------------------------------------
print("=" * 50)
print("示例2：自动微分求导")
print("=" * 50)

# 创建需要求导的张量，requires_grad=True表示需要计算梯度
x = Tensor(2.0, requires_grad=True)

# 前向传播计算函数值
f = 3 * x * x + 2 * x + 1
print(f"f(x) = 3x² + 2x + 1, x=2时 f(x) = {f.numpy()}")

# 反向传播计算梯度
f.backward()

# 查看梯度
print(f"f'(x)在x=2处的值 = {x.grad.numpy()} （预期值：14.0）")
print()

# -----------------------------------------------------------------------------
# 示例3：完整的梯度下降训练：拟合线性函数 y = 2x + 3
# -----------------------------------------------------------------------------
print("=" * 50)
print("示例3：梯度下降拟合线性函数 y = 2x + 3")
print("=" * 50)

# 生成训练数据
np.random.seed(42)
x_train = Tensor(np.random.rand(100).astype(np.float32) * 10)  # 100个0-10的随机数
y_train = (
    2 * x_train + 3 + Tensor(np.random.randn(100).astype(np.float32) * 0.5)
)  # 加一点噪声

# 初始化模型参数：我们要学习 w 和 b，使得 y = w*x + b
w = Tensor(np.random.randn(), requires_grad=True)
b = Tensor(np.random.randn(), requires_grad=True)
print(f"初始参数：w = {w.numpy():.4f}, b = {b.numpy():.4f} (目标值：w=2, b=3)")

# 训练配置
epochs = 100
lr = 0.01

# 训练循环
for i in range(epochs):
    # 前向传播：预测值
    y_pred = w * x_train + b

    # 计算损失：均方误差 MSE
    loss = ((y_pred - y_train) ** 2).mean()

    # 反向传播：计算梯度
    loss.backward()

    # 梯度下降更新参数（手动实现SGD，也可以用optim.SGD）
    w.assign((w - lr * w.grad).detach())
    b.assign((b - lr * b.grad).detach())

    # 清空梯度，准备下一轮
    w.grad = None
    b.grad = None

    # 每10轮打印一次
    if (i + 1) % 10 == 0:
        print(
            f"Epoch {i + 1:3d}/100 | 损失: {loss.numpy():.4f} | w = {w.numpy():.4f} | b = {b.numpy():.4f}"
        )

print(f"\n训练完成！最终参数：w = {w.numpy():.4f}, b = {b.numpy():.4f}")
print(f"和目标值的误差：Δw = {abs(w.numpy() - 2):.4f}, Δb = {abs(b.numpy() - 3):.4f}")
print()

# -----------------------------------------------------------------------------
# 示例4：查看计算图结构
# -----------------------------------------------------------------------------
print("=" * 50)
print("示例4：计算图结构演示")
print("=" * 50)

a = Tensor(1.0, requires_grad=True)
b = Tensor(2.0, requires_grad=True)
c = a * b
d = c + Tensor(3.0)
e = d * 2

print(f"a = {a.numpy()}, b = {b.numpy()}")
print(f"e = 2*(a*b + 3) = {e.numpy()}")

# 反向传播前查看计算图节点
print("\n计算图节点：")
print(f"e._ctx: {e._ctx.__class__.__name__}（乘法操作）")
print(
    f"e._ctx.parents[0]._ctx: {e._ctx.parents[0]._ctx.__class__.__name__}（加法操作）"
)
print(
    f"e._ctx.parents[0]._ctx.parents[0]._ctx: {e._ctx.parents[0]._ctx.parents[0]._ctx.__class__.__name__}（乘法操作）"
)

# 反向传播
e.backward()
print(f"\na的梯度（de/da）= {a.grad.numpy()} （数学上：de/da = 2*b = 4.0）")
print(f"b的梯度（de/db）= {b.grad.numpy()} （数学上：de/db = 2*a = 2.0）")
