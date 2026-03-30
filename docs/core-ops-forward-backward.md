# 四大核心操作：前向与反向传播详解

本文档详细解析 Movement、Reduce、Unary、Binary 四类核心操作的前向传播和反向传播机制。

---

## 1. Movement 操作 - 以 `Reshape` 为例

### 操作说明
`Reshape` 改变张量的形状，但不改变数据的顺序和总量。

### 前向传播

```python
class Reshape(Function):
    def forward(self, x: LazyBuffer, shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape  # 保存输入形状，用于反向传播
        return x.reshape(shape)     # 执行 reshape
```

**前向示例：**
```
输入 x: shape=(2, 3)
     [[1, 2, 3],
      [4, 5, 6]]
      
reshape(6): shape=(6,)
     [1, 2, 3, 4, 5, 6]
```

**前向要点：**
- 保存 `input_shape` 到 `self`，反向传播需要用到
- 数据本身不变，只是改变看待数据的方式

---

### 反向传播

```python
def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    return grad_output.reshape(self.input_shape)
```

**反向示例：**
```
grad_output: shape=(6,)
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            
reshape(2, 3): shape=(2, 3)
            [[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6]]
```

**反向传播数学原理：**

```
y = reshape(x)
dy/dx = I  (单位矩阵，因为 reshape 是恒等变换，只是形状改变)

因此：
dL/dx = dL/dy * dy/dx = dL/dy  (只需把梯度 reshape 回去即可)
```

**完整示例代码：**

```python
x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
y = x.reshape(6)  # 前向
z = y.sum()       # 为了能 backward，创建标量
z.backward()

print(x.grad)  # [[1, 1, 1], [1, 1, 1]]
```

---

## 2. Reduce 操作 - 以 `Sum` 为例

### 操作说明
`Sum` 沿指定轴求和，减少张量的维度。

### 前向传播

```python
class Sum(Function):
    def forward(self, x: LazyBuffer, new_shape: Tuple[int, ...]) -> LazyBuffer:
        self.input_shape = x.shape  # 保存输入形状
        return x.r(ReduceOps.SUM, new_shape)  # 执行求和归约
```

**前向示例：**
```
输入 x: shape=(2, 3)
     [[1, 2, 3],
      [4, 5, 6]]
      
sum(axis=1): shape=(2,)
     [6, 15]  (1+2+3=6, 4+5+6=15)
```

**前向要点：**
- 保存完整的 `input_shape`
- `new_shape` 是减少后的形状

---

### 反向传播

```python
def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    return grad_output.expand(self.input_shape)
```

**反向示例：**
```
grad_output: shape=(2,)
            [0.1, 0.2]
            
expand(2, 3): shape=(2, 3)
            [[0.1, 0.1, 0.1],
             [0.2, 0.2, 0.2]]
```

**反向传播数学原理：**

```
y_i = sum_j x_ij

dy_i/dx_ij = 1  (对每个 x_ij，只影响一个 y_i)

因此：
dL/dx_ij = dL/dy_i * dy_i/dx_ij = dL/dy_i

即：每个 x_ij 的梯度等于对应 y_i 的梯度
```

**完整示例代码：**

```python
x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
y = x.sum(axis=1)  # [6, 15]
z = y.sum()         # 21
z.backward()

print(x.grad)  # [[1, 1, 1], [1, 1, 1]]
```

---

## 3. Unary 操作 - 以 `Exp` 为例

### 操作说明
`Exp` 逐元素计算指数函数：y = e^x

### 前向传播

```python
class Exp(Function):
    def forward(self, x: LazyBuffer) -> LazyBuffer:
        # 用 exp2 实现，乘以常数转换底数
        self.ret = x.e(BinaryOps.MUL, x.const(1 / math.log(2))).e(UnaryOps.EXP2)
        return self.ret
```

**前向示例：**
```
输入 x: [0, 1, 2]

exp(x): [1, 2.718..., 7.389...]
        [e^0, e^1, e^2]
```

**前向要点：**
- 保存输出 `self.ret`，反向传播需要用到
- 实际用 exp2 实现，通过 `x / ln(2)` 转换

---

### 反向传播

```python
def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    return self.ret.e(BinaryOps.MUL, grad_output)
```

**反向示例：**
```
self.ret (前向输出): [1, 2.718, 7.389]
grad_output:           [0.1, 0.2, 0.3]

grad_input:            [0.1*1, 0.2*2.718, 0.3*7.389]
                     = [0.1, 0.5436, 2.2167]
```

**反向传播数学原理：**

```
y = e^x

dy/dx = e^x = y  (指数函数的导数是自身!)

因此：
dL/dx = dL/dy * dy/dx = dL/dy * y
```

这就是为什么保存 `self.ret`（即 y）就够了！

**完整示例代码：**

```python
x = Tensor([0, 1, 2], requires_grad=True)
y = x.exp()    # [1, 2.718, 7.389]
z = y.sum()    # 11.107
z.backward()

print(x.grad)  # [1, 2.718, 7.389]  等于 y 的值!
```

---

## 4. Binary 操作 - 以 `Mul` 为例

### 操作说明
`Mul` 逐元素相乘：z = x * y

### 前向传播

```python
class Mul(Function):
    def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
        self.x, self.y = x, y  # 保存两个输入
        return x.e(BinaryOps.MUL, y)
```

**前向示例：**
```
x: [2, 3, 4]
y: [5, 6, 7]

x * y: [10, 18, 28]
```

**前向要点：**
- 同时保存 `self.x` 和 `self.y`，两个都要用于反向传播

---

### 反向传播

```python
def backward(self, grad_output: LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return (
        self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None,
        self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None
    )
```

**反向示例：**
```
x: [2, 3, 4]
y: [5, 6, 7]
grad_output: [0.1, 0.2, 0.3]

grad_x = y * grad_output = [5*0.1, 6*0.2, 7*0.3] = [0.5, 1.2, 2.1]
grad_y = x * grad_output = [2*0.1, 3*0.2, 4*0.3] = [0.2, 0.6, 1.2]
```

**反向传播数学原理：**

```
z = x * y

dz/dx = y
dz/dy = x

因此：
dL/dx = dL/dz * dz/dx = dL/dz * y
dL/dy = dL/dz * dz/dy = dL/dz * x
```

**完整示例代码：**

```python
x = Tensor([2, 3, 4], requires_grad=True)
y = Tensor([5, 6, 7], requires_grad=True)
z = x * y          # [10, 18, 28]
w = z.sum()        # 56
w.backward()

print(x.grad)  # [5, 6, 7]   等于 y 的值
print(y.grad)  # [2, 3, 4]   等于 x 的值
```

---

## 对比总结

| 操作类型 | 示例 | 前向保存 | 反向操作 | 导数公式 |
|---------|------|---------|---------|---------|
| **Movement** | `Reshape` | `input_shape` | `reshape(back)` | dy/dx = I |
| **Reduce** | `Sum` | `input_shape` | `expand(back)` | dy_i/dx_ij = 1 |
| **Unary** | `Exp` | `ret` (输出) | `ret * grad` | dy/dx = y |
| **Binary** | `Mul` | `x`, `y` (输入) | `y*grad`, `x*grad` | dz/dx=y, dz/dy=x |

---

## 记忆技巧

1. **Movement**: 形状改变，数据不变 → 反向就是"恢复原状"
2. **Reduce**: 多 → 少 → 反向就是"广播回去"
3. **Unary**: y = f(x) → 通常需要保存 y 或 x 来计算 dy/dx
4. **Binary**: z = f(x, y) → 两个输入都要保存，返回两个梯度
