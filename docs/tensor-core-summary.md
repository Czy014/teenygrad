# Tensor 核心实现总结

## 概述

本文档总结 `teenygrad/tensor.py` 中的核心实现，重点关注自动微分（前向/反向传播）机制，以及区分核心功能与 PyTorch API 对齐部分。

---

## 核心组件

### 1. Function 类 - 自动微分的基础

```python
class Function:
    def __init__(self, device: str, *tensors: Tensor):
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = True if any(self.needs_input_grad) else None \
            if None in self.needs_input_grad else False
        if self.requires_grad:
            self.parents = tensors

    @classmethod
    def apply(fxn: Type[Function], *x: Tensor, **kwargs) -> Tensor:
        ctx = fxn(x[0].device, *x)
        ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs),
                    device=ctx.device, requires_grad=ctx.requires_grad)
        if ctx.requires_grad and not Tensor.no_grad:
            ret._ctx = ctx  # 用于自动微分引擎
        return ret
```

**核心要点：**
- `Function` 是所有操作的基类，记录计算历史
- `apply` 类方法是关键：创建上下文 → 执行 forward → 保存 ctx 到 `_ctx` 供反向传播使用
- `parents` 保存输入张量，构成计算图

---

### 2. Tensor 类的核心属性

```python
class Tensor:
    __slots__ = "lazydata", "requires_grad", "grad", "_ctx"
    training: ClassVar[bool] = False
    no_grad: ClassVar[bool] = False
```

| 属性 | 用途 |
|------|------|
| `lazydata` | 实际的数据存储（LazyBuffer），支持延迟计算 |
| `requires_grad` | 是否需要计算梯度 |
| `grad` | 存储计算得到的梯度 |
| `_ctx` | 指向创建该张量的 Function 上下文（反向传播入口） |

---

### 3. 前向传播机制

前向传播通过 `Function.apply()` 触发：

1. 创建 Function 实例（ctx），记录输入张量的 `requires_grad` 状态
2. 调用 `ctx.forward()` 执行实际计算（操作 LazyBuffer）
3. 创建新 Tensor 包装结果
4. **关键**：如果需要梯度且不在 `no_grad` 模式下，将 ctx 保存到 `ret._ctx`

---

### 4. 反向传播（核心中的核心）

```python
def backward(self) -> Tensor:
    assert self.shape == tuple(), "backward 只能对标量张量调用"
    
    # 初始梯度设为 1
    self.grad = Tensor(1, device=self.device, requires_grad=False)
    
    # 拓扑排序遍历计算图
    for t0 in reversed(self.deepwalk()):
        assert t0.grad is not None
        # 调用 backward 得到输入的梯度
        grads = t0._ctx.backward(t0.grad.lazydata)
        grads = [Tensor(g, device=self.device, requires_grad=False) 
                 if g is not None else None 
                 for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
        
        # 累积梯度到父节点
        for t, g in zip(t0._ctx.parents, grads):
            if g is not None and t.requires_grad:
                assert g.shape == t.shape
                t.grad = g if t.grad is None else (t.grad + g)
        del t0._ctx
    return self
```

**反向传播步骤：**
1. 从标量输出开始，初始梯度 `self.grad = 1`
2. `deepwalk()` 进行拓扑排序，确保按依赖关系遍历
3. 对每个节点调用 `_ctx.backward(grad)`，传入输出梯度
4. 将得到的梯度累积到父节点的 `grad` 属性
5. 清理 `_ctx` 释放内存

**计算图构建：**
```python
def deepwalk(self):
    def _deepwalk(node, visited, nodes):
        visited.add(node)
        if getattr(node, "_ctx", None):
            for i in node._ctx.parents:
                if i not in visited:
                    _deepwalk(i, visited, nodes)
            nodes.append(node)
        return nodes
    return _deepwalk(self, set(), [])
```

---

### 5. LazyBuffer 集成

Tensor 的 `lazydata` 是 `LazyBuffer` 类型，实现了：
- 延迟计算（操作先记录成 schedule，最后一起执行）
- 设备管理
- 内存优化

**关键方法：**
- `realize()` - 触发实际计算
- `schedule()` - 生成计算调度
- `corealize()` - 批量 realize 多个张量

---

## 核心操作分类

### A. 必须理解的核心操作

| 类别 | 操作 |
|------|------|
| **自动微分** | `backward()`, `deepwalk()`, `Function.apply()` |
| **计算图** | `_ctx`, `parents`, `requires_grad` |
| **Movement** | `reshape`, `expand`, `permute`, `shrink`, `pad` |
| **Reduce** | `sum`, `max`, `mean`（通过 `_reduce` 统一实现） |
| **Unary** | `neg`, `exp`, `log`, `relu` 等（通过 mlops） |
| **Binary** | `add`, `mul`, `sub`, `div`（带广播 `_broadcasted`） |

### B. PyTorch API 对齐（次要，仅为兼容）

以下方法主要是为了与 PyTorch 接口保持一致，核心逻辑简单或只是包装：

| 方法 | 说明 |
|------|------|
| `__getitem__` | 复杂的索引处理，支持 fancy indexing |
| `conv2d`, `max_pool2d`, `avg_pool2d` | 神经网络层，内部调用 `_pool` |
| `softmax`, `log_softmax` | 基于基本操作组合 |
| `gelu`, `mish`, `swish` 等激活函数 | 数学公式组合 |
| `layernorm`, `batchnorm` | 归一化层 |
| `scaled_dot_product_attention` | Transformer 注意力 |
| `to`, `to_`, `cpu()`, `cuda()` 等 | 设备转换包装 |
| `numpy()`, `item()` | 数据导出 |
| `triu`, `tril`, `cumsum` | 辅助操作 |
| `randn`, `randint`, `normal`, `uniform` | 随机数生成 |
| `kaiming_uniform`, `glorot_uniform` | 初始化方法 |
| `__add__`, `__mul__` 等运算符重载 | Python 语法糖 |

---

## 数据流示例

```
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)
z = x * y + x  # 前向传播
z.backward()    # 反向传播

# 计算图:
# z = Add(Mul(x, y), x)
# 
# 反向传播:
# dz/dz = 1
# dz/d(Mul) = 1, dz/d(x) = 1
# dz/dx = y*1 + 1 = 4
# dz/dy = x*1 = 2
```

---

## 总结 - 重点关注

### 🔴 核心中的核心
1. **`Function.apply()`** - 前向传播入口，连接计算与自动微分
2. **`Tensor._ctx`** - 保存计算历史，反向传播的桥梁
3. **`backward()` + `deepwalk()`** - 反向传播算法实现
4. **梯度累积** - `t.grad = g if t.grad is None else (t.grad + g)`

### 🟡 重要辅助
- `LazyBuffer` 延迟计算机制
- `_broadcasted()` 广播处理
- `_reduce()` 统一的归约操作

### 🟢 可以先跳过
- 复杂的 `__getitem__` 索引逻辑
- Winograd 卷积优化
- 各种 NN 层和激活函数的具体数学公式
- 与 PyTorch 完全对齐的辅助方法
