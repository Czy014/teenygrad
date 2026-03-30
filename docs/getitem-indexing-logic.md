
# `__getitem__` 索引逻辑详解

本文档深入解析 `teenygrad/tensor.py` 中 `__getitem__` 方法的实现原理，这是一个复杂但功能强大的索引系统。

---

## 概述

`__getitem__` 支持丰富的索引类型，完全通过现有 Tensor 操作组合实现，无需新增底层内核。

**支持的索引类型：**
- 整数索引：`x[0]`
- 切片：`x[1:5]`, `x[::-1]`
- 张量索引（Fancy Indexing）：`x[tensor]`
- None（增加新维度）：`x[None]`
- 省略号：`x[...]`
- 组合索引：`x[0, :, tensor, None]`

---

## 核心处理流程

### 1. 规范化整数索引

```python
def normalize_int(e, i, dim_sz):
    if -dim_sz &lt;= e &lt; dim_sz:
        return e if e != -1 else dim_sz - 1
    raise IndexError(
        f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}"
    )
```

**功能：**
- 处理负数索引：`x[-1]` → `x[dim_sz-1]`
- 边界检查

---

### 2. 解析输入索引

```python
orig_slices = list(val) if isinstance(val, tuple) else [val]
count = defaultdict(list)
for i, v in enumerate(orig_slices):
    count[type(v)].append(i)
```

**功能：**
- 将输入转换为列表（无论是否为 tuple）
- 按类型分类统计索引

---

### 3. 处理省略号 `...`

```python
if len(ellipsis_found := count[type(Ellipsis)]) &gt; 1:
    raise IndexError("an index can only have a single ellipsis ('...')")

ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * (
    len(self.shape) - num_slices
)
```

**示例：**
```
x.shape = (2, 3, 4, 5)
x[1, ..., 2]  →  x[1, :, :, 2]
```

---

### 4. 基本切片处理（shrink + flip）

```python
valid_slices = [v for v in orig_slices if v is not None]
valid_slices = [
    v if isinstance(v, slice)
    else slice(y_ := normalize_int(v, i, dim_sz), y_ + 1) if isinstance(v, int)
    else slice(None)
    for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))
]

start, stop, strides = zip(*[s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)])

new_slice = tuple(
    ((0, 0) if e &lt; s else (s, e)) if st &gt; 0
    else ((0, 0) if e &gt; s else (e + 1, s + 1))
    for s, e, st in zip(start, stop, strides)
)

sliced_tensor = self.shrink(new_slice).flip(
    axis=[i for i, s in enumerate(strides) if s &lt; 0]
)
```

**处理逻辑：**
- **正步长**：直接用 `shrink` 截取 `[start, stop)`
- **负步长**：先 `shrink` 截取 `[stop+1, start+1)`，再 `flip` 翻转

**示例：**
```
x = [a, b, c, d, e]
x[1:4]    → shrink((1, 4)) → [b, c, d]
x[4:1:-1] → shrink((2, 5)) → [c, d, e], 然后 flip → [e, d, c]
```

---

### 5. 处理非 1 步长（stride ≠ 1）

当步长绝对值大于 1 时，使用 **Pad → Reshape → Shrink** 技巧：

```python
if any(abs(s) != 1 for s in strides):
    strides = tuple(abs(s) for s in strides)
    
    # 步骤 1: Pad 到能被 stride 整除
    padded_tensor = sliced_tensor.pad(
        tuple(
            (0, s - (dim_sz % s) if dim_sz % s != 0 else 0)
            for s, dim_sz in zip(strides, sliced_tensor.shape)
        )
    )
    
    # 步骤 2: Reshape 成 [dim_sz_padded//s, s]
    reshaped_tensor = padded_tensor.reshape(
        flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides))
    )
    
    # 步骤 3: Shrink 取每组的第一个元素 [:, 0]
    new_shape = reshaped_tensor.shape[::2]
    sliced_tensor = reshaped_tensor.shrink(
        tuple(flatten(((0, sh), (0, 1)) for sh in new_shape))
    )
```

**示例：**
```
x = [a, b, c, d, e, f, g], stride = 3

期望结果: [a, d, g]

处理过程:
1. Pad (因为 7 % 3 = 1 ≠ 0): [a, b, c, d, e, f, g, 0, 0]
2. Reshape: [[a, b, c], [d, e, f], [g, 0, 0]]
3. Shrink [:, 0]: [a, d, g]
```

---

### 6. 处理 None 和收集张量索引

```python
final_shape, it_shape, dim, tensors, dim_collapsed = (
    [], iter(new_shape), [], [], 0
)

for i, s in enumerate(orig_slices):
    if s is None:
        final_shape.append(1)  # 增加新维度
    else:  # s is int or slice or Tensor
        dim_shape = next(it_shape)
        if isinstance(s, int):
            dim_collapsed += 1  # 整数索引会压缩维度
        else:
            final_shape.append(dim_shape)
            if isinstance(s, Tensor):
                tensors.append(s)  # 收集张量索引
                dim.append(i - dim_collapsed)

ret = sliced_tensor.reshape(tuple(final_shape))
```

**关键变量：**
- `tensors`: 收集所有张量索引
- `dim`: 记录这些张量在最终维度中的位置
- `dim_collapsed`: 被整数索引压缩的维度数

**示例：**
```
x.shape = (2, 3, 4)
idx = Tensor([0, 1])
x[None, :, idx].shape  →  (1, 3, 2)
                        #   ↑   ↑  ↑
                        # None : idx
```

---

## Fancy Indexing（张量索引）核心实现

这是最精妙的部分，完全通过基本操作组合实现！

### 核心思想

```
用 arange == idx 构造 one-hot 掩码，然后 mul + sum 提取元素
```

### 完整实现

```python
if tensors:
    # 步骤 1: 规范化索引（处理负数）
    idx = [
        t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t
        for d, t in zip(dim, tensors)
    ]
    
    max_dim = max(i.ndim for i in idx)
    
    # 步骤 2: 计算 sum_dim（在哪个维度求和）
    sum_dim = [d if n == 0 else d + max_dim - n for n, d in enumerate(dim)]
    
    # 步骤 3: 创建 arange 并 reshape 到正确形状
    arange = [
        Tensor.arange(
            ret.shape[d],
            dtype=dtypes.int32,
            requires_grad=False,
            device=self.device,
        ).reshape(
            *[1] * sd, ret.shape[d], *[1] * (ret.ndim + max_dim - n - sd - 1)
        )
        for n, (sd, d) in enumerate(zip(sum_dim, dim))
    ]
    
    # 步骤 4: 将 idx reshape 到正确形状
    first_idx = [
        idx[0].reshape(
            *[1] * dim[0],
            *[1] * (1 + max_dim - idx[0].ndim),
            *idx[0].shape,
            *[1] * (ret.ndim - dim[0] - 1),
        )
    ]
    rest_idx = [
        i.reshape(
            *[1] * dim[0],
            *[1] * (max_dim - i.ndim),
            *i.shape,
            *[1] * (ret.ndim - dim[0] - n),
        )
        for n, i in enumerate(idx[1:], 1)
    ]
    idx = first_idx + rest_idx
    
    # 步骤 5: 扩展 ret 的形状
    ret = ret.reshape(
        *ret.shape[: sum_dim[0] + 1],
        *[1] * max_dim,
        *ret.shape[sum_dim[0] + 1 :],
    )
    
    # 步骤 6: 迭代应用 fancy indexing
    for a, i, sd in zip(arange, idx, sum_dim):
        ret = (a == i).mul(ret).sum(sd)
    
    # 步骤 7: 特殊情况 - 需要 permute 调整维度顺序
    if (
        dim[0] != 0
        and len(dim) != 1
        and dim != list(range(dim[0], dim[-1] + 1))
    ):
        ret_dims = list(range(ret.ndim))
        ret = ret.permute(
            ret_dims[dim[0] : dim[0] + max_dim]
            + ret_dims[: dim[0]]
            + ret_dims[dim[0] + max_dim :]
        )
```

### 简单示例详解

```python
x = Tensor([a, b, c, d])
idx = Tensor([0, 2])
result = x[idx]  # 期望: [a, c]
```

**执行过程：**

```
步骤 1: arange = [0, 1, 2, 3]

步骤 2: arange == idx
        [[1, 0, 0, 0],   # idx=0 的 one-hot
         [0, 0, 1, 0]]  # idx=2 的 one-hot

步骤 3: (arange == idx).mul(x)
        [[a, 0, 0, 0],
         [0, 0, c, 0]]

步骤 4: sum(dim=-1)
        [a, c]
```

### 多维示例

```python
x = Tensor([[a, b, c],
            [d, e, f]])  # shape (2, 3)

idx1 = Tensor([0, 1])
idx2 = Tensor([1, 2])

result = x[idx1, idx2]  # 期望: [b, f]
```

**执行过程：**
1. 先用 `idx1` 索引：得到 `[[a,b,c], [d,e,f]]`（扩展维度）
2. 再用 `idx2` 索引：在第二个维度求和，提取 `[b, f]`

---

## 完整数据流示例

```python
x = Tensor.arange(24).reshape(2, 3, 4)
# x = [[[ 0,  1,  2,  3],
#       [ 4,  5,  6,  7],
#       [ 8,  9, 10, 11]],
#      [[12, 13, 14, 15],
#       [16, 17, 18, 19],
#       [20, 21, 22, 23]]]

idx = Tensor([0, 2])
result = x[None, :, idx, 1::2]
```

**处理步骤：**

1. **解析索引：** `[None, :, idx, 1::2]`
2. **处理切片：** 
   - `1::2` → stride=2，用 pad+reshape+shrink 处理
   - 得到列：`[1, 3]`
3. **处理 None：** 增加新维度
4. **Fancy Indexing：** 用 `idx=[0, 2]` 索引行
5. **最终形状：** `(1, 2, 2, 2)`

---

## 关键技巧总结

| 技巧 | 用途 |
|------|------|
| `shrink` | 基本切片截取 |
| `flip` | 处理负步长 |
| `pad + reshape + shrink` | 处理 stride &gt; 1 |
| `arange == idx` | 构造 one-hot 掩码 |
| `mul + sum` | 提取选中元素 |
| `reshape` 广播 | 对齐维度进行运算 |

---

## 为什么这样实现？

1. **统一性**：所有索引都用现有操作实现，无需新增内核
2. **延迟计算**：所有操作都在 LazyBuffer 层面，可以融合优化
3. **自动微分**：索引操作也支持反向传播
4. **设备无关**：同一套代码在 CPU/GPU 上都能工作

---

## 与 PyTorch 的区别

| 特性 | TeenyGrad | PyTorch |
|------|-----------|---------|
| 实现方式 | 纯操作组合 | 专用内核 |
| 性能 | 依赖融合优化 | 高度优化 |
| 可读性 | 清晰易懂 | 复杂底层 |

这就是为什么文档说"可以先跳过"——虽然复杂，但理解后你会发现它是一个非常优雅的设计！

