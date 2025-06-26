# 1. 前言

本文基于[DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)代码和MLA[论文](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/2405.04434)学习MLA计算流程。设(臆)想中，可能还会参考[flashinfer](https://github.com/flashinfer-ai/flashinfer)中的MLA代码进行讲解。

# 2. MLA in Paper

![[MLA.png]]

# 3. MLA in Code

## 3.1 Initialization and Pre-Calculation

### 3.1.1 Initialization

```python
dim = 7168
n_heads = 128
n_local_heads = 128 // world_size
q_lora_rank = 1536
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
qk_head_dim = qk_nope_head_dim + qk_rope_head_dim = 192
v_head_dim = 128

wq_a = Linear(dim, q_lora_rank) # [7168, 1536]
wq_b = ColumnParallelLinear(q_lora_rank, n_heads * qk_head_dim) # [1536, n_local_heads * 192] on one rank

wkv_a = Linear(dim, kv_lora_rank + qk_rope_head_dim) # [7168, 512 + 64]
wkv_b = ColumnParallelLinear(kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)) # [512, n_local_heads * (128 + 128)] on one rank

wo = RowParallelLinear(n_heads * v_head_dim, dim) # [n_local_heads * 128, 7168] on one rank
```

### 3.1.2 Pre-Calculation

```python
q = wq_b(q_norm(wq_a(x)))  # x is [bsz, seqlen, dim], q is [bsz, seqlen, n_local_heads * qk_head_dim]
q = q.view(bsz, seqlen, n_local_heads, qk_head_dim)
q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
q_pe = apply_rotary_emb(q_pe, freqs_cis) # [bsz, seqlen, n_local_heads, qk_rope_head_dim]
```

这里，```c = wq_a(x)```相当于
$$ c_t^Q = W^{DQ} h_t \in \mathbb{R}^{B \times L \times 1536} $$
```q = wq_b(c)```相当于下面两个公式
$$ q_t^C = W^{UQ} c_t^Q \in \mathbb{R}^{B \times L \times H \times 128}$$
$$q_t^R = W^{QR} c_t^Q \in \mathbb{R}^{B \times L \times H \times 64} $$
然后经过```apply_rotary_emb```，也就是对$q_t^R$做了$\mathrm{RoPE}$，然后可以通过拼接得到最终的$q_t$

```python
kv = wkv_a(x) # [bsz, seqlen, kv_lora_rank + qk_rope_head_dim]
kv, k_pe = torch.split(kv, [kv_lora_rank, qk_rope_head_dim], dim=-1) # kv: [bsz, seqlen, kv_lora_rank]
k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis) # k_pe.unsqueeze(2): [bsz, seqlen, 1, qk_rope_head_dim]
```

这里，```kv = wkv_a(x)```相当于以下两部分的整合
$$ c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{B \times L \times 512} $$
$$ k_t^R =W^{KR} h_t \in \mathbb{R}^{B \times L \times 64} $$
在```torch.split```之后，```kv```即压缩的KV低秩表示$c_t^{KV}$，```k_pe```经过```apply_rotary_emb```，得到了K的$\mathrm{RoPE}$部分，等待后续拼接等操作。

综上，在初始化部分，我们得到了以下变量：
	$q_t^C$  Query 无$\mathrm{RoPE}$的部分
	$q_t^R$  Query 有$\mathrm{RoPE}$的部分
	$c_t^{KV}$  KV的联合低秩表示
	$k_t^R$  Key 有$\mathrm{RoPE}$的部分

## 3.2 MLA Implementation

### 3.2.1 Naive

```python
q = torch.cat([q_nope, q_pe], dim=-1) # [bsz, seqlen, n_local_heads, qk_nope_head_dim + qk_rope_head_dim]
kv = wkv_b(kv_norm(kv)) # [bsz, seqlen, n_local_heads * (128 + 128)]
kv = kv.view(bsz, seqlen, n_local_heads, qk_nope_head_dim + v_head_dim)
k_nope, v = torch.split(kv, [qk_nope_head_dim, v_head_dim], dim=-1)
k = torch.cat([k_nope, k_pe.expand(-1, -1, n_local_heads, -1)], dim=-1) # [bsz, seqlen, n_local_heads, qk_head_dim]
```

这里，```torch.cat```得到了完整的Query
$$ q_t = [q_t^C, q_t^R] \in \mathbb{R}^{B \times L \times H \times 192} $$
```kv = wkv_b(kv_norm(kv))```相当于以下两个公式的整合
$$ k_t^C = W^{UK} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$
$$ v_t = W^{UV} c_t^{KV} \in \mathbb{R}^{B \times L \times H \times 128} $$
然后对```k_pe```做广播，拼接得到Key，即
$$ k_t = \begin{bmatrix}
    k_{t,1}^C & k_t^R \\ 
    k_{t,2}^C & k_t^R \\
    \vdots & \vdots \\
    \end{bmatrix} \in \mathbb{R}^{B \times L \times H \times 192} $$
```python
k_cache[:bsz, start_pos:end_pos] = k
v_cache[:bsz, start_pos:end_pos] = v
```

这里看到，cache的是完整的Key和Value。对于**每层每个token**，大小是
$$k = bsz * seqlen * n\_local\_heads * (qk\_nope\_head\_dim + qk\_rope\_head\_dim) = 128 * 192 = 24576$$
$$v = bsz * seqlen * n\_local\_heads * v\_head\_dim = 128*128 = 16384$$
二者求和得到40960，假设用fp8存储，那么大小为**40.96KB**

```python
# q: [bsz, seqlen, n_local_heads, qk_head_dim]
# k: [bsz, len, n_local_heads, qk_head_dim]
# scores: [bsz, seqlen, n_local_heads, len]
scores = torch.einsum("bshd,bthd->bsht", q, k_cache[:bsz, :end_pos]) * softmax_scale
# v: [bsz, len, n_local_heads, v_head_dim]
# x: [bsz, seqlen, n_local_heads, v_head_dim]
o = torch.einsum("bsht,bthd->bshd", scores, v_cache[:bsz, :end_pos])
```

即标准的Attention计算
$$ scores = \mathrm{softmax}\left(\frac{q_t^\top k_t + \mathrm{Mask}}{\sqrt{192}}\right) = 
\mathrm{softmax}\left(\frac{{q_t^C}^\top k_t^C + {q_t^R}^\top k_t^R + \mathrm{Mask}}{\sqrt{128 + 64}} \right)
\in \mathbb{R}^{B \times L \times H \times L} $$
$$ o = scores \cdot v_t \in \mathbb{R}^{B \times L \times H \times 128}$$
### 3.2.2 Absorb

论文中提到如下absorb方法
![[MLA_Absorb.png]]

从上文的分析，我们知道：
	$W^{UK}$是```wkv_b```权重的一部分，具体来说是有关```qk_nope_head_dim```的部分；
	$W^Q$在论文里指的是从输入```hidden_states```到$q_t$的变换矩阵，也就是这里的$W^{DQ}, W^{UQ}, W^{QR}$中的一部分；
	$W^{UV}$是```wkv_b```权重里有关```v_head_dim```的部分；
	$W^O$是输出的投影矩阵。

也就是说：可以把K的解压缩合并到Q投影中，V的解压缩合并到O投影中，从而起到只cache未解压缩的KV，降低cache显存占用的作用。
转换成代码角度，就是把```wkv_b```拆开，一部分与```q```的计算合并，一部分与```o```的计算合并。

考虑Attention的计算过程

对于Key的吸收，非$\mathrm{RoPE}$部分可以做如下展开

$$
{q_t^C}^\top k_t^C = (W^{UQ} c_t^Q)^{\top} W^{UK} c_t^{KV} = {c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK} c_t^{KV} = ({c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK}) c_t^{KV} 
$$
所以可以只缓存$c_t^{KV}$，计算${c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK}$的部分。

结合代码做分析

```python
wkv_b = wkv_b.weight.view(n_local_heads, -1, kv_lora_rank) # 中间的维度是(qk_nope_head_dim + v_head_dim)
# q_nope is [bsz, seqlen, n_local_heads, qk_nope_head_dim], output is [bsz, seqlen, n_local_heads, kv_lora_rank]
q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, qk_nope_head_dim])
```

这里的```q_nope```相当于${c_t^Q}^{\top}{W^{UQ}}^{\top} W^{UK}$，所以下面非$\mathrm{RoPE}$的部分缓存也只存了$c_t^{KV}$。对于$\mathrm{RoPE}$的部分，缓存对象是```squeeze```回来的```k_pe```

需要注意，这里```kv_cache```储存的$c_t^{KV}$还包含了V的低秩表示，在下面Value的吸收部分也要用到。

```python
kv_cache[:bsz, start_pos:end_pos] = kv_norm(kv) # [bsz, seqlen, kv_lora_rank]
pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2) # [bsz, seqlen, qk_rope_head_dim]
```

Attention里有关KV的部分，计算方式变为：

```python
scores = torch.einsum("bshc,btc->bsht", q_nope, kv_cache[:bsz, :end_pos])
# q_pe is [bsz, seqlen, n_local_heads, qk_rope_head_dim], output is [bsz, seqlen, n_local_heads, len]
scores += torch.einsum("bshr,btr->bsht", q_pe, pe_cache[:bsz, :end_pos])
```

其实只是把$q_t^\top k_t$按照$\mathrm{RoPE}$和非$\mathrm{RoPE}$两部分展开了。

对于Value的吸收，参考Attention公式，我们有：

$$ o = scores \cdot v_t \in \mathbb{R}^{B \times L \times H \times 128}$$
$$u_t = W^o o$$
其中，$v_t = W^{UK}·c_t^{KV}$

整合之后，即
$$u_t = W^o · scores · W^{UK} · c_t^{KV}$$
所以缓存对象还是$c_t^{KV}$不变，只需要计算公式左边的部分即可

代码分析

```python
x = torch.einsum("bsht,btc->bshc", scores, kv_cache[:bsz, :end_pos])
x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -v_head_dim:])
```

这里贴的是DeepSeek-V3仓库提供的代码，实际上就是上面公式从右到左矩阵乘的变形。

因为这部分文献不太全面，所以自己写了一段做验证：

```python
# DeepSeek原始代码
x = torch.einsum("bsht,btc->bshc", scores, ctkv)
x = torch.einsum("bshc,hdc->bshd", x, Wuk)
x = Wo(x.flatten(2))

print(x.shape)

# 自己推导的，从右向左矩阵乘
y = torch.einsum("hvk,blk->bhlv", Wuk, ctkv)
y = torch.einsum("bhtv,blht->blhv", y, scores)
y = Wo(y.flatten(2))
print(y.shape)

torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-4)
print("Test passed!")
```

结果对比可以通过，而且看耗时下面的要优于上面的。由于```torch.profiler.profile```对```torch.einsum```支持不全面，打印不出来FLOPs，所以暂且不知道为什么DeepSeek要使用上面的写法。

总的来说，此种情况下，对于**每层每个token**，cache大小为

$$bsz * seqlen * kv\_lora\_rank + bsz * seqlen * qk\_rope\_head\_dim = 512 + 64 = 576$$
假设使用fp8存储，则只需要**0.576KB**。相比naive，有巨大的节省。
