

# 大语言模型解码与采样策略学习笔记

在大型语言模型（如ChatGPT）生成文本时，模型需要逐词（Token）进行预测。当模型计算出下一个词的概率分布后，我们需要通过一定的**策略**来决定最终输出哪个词。这些策略主要分为两大类：**解码策略（Decoding Strategy）和采样策略（Sampling Strategy）**。

## 1\. 核心概念与区别

为了直观理解，我们可以用“从糖果罐里拿糖果”来打比方，目标是拿出一串颜色搭配好看的糖果（即生成一段连贯的文本）。

  * **解码策略（Decoding Strategy）**：宏观概念，即你拿糖果的“总方针”。它决定了生成整体序列的算法框架，可以是确定性的（每次结果相同），也可以是随机性的。
  * **采样策略（Sampling Strategy）**：解码策略的一种具体实现手段。当你的总方针包含“随机性”时，采样策略就是用来控制**如何进行随机选择**的具体参数和细节（如温度、Top-K、Top-P）。

-----

## 2\. 确定性解码策略 (Deterministic Decoding)

确定性策略不包含随机因素，给定相同的输入，模型总是生成相同的输出。

### 2.1 贪心解码 (Greedy Decoding)

  * **核心思想**：只顾眼前，每一步都选择当前概率最大的那个词。
  * **做法**：模型在计算出下一个词的概率分布后，直接通过 `argmax()` 选取概率最高的词，以此类推。
  * **优点**：实现简单，生成速度最快；每次生成结果确定，可复现。同一个输入总是得到相同输出，适合需要确定性的场景。
  * **缺点**：
    1.  **可能会错过全局最优**：就像下棋只看眼前一步，可能导致后面没棋可走。有时候，当前可能性最高的词，放在整个句子里看可能不是最佳选择。
    2.  **容易重复**：因为它总是选最“安全”的词，可能会导致生成的文本比较单调、重复。好比：你在自助餐厅选菜，每次都只拿你面前看起来最好吃的那一道，不考虑整体搭配。

**简单代码示例（贪心解码流程）：**

```python
output_sequence = []
input_sequence = [起始符]  # 起始符表示开始生成

for t in range(max_length):
    probs = model.predict_next_word(input_sequence)  # 模型给出下一词概率分布
    next_word = np.argmax(probs)                     # 选择概率最大的词
    if next_word == 结束符:
        break
    output_sequence.append(next_word)
    input_sequence.append(next_word)
print("生成结果:", output_sequence)
```

在 HuggingFace Transformers 库中，贪心解码是 `model.generate()` 的默认行为（如果不启用采样或束搜索）。例如：

```python
# 使用Transformers进行贪心解码
output_ids = model.generate(input_ids, max_length=50, do_sample=False) # do_sample=False 表示不采样，使用贪心策略
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### 2.2 束搜索 (Beam Search)

  * **核心思想**：稍微多想几步，保留几个“备胎”选项。
  * **做法**：
    1.  设置一个“束宽”（Beam Width），比如 $k=3$。
    2.  第一步，机器人不像贪心那样只选1个，而是选出可能性最高的 $k$ 个词（比如“很好”、“不错”、“真棒”）。
    3.  对于这 $k$ 个词（我们称之为“候选路径”），分别预测它们各自的下一个词。这样可能会产生 $k * 很多$ 个新词组合。
    4.  从这些新词组合中，再选出整体可能性最高的 $k$ 条路径。
    5.  重复这个过程，直到生成完整的句子。
  * **优点**：
    1.  比贪心解码生成的句子质量通常更高，因为它考虑了更多的可能性组合。
    2.  在一定程度上避免了贪心解码的短视问题。
  * **缺点**：
    1.  计算量比贪心大，速度慢一些。
    2.  仍然可能偏向于生成高频、安全的词语，有时缺乏创造性。
    3.  束宽 $k$ 的选择很重要，太小了接近贪心，太大了计算量剧增。

**简单代码示例：下面用简化的伪代码演示 beam search 的流程（以 beam size = 2 为例）：**

```python
beam_size = 2
beams = [([起始符], 1.0)]  # 每个元素是(序列, 概率)，初始序列概率为1

for t in range(max_length):
    new_beams = []
    # 对当前每个候选序列扩展
    for seq, prob in beams:
        probs = model.predict_next_word(seq)  # 概率分布
        top_indices = np.argsort(probs)[-beam_size:]  # 找出概率最高的beam_size个词索引
        for idx in top_indices:
            new_seq = seq + [idx]
            new_prob = prob * probs[idx]  # 累乘概率 (或累加对数概率)
            new_beams.append((new_seq, new_prob))
    
    # 从扩展得到的新序列中选出概率最高的 beam_size 个
    new_beams.sort(key=lambda x: x[1], reverse=True)
    beams = new_beams[:beam_size]
    # 检查是否都已生成结束符，省略此处

# 最终 beams[0] 即为概率最大的序列
```

-----

## 3\. 不同采样策略说明

接下来的三个策略 (Temperature, Top-K, Top-P) 通常是**采样 (Sampling) 策略**的一部分，它们引入了随机性，让模型生成更多样化的文本。

### 3.1 温度 (Temperature)

  * **原理与作用**：温度 $T$ 并不是一种单独的采样策略，而是一个调节概率分布的参数。它通过改变模型输出的概率分布的“陡峭程度”，从而控制生成文本的随机性和创造性。
    公式相当于：$P'(w) = \frac{P(w)^{1/T}}{\sum_v P(v)^{1/T}}$
  * **参数影响**：
      * **温度值低（比如 \< 1）**：会让得分高的词优势更明显，得分低的词机会更小。机器人会变得更“保守”，倾向于选择高可能性的词，句子更稳定、确定，但可能缺乏惊喜。极端情况下，当 $T \to 0$ 时，退化为贪心解码的行为。
      * **温度值 = 1**：保持原始的可能性得分不变。
      * **温度值高（比如 \> 1）**：会让所有词的可能性得分变得更平均。得分高的词优势减弱，得分低的词机会增加。机器人会变得更“奔放”，敢于尝试不那么常见的词，句子更有创造性，但也可能胡言乱语。
  * **优缺点**：Temperature 提供了一个简单而直观的手段来控制生成的随机性。但它本质上只是调整了概率分布，不能彻底解决模型可能会生成离谱词汇的问题。

**简单代码示例：通过下面的小实验可以观察温度对概率分布的影响：**

```python
import numpy as np

probs = np.array([0.4, 0.3, 0.2, 0.05, 0.05])  # 原始分布
words = ['A', 'B', 'C', 'D', '<eos>']
for T in [0.5, 1.0, 1.5]:
    # 按温度调整概率分布
    adjusted_probs = probs ** (1.0 / T)
    adjusted_probs = adjusted_probs / adjusted_probs.sum()
    print(f"\nTemperature = {T}:")
    for w, p in zip(words, adjusted_probs):
        print(f"{w}: {p:.2f}")
```

在 HuggingFace 中，通过参数 `temperature` 来设置，例如：

```python
# 结合温度系数的采样
output_ids = model.generate(input_ids, max_length=50, 
                            do_sample=True, top_p=0.9, temperature=0.7)
```

### 3.2 Top-K 采样 (Top-K Sampling)

  * **核心思想**：只在一小撮“尖子生”里选。
  * **做法**：
    1.  设置一个 $K$ 值，比如 $K=50$。
    2.  在机器人要选下一个词时，先看所有词的“可能性”得分，然后只把得分排名前 $K$ 的那些词挑出来。
    3.  然后，机器人只在这 $K$ 个词里面，根据它们（可能经过温度调整后的）可能性得分进行随机选择。其他得分再低的词，统统不考虑。
  * **优点**：避免了选到那些可能性极低但又古怪的词，在保证一定合理性的前提下引入了随机性。
  * **缺点**：需要合理地调节 $K$ 值。如果 $K$ 太小，可能错过一些本来有意义但未被包括的词，限制创造性；如果 $K$ 很大，又可能选到极低概率的词，导致不通顺。并且固定的 $K$ 无法适应概率分布的变化。

**简单代码示例：下面通过一个例子演示 Top-K 采样的过程。假设模型当前给出以下词的概率分布：**

```python
import numpy as np

probs = np.array([0.4, 0.3, 0.2, 0.05, 0.05])  # 模型给出5个词的概率
words = ['A', 'B', 'C', 'D', '<eos>']          # 5个词 (<eos>表示结束符)
K = 3  # 选择 Top-3

# 1. 获取概率最高的 K 个词的索引
top_indices = np.argsort(probs)[-K:]  # 概率从小到大排序后取最后3个索引
# 2. 保留这 K 个词的概率，其他置零
top_k_probs = np.zeros_like(probs)
top_k_probs[top_indices] = probs[top_indices]
# 3. 重新归一化
top_k_probs = top_k_probs / np.sum(top_k_probs)

# 打印候选词及归一化后的概率
print("Top-K 候选及概率:")
for idx in top_indices:
    print(f"{words[idx]}: {top_k_probs[idx]:.2f}")
```

### 3.3 Top-P (核心) 采样 (Top-P / Nucleus Sampling)

  * **核心思想**：不固定保留多少个词，而是依据累积概率来动态决定候选集合的大小。
  * **做法**：
    1.  设置一个概率阈值 $P$，比如 $P=0.9$ (表示90%)。
    2.  机器人把所有可能的下一个词按“可能性”从高到低排序。
    3.  从可能性最高的词开始，依次把它们的可能性加起来，直到这个累加和第一次达到或超过 $P$ 为止。
    4.  这些被累加进来的词，就构成了“核心候选圈” (Nucleus)。其余词舍弃。
    5.  最后，机器人只在这个圈子里进行随机采样。
  * **优势**：自适应性强。当模型对下一个词很确定时，候选圈很小；当模型犹豫不决时，候选圈会变大，以此保证候选集覆盖了总概率的 $P$ 比例。

**简单代码示例：演示 Top-P 采样的过程**

```python
import numpy as np

# 假设模型当前给出的原始概率分布 (已按从高到低排序为前提演示)
probs = np.array([0.4, 0.3, 0.2, 0.05, 0.05])
words = ['A', 'B', 'C', 'D', '<eos>']
P = 0.8  # 设置 Top-P 阈值为 80%

print("原始概率分布:")
for w, p in zip(words, probs):
    print(f"{w}: {p:.2f}")

# 1. 计算累积概率
cumulative_probs = np.cumsum(probs)
# cumulative_probs 结果为: [0.4, 0.7, 0.9, 0.95, 1.0]

# 2. 找出需要剔除的索引 (累积概率超过 P 的部分)
# 我们需要保留累积概率刚好覆盖到 P 的那些词
indices_to_remove = cumulative_probs > P

# 将掩码向右平移一位，确保第一个使累积概率超过 P 的词（即本例中的'C'）被保留
indices_to_remove[1:] = indices_to_remove[:-1].copy()
indices_to_remove[0] = False 

# 3. 将被剔除的词概率设为0
top_p_probs = np.copy(probs)
top_p_probs[indices_to_remove] = 0.0

# 4. 重新归一化，使得保留下来的词概率和为1
top_p_probs = top_p_probs / np.sum(top_p_probs)

print(f"\nTop-P={P} 候选及归一化后概率:")
for w, p in zip(words, top_p_probs):
    if p > 0:
         print(f"{w}: {p:.2f}")
```

-----

## 4\. 总结与组合使用使用

在现代大语言模型的实际应用中，上述策略往往不是孤立使用的，而是**组合出击**：

1.  **明确场景，选择基调**：

      * **需要绝对准确（如：代码生成、数学推导、信息提取）**：关闭采样（`do_sample=False`），直接使用 **贪心解码** 或 **束搜索 (Beam Search)**，或将 **Temperature 设为极低**。
      * **需要创造力与多样性（如：故事续写、闲聊机器人）**：开启采样（`do_sample=True`），组合使用 Temperature、Top-K 和 Top-P。

2.  **经典的组合技配置顺序**：
    通常在代码底层，这些参数是按顺序对模型输出进行修剪的：

      * **第一步：温度 (Temperature)** 先改变整体的概率分布形态。
      * **第二步：Top-K** 砍掉排名靠后的长尾词，避免出现极其生僻的词。
      * **第三步：Top-P** 进一步动态裁剪，留下核心候选集。
      * **第四步**：在最终清洗过且重新归一化的概率池里进行随机抽样。

3.  **常用参数经验值参考**：

      * `temperature = 0.7`
      * `top_k = 50`
      * `top_p = 0.9`

-----

### 笔记外附加说明：原文错误与不准确之处指正

这部分是我在为您整理笔记时，发现原图中存在的瑕疵并已在正文中修正的内容：


1.  **修正了 Top-K 描述中的逻辑语病**：图片介绍 Top-K 时写道“其他得分再高的词（如果不在前K名）或者得分再低的词，统统不考虑”。既然已经被排除在前K名之外，就不可能存在“得分再高”的词，只可能是得分相对较低的词。笔记中已修正了这一逻辑表述。
2.  **更正了对“温度”的定义**：图片中有一句“温度本质上只是重排了概率分布”。“重排 (Reorder)”这个词是完全错误的。温度参数改变的是概率分布的数值大小和陡峭程度，它**绝对不会改变词汇的概率排名顺序**。因此，笔记中将其表述更正为“调节概率分布的陡峭程度”。