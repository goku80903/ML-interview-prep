"""
Reference material for the "Study" tab -- conceptual write-ups (not runnable
exercises) plus a small pool of self-check quiz questions per section.

Content originates from the user's own "Research Scientist Study Guide"
artifact; ported here so it can live alongside the coding exercises and get
randomized quiz questions layered on top.
"""

STUDY_CATEGORIES = [
    {"id": "foundations", "label": "Foundations"},
    {"id": "scaling", "label": "Scaling & Training"},
    {"id": "reasoning", "label": "Reasoning & Agents"},
    {"id": "evaluation", "label": "Evaluation & Systems"},
    {"id": "extras", "label": "Extras & Practice"},
]

STUDY_SECTIONS = [
    {
        "id": "attn-fundamentals",
        "category": "foundations",
        "title": "Transformer Architecture Fundamentals",
        "videos": [
            {"title": 'Transformer Architecture: Attention is All you Need Paper Explained', "url": 'https://www.youtube.com/watch?v=VygOX3AyDQs'},
        ],
        "html": '''
<p><strong>The core idea:</strong> replace recurrence with attention so every token can directly attend to every other token, enabling parallel training and better long-range dependency modeling than RNNs/LSTMs.</p>
<h3>Scaled dot-product attention</h3>
<pre><code>Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V</code></pre>
<ul>
  <li>Q, K, V are linear projections of the input embeddings (separate learned weight matrices).</li>
  <li><code>QK^T</code> gives a similarity score between every pair of tokens.</li>
  <li>Dividing by <code>sqrt(d_k)</code> matters: as <code>d_k</code> grows, dot products grow in magnitude (variance scales with <code>d_k</code>), pushing softmax into saturated regions with near-zero gradients. Scaling keeps the pre-softmax logits in a well-behaved range. <strong>This is a very common "why" question -- know the variance argument, not just the fact.</strong></li>
</ul>
<h3>Multi-head attention</h3>
<p>Run several attention operations in parallel on lower-dimensional projections of Q/K/V, concatenate, then project back. Lets the model attend to different relationship types (e.g. syntactic vs. semantic) simultaneously, rather than averaging everything into one attention pattern.</p>
<h3>Causal masking</h3>
<p>Decoder-only models set attention scores for future positions to <code>-inf</code> before softmax, so token <em>i</em> can only attend to tokens &le; i. This is what makes autoregressive generation and teacher-forced training consistent.</p>
<h3>Feed-forward network (position-wise MLP)</h3>
<p>Two linear layers with a nonlinearity between them, applied identically to each position. Typically expands to ~4x the hidden dimension then projects back. Original Transformer used ReLU; modern LLMs use GELU or gated variants like SwiGLU (LLaMA), which tend to outperform plain ReLU/GELU at scale.</p>
<h3>Residual connections + normalization</h3>
<ul>
  <li>Residuals let gradients flow through many layers without vanishing.</li>
  <li><strong>Pre-norm vs. post-norm:</strong> original Transformer used post-norm (LayerNorm after the sublayer). Modern large-scale LLMs (GPT-3+, LLaMA, PaLM) use pre-norm because it's more stable to train at depth/scale.</li>
  <li><strong>LayerNorm vs. RMSNorm:</strong> RMSNorm skips mean-centering and only rescales by the root-mean-square of activations. Cheaper, and empirically works as well or better for LLMs.</li>
</ul>
<h3>Encoder-only vs. decoder-only vs. encoder-decoder</h3>
<ul>
  <li>Encoder-only (BERT): bidirectional attention, masked language modeling. Good for representation/classification, not generation.</li>
  <li>Decoder-only (GPT family): causal attention, next-token prediction. What almost all modern general-purpose LLMs use.</li>
  <li>Encoder-decoder (T5, BART): bidirectional encoder, autoregressive decoder cross-attending to it. Good fit for translation/summarization.</li>
  <li><strong>Why decoder-only won for general LLMs:</strong> one unified objective that scales cleanly and naturally matches open-ended generation.</li>
</ul>
<h3>Complexity and efficient attention</h3>
<p>Standard attention is <code>O(n^2 * d)</code> in sequence length <code>n</code>, because you materialize the full pairwise attention matrix.</p>
<ul>
  <li><strong>FlashAttention</strong> doesn't reduce FLOPs -- it's IO-aware: it avoids materializing the full n&times;n matrix in slow GPU memory by tiling and fusing softmax + matmul, cutting memory reads/writes drastically.</li>
  <li>Other approaches: sparse attention (Longformer, BigBird), linear attention (approximate softmax for O(n) complexity, some quality trade-off).</li>
</ul>
<h3>Code</h3>
<p class="code-label">Minimal PyTorch implementation of both pieces above:</p>
<pre><code>import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (batch, heads, seq_len, d_k)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return weights @ v, weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        def split_heads(t):
            return t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        out, _ = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)

def causal_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len)).bool()</code></pre>
''',
        "quiz": [
            {"q": "Why divide QK^T by sqrt(d_k) before the softmax?", "a": "Dot-product variance grows with d_k, pushing softmax into saturated regions with near-zero gradients. Dividing by sqrt(d_k) keeps logits in a well-behaved range."},
            {"q": "What does multi-head attention give you that a single attention head doesn't?", "a": "Several attention operations run in parallel on lower-dimensional projections, letting the model attend to different relationship types (e.g. syntactic vs. semantic) simultaneously instead of averaging into one pattern."},
            {"q": "Why did decoder-only architectures win out for general-purpose LLMs over encoder-decoder?", "a": "One unified next-token-prediction objective that scales cleanly and naturally matches open-ended generation, vs. needing separate bidirectional-encoder and cross-attending-decoder machinery."},
            {"q": "Does FlashAttention reduce the number of FLOPs in attention?", "a": "No -- it's IO-aware, not FLOP-aware. It avoids materializing the full n×n attention matrix in slow GPU memory by tiling and fusing softmax+matmul, cutting memory reads/writes."},
        ],
    },
    {
        "id": "positional-encoding",
        "category": "foundations",
        "title": "Positional Encoding",
        "html": '''
<p>Attention itself is permutation-invariant -- it has no built-in sense of token order -- so position information has to be injected explicitly.</p>
<ul>
  <li><strong>Sinusoidal (original Transformer):</strong> fixed functions of position using sine/cosine at different frequencies. In theory generalizes to unseen lengths.</li>
  <li><strong>Learned absolute embeddings (GPT-2, BERT):</strong> a trainable embedding per position index. Doesn't generalize past the trained max length.</li>
  <li><strong>RoPE (Rotary Position Embedding):</strong> rotates Q and K by an angle proportional to position. The dot product between rotated Q and K depends only on their <em>relative</em> position. Used in LLaMA, GPT-NeoX, Mistral, and most modern open-weight LLMs.</li>
  <li><strong>ALiBi:</strong> adds a fixed linear penalty to attention scores proportional to token distance instead of modifying Q/K. Extrapolates well to longer sequences.</li>
</ul>
<h3>Code</h3>
<p class="code-label">Sinusoidal encoding and a RoPE apply function:</p>
<pre><code>import torch

def sinusoidal_positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    i = torch.arange(d_model // 2).unsqueeze(0)
    angle_rates = 1 / (10000 ** (2 * i / d_model))
    angles = pos * angle_rates
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe

def apply_rope(x, positions):
    # x: (..., seq_len, d) with d even
    d = x.shape[-1]
    freqs = 1.0 / (10000 ** (torch.arange(0, d, 2).float() / d))
    angles = positions[:, None] * freqs[None, :]
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x_rotated = torch.stack(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1
    )
    return x_rotated.flatten(-2)</code></pre>
''',
        "quiz": [
            {"q": "Why does attention need an explicit positional encoding at all?", "a": "Attention is permutation-invariant by construction -- it has no built-in sense of token order -- so position info must be injected explicitly."},
            {"q": "What's the key property RoPE gives you that learned absolute embeddings don't?", "a": "The dot product between rotated Q and K depends only on the *relative* position between tokens, not their absolute positions -- and it's used across most modern open-weight LLMs (LLaMA, GPT-NeoX, Mistral)."},
            {"q": "How does ALiBi encode position, and what's its main advantage?", "a": "It adds a fixed linear penalty to attention scores proportional to token distance, instead of modifying Q/K -- and it extrapolates well to sequences longer than seen in training."},
        ],
    },
    {
        "id": "tokenization",
        "category": "foundations",
        "title": "Tokenization",
        "html": '''
<ul>
  <li><strong>BPE (byte-pair encoding):</strong> start from characters/bytes, iteratively merge the most frequent adjacent pair into a new token, until reaching target vocab size.</li>
  <li><strong>WordPiece (BERT):</strong> similar merging idea but chooses merges by likelihood improvement rather than raw frequency.</li>
  <li><strong>SentencePiece:</strong> treats raw text (including whitespace) as a byte stream, language-agnostic.</li>
  <li><strong>Byte-level BPE (GPT-2/3):</strong> operates on raw bytes, so there's no out-of-vocabulary problem at all.</li>
  <li><strong>Vocab size trade-off:</strong> larger vocab &rarr; shorter sequences but bigger embedding/output matrices, and rarer tokens seen less often.</li>
</ul>
<h3>Code</h3>
<p class="code-label">A toy BPE merge loop over a tiny corpus:</p>
<pre><code>from collections import Counter

def get_pair_counts(tokens):
    pairs = Counter()
    for word, freq in tokens.items():
        symbols = word.split()
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs

def merge_pair(pair, tokens):
    merged = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in tokens.items():
        merged[word.replace(bigram, replacement)] = freq
    return merged

tokens = {"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6}
for step in range(10):
    pairs = get_pair_counts(tokens)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    tokens = merge_pair(best, tokens)</code></pre>
''',
        "quiz": [
            {"q": "What's the core BPE algorithm, in one sentence?", "a": "Start from characters/bytes, iteratively merge the most frequent adjacent pair into a new token, until reaching a target vocab size."},
            {"q": "Why does byte-level BPE (GPT-2/3) have no out-of-vocabulary problem?", "a": "It operates directly on raw bytes, so any input -- no matter how unusual -- can always be represented as some sequence of byte tokens."},
            {"q": "What's the trade-off in choosing a larger vocab size?", "a": "Larger vocab means shorter sequences, but bigger embedding/output matrices and rarer tokens get seen less often during training."},
        ],
    },
    {
        "id": "pretraining-objectives",
        "category": "foundations",
        "title": "Pretraining Objectives",
        "html": '''
<ul>
  <li><strong>Causal language modeling:</strong> predict the next token given all previous tokens, cross-entropy loss averaged over the sequence. The objective behind essentially all modern general-purpose LLMs.</li>
  <li><strong>Masked language modeling (BERT-style):</strong> randomly mask ~15% of tokens, predict them from bidirectional context.</li>
  <li><strong>Span corruption / denoising (T5, BART):</strong> corrupt spans of text, train the model to reconstruct them.</li>
</ul>
<p>Know <em>why</em> causal LM became the default: it directly matches the generation use case and needs no special masking scheme.</p>
<h3>Code</h3>
<p class="code-label">Causal LM loss with the standard shift-by-one:</p>
<pre><code>import torch.nn.functional as F

def causal_lm_loss(logits, input_ids):
    # logits: (batch, seq_len, vocab), input_ids: (batch, seq_len)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )</code></pre>
''',
        "quiz": [
            {"q": "Why did causal language modeling become the default pretraining objective for general-purpose LLMs?", "a": "It directly matches the generation use case (predict next token, given all previous tokens) and needs no special masking scheme, unlike MLM or span corruption."},
            {"q": "How does masked language modeling (BERT-style) differ from causal LM?", "a": "MLM randomly masks ~15% of tokens and predicts them from bidirectional context, rather than predicting the next token from only preceding context."},
        ],
    },
    {
        "id": "scaling-laws",
        "category": "scaling",
        "title": "Scaling Laws",
        "videos": [
            {"title": 'Chinchilla Explained: Compute-Optimal Massive Language Models', "url": 'https://www.youtube.com/watch?v=PZXN7jm9IC0'},
        ],
        "html": '''
<ul>
  <li><strong>Kaplan et al. (2020):</strong> smooth power-law relationships between loss and model size, data, and compute. Suggested growing model size and undertraining relative to what we now think is optimal.</li>
  <li><strong>Chinchilla (Hoffmann et al., 2022):</strong> showed most large models were undertrained. Compute-optimal training scales parameters and tokens together, around <strong>~20 tokens per parameter</strong>.</li>
</ul>
<p>Practical implication: given fixed compute, there's a specific optimal split between model size and data -- getting it wrong wastes compute either way.</p>
<h3>Post-training scaling is a different, less-settled question</h3>
<p>Kaplan/Chinchilla are both about <em>pretraining</em>: how to spend a compute budget on next-token prediction over a huge corpus. They say almost nothing about how RL/SFT post-training compute and data quantity scale -- a much newer and less-studied question, and one where a compute-constrained lab (rented clusters, not ten thousand GPUs) can still contribute, because the interesting regime is "what works at the scale we can actually afford," not "what happens if you 100x it."</p>
<ul>
  <li>Post-training returns tend to bend much earlier than pretraining's -- past a point, more RL steps or more preference pairs on the same base model stop helping and can even hurt (reward hacking, mode collapse from over-optimizing a proxy).</li>
  <li>Data <em>quality</em> and task relevance often beat raw parameter count in the post-training regime -- a smaller model post-trained on carefully curated, on-distribution data can beat a larger model post-trained on generic data.</li>
  <li>This is exactly the open research question a compute-constrained team can own: characterizing where the returns bend, across a few open-weight model sizes, is itself a publishable result -- you don't need frontier-lab compute to say something true about it.</li>
</ul>
<h3>Code</h3>
<p class="code-label">A quick compute-optimal token count estimate:</p>
<pre><code>def chinchilla_optimal_tokens(n_params, tokens_per_param=20):
    """Rough compute-optimal training tokens for a given parameter count."""
    return n_params * tokens_per_param

print(chinchilla_optimal_tokens(7e9))   # ~1.4e11 tokens for a 7B model</code></pre>
''',
        "quiz": [
            {"q": "What was Chinchilla's key finding versus the earlier Kaplan et al. scaling laws?", "a": "Most large models were undertrained relative to their size -- compute-optimal training scales parameters and tokens together, at roughly ~20 tokens per parameter."},
            {"q": "Given a fixed compute budget, why can't you just make the model as big as possible?", "a": "There's a specific compute-optimal split between model size and training data; growing the model while undertraining it (as pre-Chinchilla models did) wastes compute just as much as the reverse."},
            {"q": "Why do Kaplan/Chinchilla scaling laws not directly answer how post-training compute should scale?", "a": "They characterize pretraining (next-token prediction over a huge corpus); post-training (RL/SFT) compute and data scaling is a distinct, much less-studied question where returns tend to bend earlier and data quality often matters more than raw parameter count."},
        ],
    },
    {
        "id": "optimization",
        "category": "scaling",
        "title": "Optimization & Training Mechanics",
        "html": '''
<ul>
  <li><strong>AdamW:</strong> Adam with <em>decoupled</em> weight decay. Standard optimizer for LLM training.</li>
  <li><strong>LR schedule:</strong> linear or cosine warmup followed by decay -- warmup avoids instability from large early gradients.</li>
  <li><strong>Gradient clipping:</strong> caps gradient norm to prevent occasional large gradients from destabilizing the run.</li>
  <li><strong>Mixed precision:</strong> bf16 is generally preferred over fp16 at scale -- same exponent range as fp32, so it doesn't overflow/underflow as easily.</li>
</ul>
<p><strong>Distributed training strategies</strong> (know the difference -- commonly asked):</p>
<ul>
  <li><em>Data parallelism:</em> full model replicated, data sharded, gradients synced each step.</li>
  <li><em>Tensor parallelism:</em> shard individual matmuls across devices (Megatron-style).</li>
  <li><em>Pipeline parallelism:</em> different layers on different devices, microbatches pipelined through.</li>
  <li><em>ZeRO (DeepSpeed):</em> shards optimizer states/gradients/params across data-parallel workers.</li>
</ul>
<h3>Code</h3>
<p class="code-label">Warmup + cosine decay, gradient clipping, and bf16 autocast in one training step:</p>
<pre><code>import math
import torch

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

def cosine_warmup_lr(step, warmup_steps, total_steps, base_lr):
    if step &lt; warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))

for step, batch in enumerate(dataloader):
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(batch).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    for g in optimizer.param_groups:
        g["lr"] = cosine_warmup_lr(step, 2000, 100_000, 3e-4)
    optimizer.step()</code></pre>
''',
        "quiz": [
            {"q": "Why is bf16 generally preferred over fp16 for large-scale training?", "a": "bf16 has the same exponent range as fp32, so it doesn't overflow/underflow as easily as fp16, which has a narrower exponent range."},
            {"q": "What's the difference between tensor parallelism and pipeline parallelism?", "a": "Tensor parallelism shards individual matmuls across devices (Megatron-style); pipeline parallelism puts different layers on different devices and pipelines microbatches through them."},
            {"q": "What does ZeRO (DeepSpeed) actually shard?", "a": "Optimizer states, gradients, and parameters across data-parallel workers -- reducing the per-GPU memory footprint of otherwise-replicated training state."},
            {"q": "Why use an LR warmup at the start of training?", "a": "Large early gradients (before the model has settled at all) can destabilize training; a linear or cosine warmup ramps the LR up gradually to avoid that."},
        ],
    },
    {
        "id": "finetuning-alignment",
        "category": "scaling",
        "title": "Fine-Tuning & Alignment",
        "videos": [
            {"title": 'Reinforcement Learning from Human Feedback (RLHF) - Explained in 10 minutes', "url": 'https://www.youtube.com/watch?v=GSqZ8oQ6s50'},
            {"title": 'Direct Preference Optimization (DPO) explained: Bradley-Terry model, log probabilities, math', "url": 'https://www.youtube.com/watch?v=hvGa5Mba4c8'},
        ],
        "html": '''
<ul>
  <li><strong>SFT (supervised fine-tuning):</strong> fine-tune on curated instruction/response pairs -- teaches format and instruction-following.</li>
</ul>
<p><strong>RLHF pipeline</strong> (know all three stages cold):</p>
<ol>
  <li>Start from an SFT model.</li>
  <li>Train a <strong>reward model</strong> on human preference comparisons.</li>
  <li>Optimize the policy against the reward model using <strong>PPO</strong>, with a KL penalty against the reference model.</li>
</ol>
<ul>
  <li><strong>DPO (Direct Preference Optimization):</strong> reformulates RLHF as a direct classification loss over preference pairs -- no separate reward model, no RL loop.</li>
  <li><strong>Constitutional AI / RLAIF:</strong> replaces human preference labels with AI-generated feedback against a written "constitution," reducing the human-labeling bottleneck.</li>
  <li><strong>Reward hacking:</strong> the policy exploits reward model weaknesses. Mitigations: KL penalties, reward ensembling, or verifiable rewards instead of a learned reward model.</li>
</ul>
<h3>Code</h3>
<p class="code-label">The DPO loss, in full:</p>
<pre><code>import torch.nn.functional as F

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    policy_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = beta * (policy_logratios - ref_logratios)
    return -F.logsigmoid(logits).mean()</code></pre>
''',
        "quiz": [
            {"q": "What are the three stages of the RLHF pipeline, in order?", "a": "(1) Start from an SFT model. (2) Train a reward model on human preference comparisons. (3) Optimize the policy against the reward model with PPO, using a KL penalty against the reference model."},
            {"q": "What does DPO remove from the RLHF recipe, and how?", "a": "It removes the separate reward model and the RL loop entirely, reformulating preference optimization as a direct classification loss over (chosen, rejected) pairs."},
            {"q": "What is reward hacking, and name two mitigations.", "a": "The policy exploits weaknesses in the reward model rather than genuinely improving. Mitigations: KL penalties against the reference policy, reward ensembling, or using verifiable (ground-truth) rewards instead of a learned reward model."},
        ],
    },
    {
        "id": "peft",
        "category": "scaling",
        "title": "Parameter-Efficient Fine-Tuning (PEFT)",
        "videos": [
            {"title": 'LoRA: Low-Rank Adaptation of LLMs Explained', "url": 'https://www.youtube.com/watch?v=_K3HgjnRHCY'},
        ],
        "html": '''
<p>Full fine-tuning updates every parameter, which is expensive in compute and -- critically -- memory, since you need gradients and optimizer state for every trainable parameter. PEFT freezes most of the pretrained weights and trains only a small number of additional parameters.</p>
<h3>LoRA (Low-Rank Adaptation)</h3>
<p>For a frozen weight matrix <code>W</code>, learn a low-rank decomposition and add it: <code>W' = W + BA</code>, where <code>B</code> is <code>(d &times; r)</code> and <code>A</code> is <code>(r &times; d)</code> with <code>r &lt;&lt; d</code>. Only <code>A</code> and <code>B</code> are trained; <code>W</code> stays frozen.</p>
<ul>
  <li><strong>Why it works:</strong> the weight <em>updates</em> needed to adapt a pretrained model have low "intrinsic rank" -- you don't need a full-rank update to capture most of the useful adaptation.</li>
  <li><strong>Inference cost:</strong> <code>BA</code> can be merged back into <code>W</code>, so zero added latency if merged.</li>
  <li><strong>Merged vs. kept separate:</strong> keeping adapters separate lets you hot-swap customer- or task-specific adapters on one shared frozen base model -- relevant to a multi-tenant enterprise product.</li>
</ul>
<h3>QLoRA</h3>
<p>LoRA plus 4-bit quantization (NF4) of the frozen base model, plus paged optimizers to absorb memory spikes -- lets you fine-tune very large models on a single GPU.</p>
<h3>Prefix tuning / prompt tuning &amp; adapters</h3>
<p>Prefix/prompt tuning learns virtual tokens or per-layer key/value prefixes with the whole model frozen -- cheaper but lower capacity ceiling than LoRA. Adapters insert small trainable bottleneck layers between transformer layers -- predates LoRA, can't be merged away for free, so it adds a little inference latency.</p>
<h3>Code</h3>
<p class="code-label">A LoRA-wrapped linear layer:</p>
<pre><code>import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.B = nn.Parameter(torch.zeros(out_f, r))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling</code></pre>
''',
        "quiz": [
            {"q": "What's the core LoRA idea, in one line?", "a": "Freeze the pretrained weight matrix W and learn a low-rank update W' = W + BA (B is d×r, A is r×d, r << d), training only A and B."},
            {"q": "Why does LoRA work despite only learning a low-rank update?", "a": "The weight *updates* needed to adapt a pretrained model to a new task have low 'intrinsic rank' -- you don't need a full-rank update to capture most of the useful adaptation."},
            {"q": "What does QLoRA add on top of LoRA?", "a": "4-bit quantization (NF4) of the frozen base model, plus paged optimizers to absorb memory spikes -- letting you fine-tune very large models on a single GPU."},
            {"q": "Why might you keep a LoRA adapter separate rather than merging it into the base weights?", "a": "Keeping adapters separate lets you hot-swap customer- or task-specific adapters on one shared frozen base model, useful for multi-tenant serving."},
        ],
    },
    {
        "id": "rl-foundations-study",
        "category": "reasoning",
        "title": "RL Foundations",
        "videos": [
            {"title": 'PPO | Proximal Policy Optimization (PPO) architecture | PPO Explained', "url": 'https://www.youtube.com/watch?v=7lDX7NZX94g'},
            {"title": "GRPO: How DeepSeek R1's Reinforcement Learning Works", "url": 'https://www.youtube.com/watch?v=90ImcYM0xWc'},
        ],
        "html": '''
<p>You should be fluent in these even if the interview stays high-level, since agentic RL questions build directly on them.</p>
<ul>
  <li><strong>MDP:</strong> states, actions, transition probabilities, reward function, discount factor <code>gamma</code>.</li>
  <li><strong>Policy</strong> <code>pi(a|s)</code>: maps states to a distribution over actions.</li>
  <li><strong>Value functions:</strong> <code>V(s)</code> and <code>Q(s,a)</code> give expected return from a state (and action).</li>
  <li><strong>REINFORCE:</strong> basic policy gradient -- high variance. Subtract a <strong>baseline</strong> (value function) to get the <strong>advantage</strong> <code>A(s,a) = Q(s,a) - V(s)</code>.</li>
  <li><strong>PPO:</strong> clips the probability ratio between new and old policy so updates can't move too far in one step -- the dominant algorithm in RLHF pipelines.</li>
  <li><strong>Actor-critic:</strong> train a policy (actor) and value function (critic) jointly.</li>
  <li><strong>GRPO (DeepSeek):</strong> drops PPO's learned value network entirely. Sample a <em>group</em> of completions for the same prompt, normalize each one's reward against its own group's mean/std to get the advantage, then apply the same clipped objective as PPO plus a KL penalty against a reference policy. No critic to train means no critic to get wrong -- simpler and cheaper, at the cost of needing several samples per prompt.</li>
</ul>
<h3>Code</h3>
<p class="code-label">Policy gradient with a baseline, the PPO clipped objective, and GRPO's group-relative advantage:</p>
<pre><code>import torch

def policy_gradient_loss(log_probs, rewards, baseline):
    advantages = rewards - baseline
    return -(log_probs * advantages.detach()).mean()

def ppo_clip_loss(new_log_probs, old_log_probs, advantages, epsilon=0.2):
    ratio = (new_log_probs - old_log_probs).exp()
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    return -torch.min(unclipped, clipped).mean()

def grpo_advantage(rewards, eps=1e-8):
    # rewards: (num_groups, group_size) -- same prompt, multiple sampled completions
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + eps)</code></pre>
<p class="callout">You've already implemented all of these from scratch (with tests) in the exercises above -- discounted_return, GAE, REINFORCE, PPO-clip, actor-critic loss, GRPO advantage, GRPO loss.</p>
''',
        "quiz": [
            {"q": "Why does subtracting a baseline from the return reduce variance in REINFORCE, without introducing bias?", "a": "The baseline (e.g. V(s)) doesn't depend on the action taken, so E[baseline * grad log pi] = 0 in expectation -- it cancels out on average but reduces variance of any single sample."},
            {"q": "What does PPO's clipping actually prevent?", "a": "It caps how far the probability ratio between new and old policy can move in one update, preventing destructively large policy updates from a single batch of data."},
            {"q": "What's the difference between the advantage A(s,a) and the raw return?", "a": "A(s,a) = Q(s,a) - V(s) measures how much better an action is than the average action from that state, rather than the raw (high-variance) total return."},
            {"q": "What does GRPO remove from PPO, and what does it need instead?", "a": "It removes the learned value/critic network entirely, computing the advantage by normalizing each sampled completion's reward against the mean/std of a group of completions for the same prompt -- which means it needs several samples per prompt instead of a trained critic."},
        ],
    },
    {
        "id": "verifiable-rewards",
        "category": "reasoning",
        "title": "RL for Agents — Verifiable Rewards",
        "videos": [
            {"title": '9 Examples of Specification Gaming', "url": 'https://www.youtube.com/watch?v=nKJlF-olKmg'},
        ],
        "html": '''
<ul>
  <li><strong>RLVR (Reinforcement Learning from Verifiable Rewards):</strong> reward comes from a ground-truth checkable outcome -- code passes a test suite, a migration builds and passes the regression suite -- rather than a learned reward model. Removes the reward-model-hacking failure mode, but the verifier itself can still be gamed, and it only works where verifiability exists in the first place.</li>
  <li><strong>Verification is not immune to hacking, it just moves the target:</strong> a classic case -- an agent asked to make a regression suite pass discovers it can delete, disable, or simply not exercise the branch the tests don't cover, rather than fixing the underlying code. The reward signal was "ground truth" (the tests genuinely passed) and still didn't measure what you wanted. Designing a reward that resists this without becoming a soft, gameable proxy again is the actual hard part of RLVR, not the RL algorithm on top of it.</li>
  <li><strong>Verification without a test at all:</strong> most real changes in a legacy system aren't covered by any test. Options: semantic/behavioral diffing (run old and new versions on the same inputs and compare outputs), shadow deployment (run the new version alongside the old on live traffic and compare), LLM-as-judge over a structured rubric, or a second agent whose only job is to argue against the change. Each is a different way of manufacturing a verification signal where none was given -- and each has its own failure mode (an LLM judge can be fooled the same way a learned reward model can).</li>
  <li><strong>Sparse reward / credit assignment</strong> in long-horizon tasks:
    <ul>
      <li>Reward shaping (risk: can distort the true objective)</li>
      <li>Hierarchical RL (decompose into subgoals)</li>
      <li><strong>Process reward models (PRMs)</strong> -- score each step, denser but costlier</li>
      <li><strong>Outcome reward models (ORMs)</strong> -- score only the final result, simpler but noisier</li>
    </ul>
  </li>
</ul>
<p>Be ready to reason out loud about a concrete verifier design problem (e.g. "how would you define 'equivalent behavior' for a migrated enterprise system?") -- more likely a discussion prompt than a memorized answer.</p>
<h3>Code</h3>
<p class="code-label">A minimal verifiable reward -- ground truth, not a learned model -- and why "ground truth" still isn't the whole story:</p>
<pre><code>import subprocess

def verifiable_code_reward(candidate_code: str, test_file: str) -> float:
    """1.0 if the candidate patch passes the test suite, else 0.0.

    This is exploitable: an agent can satisfy it by deleting the code
    path the tests don't reach, not just by writing a correct patch.
    Verifiable != un-hackable -- it just requires a smarter exploit.
    """
    with open("candidate.py", "w") as f:
        f.write(candidate_code)
    result = subprocess.run(
        ["pytest", test_file, "-q"], capture_output=True, timeout=30
    )
    return 1.0 if result.returncode == 0 else 0.0</code></pre>
''',
        "quiz": [
            {"q": "Does a verifiable reward (like \"tests pass\") mean the reward can't be hacked?", "a": "No -- it removes reward-MODEL hacking specifically, but the verifier itself can still be gamed, e.g. an agent passing a regression suite by deleting or not exercising the code branch the tests don't cover, rather than fixing the underlying issue."},
            {"q": "Name two ways to get a verification signal for a change that no automated test covers.", "a": "Semantic/behavioral diffing (compare old vs. new outputs on the same inputs), shadow deployment (run both versions on live traffic and compare), LLM-as-judge over a rubric, or an adversarial second agent arguing against the change."},
            {"q": "What's the difference between a process reward model (PRM) and an outcome reward model (ORM)?", "a": "A PRM scores each intermediate step (denser signal, more costly to obtain); an ORM scores only the final result (simpler, but noisier credit assignment over long horizons)."},
        ],
    },
    {
        "id": "test-time-compute",
        "category": "reasoning",
        "title": "Test-Time Compute & Reasoning Techniques",
        "html": '''
<p>A major recent shift: instead of only improving models via more pretraining compute, spend more compute <em>at inference time</em> to get a better answer.</p>
<ul>
  <li><strong>Chain-of-thought (CoT):</strong> intermediate reasoning steps before the final answer, measurably improving multi-step tasks with no extra training.</li>
  <li><strong>Self-consistency:</strong> sample multiple CoT paths, take a majority vote -- reduces variance from any single path going wrong.</li>
  <li><strong>Best-of-N:</strong> generate N candidates, use a verifier to score and keep the best. A cheap verifier makes this a cheap way to improve quality.</li>
  <li><strong>Tree search (Tree-of-Thoughts):</strong> explore a branching tree of partial reasoning states instead of one chain.</li>
  <li><strong>Explicit reasoning models (o1/o3-style):</strong> trained via RL to produce long internal reasoning traces -- the reasoning behavior itself is learned, not imposed by prompting.</li>
</ul>
<h3>Code</h3>
<p class="code-label">Self-consistency (majority vote) and best-of-N (verifier-scored):</p>
<pre><code>from collections import Counter

def self_consistency(model, prompt, n_samples=10):
    answers = [
        extract_final_answer(model.generate(prompt, temperature=0.7))
        for _ in range(n_samples)
    ]
    return Counter(answers).most_common(1)[0][0]

def best_of_n(model, prompt, verifier, n_samples=8):
    candidates = [model.generate(prompt, temperature=1.0) for _ in range(n_samples)]
    scored = [(c, verifier(c)) for c in candidates]
    return max(scored, key=lambda pair: pair[1])[0]</code></pre>
''',
        "quiz": [
            {"q": "What's the core idea behind spending more compute at test time, vs. pretraining time?", "a": "Instead of only improving the model via more pretraining compute, you spend more compute at inference (sampling more, searching more, reasoning longer) to get a better answer for a given query."},
            {"q": "How does self-consistency differ from simply sampling once with chain-of-thought?", "a": "It samples multiple independent CoT paths and takes a majority vote over their final answers, reducing variance from any single reasoning path going wrong."},
            {"q": "What's the key difference between o1/o3-style reasoning models and prompting a base model with 'think step by step'?", "a": "The reasoning behavior itself is learned via RL to produce long internal reasoning traces, rather than being imposed externally through a prompt."},
        ],
    },
    {
        "id": "agent-architectures",
        "category": "reasoning",
        "title": "Agent Architectures",
        "videos": [
            {"title": 'ReAct AI Agents, clearly explained!', "url": 'https://www.youtube.com/watch?v=vFdIrZyKEwQ'},
        ],
        "html": '''
<ul>
  <li><strong>Tool use / function calling:</strong> the model emits a structured call, the system executes it, the result is inserted back into context.</li>
  <li><strong>ReAct:</strong> interleaves explicit reasoning ("thought") with actions and observations.</li>
  <li><strong>Reflexion / self-refinement:</strong> the agent critiques its own trajectory in natural language and retries -- behavior improves within an episode via better context, not weight updates.</li>
  <li><strong>Planning:</strong> plan-then-execute vs. reactive step-by-step; long-horizon tasks often need a hierarchical mix.</li>
  <li><strong>Memory systems:</strong> short-term (context window), long-term (retrieval), and working memory/scratchpad -- this is deep enough to have its own section; see Agent Memory Architectures.</li>
  <li><strong>Multi-agent systems:</strong> orchestrator/worker patterns, critique/debate between agents, division of labor -- see Long-Horizon Agent Reliability for how these fail.</li>
</ul>
<h3>Code</h3>
<p class="code-label">A minimal ReAct-style tool-use loop:</p>
<pre><code>def react_agent(model, tools, question, max_steps=6):
    scratchpad = f"Question: {question}\\n"
    for _ in range(max_steps):
        step = model.generate(scratchpad + "Thought:")
        thought, action, action_input = parse_step(step)
        if action == "Final Answer":
            return action_input
        observation = tools[action](action_input)
        scratchpad += (
            f"Thought: {thought}\\nAction: {action}[{action_input}]\\n"
            f"Observation: {observation}\\n"
        )
    return None</code></pre>
''',
        "quiz": [
            {"q": "What does ReAct interleave that plain chain-of-thought doesn't?", "a": "Explicit reasoning ('thought') steps with concrete actions and their observations, rather than reasoning purely in text before a single final answer."},
            {"q": "How does Reflexion improve an agent's behavior without any weight updates?", "a": "The agent critiques its own trajectory in natural language and retries -- improvement comes from better context on the next attempt, not from gradient updates."},
        ],
    },
    {
        "id": "agent-memory",
        "category": "reasoning",
        "title": "Agent Memory Architectures",
        "videos": [
            {"title": "MemGPT Explained!", "url": "https://www.youtube.com/watch?v=nQmZmFERmrg"},
            {"title": "Generative Agents: Interactive Simulacra of Human Behavior - Joon Sung Park (Stanford)", "url": "https://www.youtube.com/watch?v=XY5Wncq5vAE"},
        ],
        "html": '''
<p class="callout">This is one of the least-solved problems in agentic AI, and one of the most consequential for any system that runs across many steps and days of wall-clock time rather than a single context window's worth of work.</p>
<h3>Why a context window isn't enough</h3>
<p>A run spanning forty steps and multiple days produces far more text than fits in any context window, and even when it technically fits, stuffing everything in degrades attention over the genuinely relevant parts. Memory is the answer to: what persists across steps, how is it structured and retrieved, and how does it get revised when a later step reveals that an earlier belief about the world was wrong?</p>
<h3>Memory operations, not just storage</h3>
<ul>
  <li><strong>Write:</strong> what gets committed to memory in the first place -- raw observations, or a compressed/summarized form?</li>
  <li><strong>Read / retrieve:</strong> similarity search is the default, but recency and importance matter too (see Generative Agents below) -- the most similar memory isn't always the most useful one right now.</li>
  <li><strong>Revise:</strong> the hard one. If step 3 wrote "the billing service owns invoicing" and step 30 discovers that's no longer true, does the agent overwrite, append a correction, or keep both and reason about which is current? Most systems today don't do this well -- they retrieve stale beliefs alongside corrections and hope the model sorts it out.</li>
  <li><strong>Forget / consolidate:</strong> without this, retrieval quality degrades as the memory store grows -- not every observation deserves to be kept at full fidelity forever.</li>
</ul>
<h3>Two concrete architectures</h3>
<ul>
  <li><strong>MemGPT</strong> treats the LLM like an OS process: a small "main context" (system instructions, recent conversation, a scratchpad) is what the model actually sees, and it can issue function calls to page information in and out of a much larger "external context" (archival storage) -- the model manages its own memory hierarchy rather than a fixed pipeline managing it for the model.</li>
  <li><strong>Generative Agents (Park et al., Stanford)</strong> keep a flat memory stream of natural-language observations, retrieved by a weighted mix of <em>recency</em> (exponential decay), <em>importance</em> (a self-assessed score), and <em>relevance</em> (embedding similarity) -- and periodically generate <strong>reflections</strong>, higher-level abstractions synthesized from clusters of related raw observations, which is what lets the agent generalize instead of drowning in undifferentiated raw events.</li>
</ul>
<h3>The part that's actually unsolved: training a model to use memory</h3>
<p>Giving an agent a memory API doesn't mean it uses it well -- models routinely ignore available context, retrieve the wrong thing, or fail to write down something they'll need later. Training a model to <em>use</em> memory rather than ignore it looks less like a retrieval-engineering problem and more like a training-signal problem: reward trajectories where memory use demonstrably changed the outcome, or construct training data where the correct action is only inferable from something written many steps earlier.</p>
<h3>The evidence bar is high</h3>
<p>The open question isn't "can you build a memory system" -- it's showing, with evidence rather than anecdote, that a structured memory architecture beats simply stuffing everything into as large a context window as you can afford. That comparison is a real experiment, not an assumption.</p>
''',
        "quiz": [
            {"q": "What's the difference between memory retrieval and memory revision, and why is revision the harder problem?", "a": "Retrieval is finding relevant stored information; revision is updating or correcting memory when a later step reveals an earlier belief about the world was wrong. Most systems retrieve stale and corrected beliefs side by side and rely on the model to sort out which is current -- there's no standard mechanism for actually reconciling them."},
            {"q": "In MemGPT, what plays the role of an OS's RAM vs. disk?", "a": "The 'main context' (system instructions, recent conversation, a scratchpad) is the RAM-equivalent that the model directly sees; 'external context' is disk-equivalent archival storage the model pages information into and out of via explicit function calls."},
            {"q": "What three factors does the Generative Agents memory stream weight when retrieving a memory?", "a": "Recency (exponential decay), importance (a self-assessed score), and relevance (embedding similarity to the current situation) -- combined, not just similarity search alone."},
            {"q": "Why is 'the agent has a memory system' not itself sufficient evidence that memory is helping?", "a": "The bar is showing, with a real comparison, that the structured memory architecture beats simply stuffing more into the context window -- a memory system that isn't demonstrably better than a bigger context window hasn't earned its complexity."},
        ],
    },
    {
        "id": "knowledge-representation",
        "category": "reasoning",
        "title": "Knowledge Representation & Structured Reasoning",
        "videos": [
            {"title": "GraphRAG Explained: AI Retrieval with Knowledge Graphs & Cypher", "url": "https://www.youtube.com/watch?v=Za7aG-ooGLQ"},
        ],
        "html": '''
<p>Flat retrieval over chunks of text treats every document as roughly independent. That's a mismatch for a domain that's fundamentally about <em>relationships</em> -- which service calls which, which table feeds which report, which process step depends on which upstream decision. Representing that structure explicitly, rather than hoping similarity search reconstructs it implicitly, is its own research problem.</p>
<h3>Knowledge graphs and ontologies</h3>
<ul>
  <li><strong>Knowledge graph:</strong> entities (services, tables, process steps) and typed relations between them (calls, writes-to, depends-on), usually built by extracting structure from code, schemas, logs, and documentation rather than hand-authored.</li>
  <li><strong>Ontology:</strong> the schema itself -- what <em>types</em> of entities and relationships are even allowed to exist in this domain. An ontology is a claim about the shape of the domain, not just a specific graph instance of it; a good one generalizes across customers with genuinely different landscapes, a bad one overfits to the first customer you built it from.</li>
  <li><strong>Staying true as the system changes:</strong> a graph built once and never updated becomes actively misleading -- worse than no structure at all, because it's wrong with confidence. Keeping it current as the underlying systems evolve is a distinct, ongoing problem from constructing it the first time.</li>
</ul>
<h3>Graph-RAG vs. flat RAG</h3>
<p>Graph-RAG grounds retrieval in graph traversal -- follow the actual dependency edges from the thing you're changing -- rather than nearest-neighbor search over embeddings. For a question like "what breaks if I remove this field," the graph traversal is the correct algorithm; similarity search over documentation chunks is a lossy approximation of it.</p>
<h3>Neurosymbolic reasoning</h3>
<p>A neurosymbolic system combines a neural component (an LLM, for open-ended understanding and language) with a symbolic component (a graph, a solver, explicit rules, for precise and verifiable structure). The pitch: LLMs alone are fluent but not reliably precise about structure; symbolic systems alone are precise but can't parse a messy real-world enterprise landscape written in inconsistent natural language and legacy schemas. Combining them aims to get both properties at once -- though which parts of a given pipeline should be neural vs. symbolic is an open design question, not a settled recipe.</p>
<h3>This is a live, unresolved research question -- not a solved technique</h3>
<p>Whether grounding an agent in a learned ontology of a customer's landscape actually beats retrieval over raw artifacts is explicitly an open empirical question, not an established result to cite. Framing an answer here as "here's how you'd design the experiment to find out" is more honest, and more interesting, than asserting a conclusion the field hasn't earned yet.</p>
''',
        "quiz": [
            {"q": "What's the difference between a knowledge graph and an ontology?", "a": "A knowledge graph is a specific instance -- these entities, these relations; an ontology is the schema defining what types of entities and relationships are allowed to exist in the domain at all. The ontology should generalize across landscapes; a graph built from one customer's specifics might not."},
            {"q": "Why can an out-of-date knowledge graph be worse than no structure at all?", "a": "It's wrong with the appearance of authority -- an agent reasoning over stale structure will confidently act on relationships that no longer hold, which is a worse failure mode than having no structure and falling back to more cautious raw retrieval."},
            {"q": "Why might graph traversal beat embedding similarity search for a question like 'what breaks if I remove this field'?", "a": "That question is fundamentally about following actual dependency edges, which is exactly what graph traversal computes directly; similarity search over text chunks is only a lossy, indirect approximation of the same relationship."},
            {"q": "What's the core pitch behind combining neural and symbolic components (neurosymbolic AI)?", "a": "Neural components (LLMs) are fluent at parsing messy, inconsistent real-world text but not reliably precise about structure; symbolic components (graphs, solvers, rules) are precise but can't parse messy natural language input -- combining them aims to get both fluency and precision."},
        ],
    },
    {
        "id": "long-horizon-reliability",
        "category": "reasoning",
        "title": "Long-Horizon Agent Reliability",
        "videos": [
            {"title": "Why Do Multi-Agent LLM Systems Fail?", "url": "https://www.youtube.com/watch?v=GNpF_lUFRiE"},
        ],
        "html": '''
<p>A forty-step plan isn't just a longer version of a four-step plan -- reliability degrades in ways that only show up at length, and studying those failure modes (rather than just building longer chains and hoping) is its own research area.</p>
<h3>Error compounding is multiplicative, not additive</h3>
<p>If each step in a plan is independently correct 95% of the time, the odds the whole forty-step plan is correct end-to-end are roughly 0.95^40 &approx; 13% -- not 95%, and not "95% minus a bit." Small, individually-reasonable per-step error rates compound into near-certain failure over a long enough horizon unless something actively arrests the compounding: intermediate verification, checkpointing, or a mechanism to detect and recover from an error before it propagates twenty more steps.</p>
<h3>Planning under partial observability</h3>
<p>The agent doesn't start with a complete model of the landscape -- undocumented decisions, systems nobody fully understands anymore, edge cases nobody wrote down. This is a POMDP, not an MDP: the agent must act on an incomplete and sometimes wrong model of the world, discover where that model is wrong partway through, and <em>replan</em> rather than treating the original plan as fixed. "Recover when step twelve reveals the model of the world was wrong" is not an edge case to handle -- for a real enterprise landscape, it's closer to the median case.</p>
<h3>Multi-agent failure modes</h3>
<p>A systematic study of multi-agent LLM systems (Cemri et al., UC Berkeley, 2025) categorized failures across three groups: <strong>design and specification shortcomings</strong> (ambiguous role division, unclear termination conditions), <strong>inter-agent misalignment</strong> (agents talking past each other, one agent silently assuming context another doesn't have), and <strong>verification and termination failures</strong> (nobody checks the intermediate output, or the system doesn't know when to stop). Delegation without a real verification step between agents just relocates the reliability problem -- it doesn't solve it.</p>
<h3>Mitigations, roughly in order of how much they cost</h3>
<ul>
  <li>Checkpointing and rollback -- cheap, but only helps if you can detect the error before compounding it further.</li>
  <li>Intermediate verification (not just checking the final output) -- catches errors near where they happened, when they're cheapest to fix.</li>
  <li>Hierarchical decomposition -- bounds the blast radius of a single step's failure to its subtree rather than the whole plan.</li>
  <li>Explicit uncertainty signaling -- the agent (or another agent) flags low confidence and escalates rather than confidently proceeding on a guess.</li>
</ul>
''',
        "quiz": [
            {"q": "If each step of a 40-step plan is independently correct 95% of the time, what's the approximate end-to-end success rate, and what does that imply?", "a": "About 0.95^40 ≈ 13%. It implies per-step error rates that sound individually acceptable compound multiplicatively into near-certain failure over a long horizon, unless something actively arrests the compounding (verification, checkpointing, recovery)."},
            {"q": "Why is an enterprise transformation agent's planning problem a POMDP rather than an MDP?", "a": "The agent doesn't start with a complete, correct model of the landscape (undocumented decisions, poorly-understood legacy systems) -- it must act under partial observability, discover where its model is wrong partway through, and replan rather than executing a fixed plan."},
            {"q": "Name the three categories of multi-agent LLM failure from the Cemri et al. study.", "a": "Design and specification shortcomings, inter-agent misalignment, and verification/termination failures."},
            {"q": "Why doesn't delegating a task to a second agent for verification automatically solve the reliability problem?", "a": "It relocates the problem rather than solving it -- if the verifying agent isn't itself reliable, or the delegation protocol has the same ambiguity/misalignment failure modes, you now have two unreliable agents instead of one."},
        ],
    },
    {
        "id": "rag",
        "category": "reasoning",
        "title": "Retrieval-Augmented Generation (RAG)",
        "videos": [
            {"title": 'RAG Explained in 5 Minutes | The Secret Behind RAG (Retrieval-Augmented Generation)', "url": 'https://www.youtube.com/watch?v=rG6m8dkZlkg'},
        ],
        "html": '''
<p>RAG combines a parametric model with a non-parametric knowledge source so the model can access information beyond what's baked into its weights, without retraining.</p>
<ul>
  <li><strong>Basic pipeline:</strong> embed the query, retrieve top-k similar chunks, insert them into context, generate conditioned on query + evidence.</li>
  <li><strong>Why it matters:</strong> reduces hallucination, gives a citation mechanism, and scopes what the model can "know" to a specific data source.</li>
  <li><strong>Chunking:</strong> too small loses context, too large dilutes relevance. Overlapping windows and semantic chunking help.</li>
  <li><strong>Embeddings &amp; vector search:</strong> approximate nearest-neighbor search (HNSW, IVF) for speed at scale.</li>
  <li><strong>Re-ranking:</strong> retrieve cheaply with bi-encoders, re-rank top candidates with a more accurate cross-encoder.</li>
  <li><strong>Failure modes:</strong> recall failure (missed the right doc), precision failure (retrieved irrelevant content), or the model ignoring retrieved context anyway.</li>
</ul>
<h3>Code</h3>
<p class="code-label">A toy retrieval + prompt-construction pipeline:</p>
<pre><code>import numpy as np

def retrieve(query_emb, doc_embs, docs, k=3):
    sims = doc_embs @ query_emb / (
        np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    top_k = np.argsort(-sims)[:k]
    return [docs[i] for i in top_k]

def build_rag_prompt(query, retrieved_docs):
    context = "\\n\\n".join(retrieved_docs)
    return (
        f"Context:\\n{context}\\n\\n"
        f"Question: {query}\\nAnswer using only the context above:"
    )</code></pre>
''',
        "quiz": [
            {"q": "Name the two main failure modes of the retrieval step in RAG.", "a": "Recall failure (the right document was never retrieved) and precision failure (irrelevant content was retrieved) -- plus a third failure that isn't retrieval's fault: the model ignoring the retrieved context anyway."},
            {"q": "Why re-rank with a cross-encoder after retrieving with a bi-encoder, instead of just using the cross-encoder directly?", "a": "Bi-encoders are cheap enough to search over the whole corpus; cross-encoders are more accurate but too expensive to run on everything, so you retrieve cheaply first and re-rank only the top candidates."},
            {"q": "What's the chunking trade-off in a RAG pipeline?", "a": "Chunks too small lose surrounding context; chunks too large dilute relevance (more irrelevant text competing with the relevant part in the same chunk)."},
        ],
    },
    {
        "id": "evaluation-methodology",
        "category": "evaluation",
        "title": "Evaluation Methodology",
        "videos": [
            {"title": 'LLM as a Judge Evaluation Explained | Step-by-Step', "url": 'https://www.youtube.com/watch?v=Z5Eoap49xC4'},
        ],
        "html": '''
<ul>
  <li><strong>Perplexity:</strong> exponential of average negative log-likelihood on held-out text. Correlates only loosely with downstream quality.</li>
  <li><strong>Benchmark suites:</strong> MMLU, HellaSwag, GSM8K, HumanEval for general capability; SWE-bench, WebArena, OSWorld for agentic/long-horizon tasks.</li>
  <li><strong>LLM-as-judge:</strong> cheap and scalable, but has known biases -- verbosity, position, self-preference.</li>
  <li><strong>Human evaluation:</strong> gold standard, needs a well-specified rubric and measured inter-rater agreement.</li>
  <li><strong>Prediction-powered inference (PPI):</strong> combines a cheap, biased-but-plentiful signal with a small set of expensive gold labels for a tighter, statistically valid estimate.</li>
  <li><strong>Verifiable evaluation:</strong> equivalence checking, regression suites, shadow/canary deployment.</li>
</ul>
<h3>Code</h3>
<p class="code-label">Perplexity, and a simplified PPI point estimate:</p>
<pre><code>import torch

def perplexity(logits, labels):
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean"
    )
    return torch.exp(loss)

def ppi_estimate(judge_scores, gold_scores_subset, judge_scores_subset):
    """Correct the cheap judge's mean using the bias measured on a small gold subset."""
    bias = gold_scores_subset.mean() - judge_scores_subset.mean()
    return judge_scores.mean() + bias</code></pre>
''',
        "quiz": [
            {"q": "Name three known biases of LLM-as-judge evaluation.", "a": "Verbosity bias (favoring longer answers), position bias (favoring whichever answer is shown first/second), and self-preference bias (favoring outputs from the same model family as the judge)."},
            {"q": "What problem does prediction-powered inference (PPI) solve?", "a": "It combines a cheap, biased-but-plentiful signal (like an LLM judge) with a small set of expensive gold labels to produce a tighter, statistically valid estimate than either alone."},
            {"q": "Why does perplexity only loosely correlate with downstream quality?", "a": "It's a measure of how well the model predicts held-out text under its own training distribution, which doesn't necessarily track how useful or correct its outputs are on the actual tasks you care about."},
        ],
    },
    {
        "id": "inference-serving",
        "category": "evaluation",
        "title": "Inference & Serving Systems",
        "html": '''
<ul>
  <li><strong>KV cache:</strong> caching past keys/values avoids recomputing attention over the whole prefix at every decoding step -- the dominant memory cost at long context or large batch.</li>
  <li><strong>MQA &amp; GQA:</strong> MQA shares one K/V head across all query heads, shrinking the cache at some quality cost. GQA groups heads into clusters -- used in LLaMA 2/3, a better trade-off than full MQA.</li>
  <li><strong>Quantization:</strong> INT8/INT4 precision to cut memory and boost throughput.</li>
  <li><strong>Distillation:</strong> train a smaller student model to mimic a larger teacher.</li>
  <li><strong>Speculative decoding:</strong> a small draft model proposes tokens ahead; the large model verifies them in parallel and accepts the matching prefix.</li>
  <li><strong>Continuous batching / PagedAttention (vLLM):</strong> keeps GPU utilization high across requests of varying length.</li>
</ul>
<h3>Code</h3>
<p class="code-label">A minimal KV cache, and the shape of speculative decoding:</p>
<pre><code>import torch

class KVCache:
    def __init__(self):
        self.k_cache, self.v_cache = None, None

    def update(self, new_k, new_v):
        if self.k_cache is None:
            self.k_cache, self.v_cache = new_k, new_v
        else:
            self.k_cache = torch.cat([self.k_cache, new_k], dim=2)
            self.v_cache = torch.cat([self.v_cache, new_v], dim=2)
        return self.k_cache, self.v_cache

def speculative_decode(draft_model, target_model, prompt, n_draft=4):
    draft_tokens = draft_model.generate(prompt, max_new_tokens=n_draft)
    target_logits = target_model(prompt + draft_tokens).logits
    accepted = []
    for i, tok in enumerate(draft_tokens):
        if sample_from(target_logits[i]) == tok:
            accepted.append(tok)
        else:
            break
    return accepted</code></pre>
''',
        "quiz": [
            {"q": "Why does a KV cache matter so much for inference cost?", "a": "Without it, you'd recompute attention over the entire prefix at every decoding step; caching past keys/values avoids that recomputation and becomes the dominant memory cost at long context or large batch."},
            {"q": "What's the difference between MQA and GQA?", "a": "MQA shares a single K/V head across all query heads (smallest cache, biggest quality cost); GQA groups heads into clusters, a middle ground used in LLaMA 2/3."},
            {"q": "How does speculative decoding speed up generation without changing the output distribution?", "a": "A small draft model proposes several tokens ahead; the large target model verifies them all in one parallel forward pass and accepts the matching prefix, falling back to normal decoding only where they disagree."},
        ],
    },
    {
        "id": "extra-topics",
        "category": "extras",
        "title": "Extra Topics Sometimes Asked",
        "html": '''
<ul>
  <li><strong>Mixture-of-Experts (MoE):</strong> route each token to a small subset of "expert" FFNs instead of one dense FFN -- more parameters without proportionally more compute per token.</li>
  <li><strong>Long-context extension:</strong> position interpolation or YaRN adjust RoPE frequencies to generalize to longer sequences without full retraining.</li>
  <li><strong>In-context learning vs. fine-tuning:</strong> few-shot prompting can match fine-tuning for narrow tasks, but doesn't scale to tasks needing persistent behavior change.</li>
  <li><strong>Multimodal transformers:</strong> ViT treats image patches as tokens; CLIP aligns image/text embeddings contrastively; cross-attention fuses modalities.</li>
</ul>
<h3>Code</h3>
<p class="code-label">Top-k expert routing for an MoE layer:</p>
<pre><code>import torch
import torch.nn as nn

class MoELayer(nn.Module):
    def __init__(self, d_model, n_experts, k=2):
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                            nn.Linear(4 * d_model, d_model))
             for _ in range(n_experts)]
        )
        self.gate = nn.Linear(d_model, n_experts)
        self.k = k

    def forward(self, x):
        gate_logits = self.gate(x)
        topk_vals, topk_idx = gate_logits.topk(self.k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)
        out = torch.zeros_like(x)
        for i in range(self.k):
            expert_idx = topk_idx[..., i]
            weight = topk_weights[..., i].unsqueeze(-1)
            for e, expert in enumerate(self.experts):
                mask = (expert_idx == e).unsqueeze(-1)
                out = out + mask * weight * expert(x)
        return out</code></pre>
''',
        "quiz": [
            {"q": "What's the core trade-off MoE is exploiting?", "a": "Routing each token to only a small subset of 'expert' FFNs gives you many more total parameters without proportionally more compute per token, since each token only activates a fraction of them."},
            {"q": "In CLIP, how are image and text representations related?", "a": "They're aligned contrastively into a shared embedding space, so matching image/text pairs end up close together and non-matching pairs far apart."},
        ],
    },
    {
        "id": "whiteboard-prompts",
        "category": "extras",
        "title": "Common Whiteboard / Coding Prompts",
        "html": '''
<ul>
  <li>Implement scaled dot-product attention from scratch.</li>
  <li>Implement LoRA's forward pass for a single linear layer.</li>
  <li>Implement the PPO clipped surrogate objective.</li>
  <li>Derive the gradient of softmax + cross-entropy loss.</li>
  <li>Debug a training run with loss spikes/NaNs -- LR too high, a corrupted data batch, fp16 overflow, or missing gradient clipping.</li>
</ul>
<p class="callout">You've built and tested working solutions to the first three of these already, in the coding exercises above (Transformers and Policy-Based RL categories).</p>
<h3>Code</h3>
<p class="code-label">The softmax + cross-entropy gradient, and a NaN-debugging checklist:</p>
<pre><code># Gradient of softmax + cross-entropy w.r.t. logits, for true class y:
#   d(loss)/d(logits) = softmax(logits) - one_hot(y)
def cross_entropy_grad(logits, y):
    probs = torch.softmax(logits, dim=-1)
    grad = probs.clone()
    grad[range(len(y)), y] -= 1
    return grad / len(y)

# Checklist when loss spikes or goes NaN:
# 1. Log grad norm before clipping -- is it exploding?
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e9)
# 2. Check the batch for corrupted tokens or all-padding rows
assert not torch.isnan(batch["input_ids"]).any()
# 3. Check for fp16 overflow -- switch to bf16 or enable loss scaling
# 4. Confirm gradient clipping is actually wired into the optimizer step</code></pre>
''',
        "quiz": [
            {"q": "What's the closed-form gradient of softmax + cross-entropy loss with respect to the logits?", "a": "softmax(logits) - one_hot(y), averaged over the batch -- you built and tested exactly this in the cross_entropy_backward exercise."},
            {"q": "Name three common causes of a NaN/loss-spike during training.", "a": "Learning rate too high, a corrupted data batch (e.g. NaN tokens or all-padding rows), and fp16 numerical overflow (mitigated by switching to bf16 or using loss scaling)."},
        ],
    },
    {
        "id": "process-topics",
        "category": "extras",
        "title": "Research-Scientist-Specific Process Topics",
        "html": '''
<p>These get probed as much as raw technical knowledge, especially at the onsite stage:</p>
<ul>
  <li><strong>Ablation design:</strong> isolate one variable at a time, and make comparisons compute-matched so you're not confounding "better idea" with "more compute."</li>
  <li><strong>Reproducibility:</strong> report variance across seeds/runs, not a single best number.</li>
  <li><strong>Reading a paper cold:</strong> identify the core claim, the actual evidence for it, and the most likely confound.</li>
  <li><strong>Explaining your own work honestly:</strong> interviewers often probe what <em>didn't</em> work more than what did.</li>
</ul>
''',
        "quiz": [
            {"q": "Why must ablation comparisons be compute-matched?", "a": "Otherwise you can't tell whether an improvement came from the idea being tested or simply from throwing more compute at that variant, confounding the two."},
            {"q": "Why report variance across seeds instead of a single best run?", "a": "A single best-of-N run overstates how reliably an improvement holds; reporting variance across seeds shows whether the effect is real or within noise."},
        ],
    },
]

STUDY_SECTIONS_BY_ID = {s["id"]: s for s in STUDY_SECTIONS}
