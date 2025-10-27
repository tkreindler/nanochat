using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core;

/// <summary>
/// GPT model configuration.
/// Notable features:
/// - Rotary embeddings (no positional embeddings)
/// - QK norm
/// - Untied weights for token embedding and lm_head
/// - ReLU² activation in MLP
/// - Norm after token embedding
/// - No learnable params in RMSNorm
/// - No bias in linear layers
/// - Multi-Query Attention (MQA) support for efficient inference
/// </summary>
public class GPTConfig
{
    public int SequenceLen { get; set; } = 1024;
    public int VocabSize { get; set; } = 50304;
    public int NLayer { get; set; } = 12;
    public int NHead { get; set; } = 6;  // number of query heads
    public int NKVHead { get; set; } = 6; // number of key/value heads (MQA)
    public int NEmbd { get; set; } = 768;
}

/// <summary>
/// Purely functional RMSNorm with no learnable parameters.
/// </summary>
public static class Norm
{
    public static Tensor Forward(Tensor x)
    {
        // RMSNorm: normalize by root mean square
        // Manual implementation since TorchSharp doesn't have rms_norm yet
        var variance = x.pow(2).mean(dimensions: new long[] { -1 }, keepdim: true);
        return x * torch.rsqrt(variance + 1e-6f);
    }
}

/// <summary>
/// Apply rotary positional embeddings to query or key tensors.
/// </summary>
public static class RotaryEmbedding
{
    public static Tensor Apply(Tensor x, Tensor cos, Tensor sin)
    {
        if (x.dim() != 4)
            throw new ArgumentException("Expected 4D tensor for multihead attention");

        var d = x.shape[3] / 2;
        var x1 = x[.., .., .., ..(int)d];
        var x2 = x[.., .., .., (int)d..];
        
        // Rotate pairs of dimensions
        var y1 = x1 * cos + x2 * sin;
        var y2 = x1 * (-sin) + x2 * cos;
        
        var output = torch.cat(new[] { y1, y2 }, dim: 3);
        return output.to(x.dtype); // ensure input/output dtypes match
    }
}

/// <summary>
/// Causal self-attention with rotary embeddings, QK norm, and optional MQA/GQA.
/// </summary>
public class CausalSelfAttention : Module<Tensor, (Tensor, Tensor), object?, Tensor>
{
    private readonly int _layerIdx;
    private readonly int _nHead;
    private readonly int _nKVHead;
    private readonly int _nEmbd;
    private readonly int _headDim;
    
    private readonly Linear _cQ;
    private readonly Linear _cK;
    private readonly Linear _cV;
    internal readonly Linear _cProj;

    public CausalSelfAttention(GPTConfig config, int layerIdx, string name = "CausalSelfAttention")
        : base(name)
    {
        _layerIdx = layerIdx;
        _nHead = config.NHead;
        _nKVHead = config.NKVHead;
        _nEmbd = config.NEmbd;
        _headDim = _nEmbd / _nHead;

        if (_nEmbd % _nHead != 0)
            throw new ArgumentException($"n_embd ({_nEmbd}) must be divisible by n_head ({_nHead})");
        
        if (_nKVHead > _nHead || _nHead % _nKVHead != 0)
            throw new ArgumentException($"Invalid MQA/GQA configuration: n_head={_nHead}, n_kv_head={_nKVHead}");

        _cQ = Linear(_nEmbd, _nHead * _headDim, hasBias: false);
        _cK = Linear(_nEmbd, _nKVHead * _headDim, hasBias: false);
        _cV = Linear(_nEmbd, _nKVHead * _headDim, hasBias: false);
        _cProj = Linear(_nEmbd, _nEmbd, hasBias: false);

        RegisterComponents();
    }

    public override Tensor forward(Tensor x, (Tensor, Tensor) cosSin, object? kvCache)
    {
        var (B, T, C) = (x.shape[0], x.shape[1], x.shape[2]);

        // Project the input to get queries, keys, and values
        var q = _cQ.forward(x).view(B, T, _nHead, _headDim);
        var k = _cK.forward(x).view(B, T, _nKVHead, _headDim);
        var v = _cV.forward(x).view(B, T, _nKVHead, _headDim);

        // Apply Rotary Embeddings to queries and keys
        var (cos, sin) = cosSin;
        q = RotaryEmbedding.Apply(q, cos, sin);
        k = RotaryEmbedding.Apply(k, cos, sin);
        
        // QK norm
        q = Norm.Forward(q);
        k = Norm.Forward(k);
        
        // Transpose to make head be batch dim: (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2);
        k = k.transpose(1, 2);
        v = v.transpose(1, 2);

        // Handle KV cache for inference
        if (kvCache is KVCache cache)
        {
            (k, v) = cache.InsertKV(_layerIdx, k, v);
        }
        
        var Tq = q.shape[2]; // number of queries
        var Tk = k.shape[2]; // number of keys/values

        // Attention: queries attend to keys/values autoregressively
        var enableGQA = _nHead != _nKVHead;
        
        Tensor y;
        if (kvCache == null || Tq == Tk)
        {
            // Training mode: use causal attention
            y = functional.scaled_dot_product_attention(q, k, v, null, 0.0, true);
        }
        else if (Tq == 1)
        {
            // Inference with single query: attend to all cached keys/values
            y = functional.scaled_dot_product_attention(q, k, v, null, 0.0, false);
        }
        else
        {
            // Inference with multiple queries: use custom attention mask
            var attnMask = torch.zeros(new long[] { Tq, Tk }, dtype: ScalarType.Bool, device: q.device);
            var prefixLen = Tk - Tq;
            
            if (prefixLen > 0)
            {
                attnMask[.., ..(int)prefixLen] = true;
            }
            
            // Causal attention within the chunk
            var causalMask = torch.tril(torch.ones(new long[] { Tq, Tq }, dtype: ScalarType.Bool, device: q.device));
            attnMask[.., (int)prefixLen..] = causalMask;
            
            y = functional.scaled_dot_product_attention(q, k, v, attnMask, 0.0, false);
        }

        // Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1);
        y = _cProj.forward(y);
        
        return y;
    }
}

/// <summary>
/// MLP with ReLU² activation.
/// </summary>
public class MLP : Module<Tensor, Tensor>
{
    private readonly Linear _cFc;
    internal readonly Linear _cProj;

    public MLP(GPTConfig config, string name = "MLP") : base(name)
    {
        _cFc = Linear(config.NEmbd, 4 * config.NEmbd, hasBias: false);
        _cProj = Linear(4 * config.NEmbd, config.NEmbd, hasBias: false);
        RegisterComponents();
    }

    public override Tensor forward(Tensor x)
    {
        x = _cFc.forward(x);
        x = functional.relu(x).square(); // ReLU²
        x = _cProj.forward(x);
        return x;
    }
}

/// <summary>
/// Transformer block with attention and MLP.
/// </summary>
public class Block : Module<Tensor, (Tensor, Tensor), object?, Tensor>
{
    internal readonly CausalSelfAttention _attn;
    internal readonly MLP _mlp;

    public Block(GPTConfig config, int layerIdx, string name = "Block") : base(name)
    {
        _attn = new CausalSelfAttention(config, layerIdx, $"attn_{layerIdx}");
        _mlp = new MLP(config, $"mlp_{layerIdx}");
        RegisterComponents();
    }

    public override Tensor forward(Tensor x, (Tensor, Tensor) cosSin, object? kvCache)
    {
        x = x + _attn.forward(Norm.Forward(x), cosSin, kvCache);
        x = x + _mlp.forward(Norm.Forward(x));
        return x;
    }
}

/// <summary>
/// GPT model with rotary embeddings, QK norm, and ReLU² MLP.
/// </summary>
public class GPT : Module<Tensor, Tensor?, object?, string, Tensor>
{
    private readonly GPTConfig _config;
    private readonly Embedding _wte;
    private readonly ModuleList<Block> _blocks;
    private readonly Linear _lmHead;
    private readonly int _rotarySeqLen;
    
    // Rotary embeddings (non-persistent buffers)
    private Tensor? _cos;
    private Tensor? _sin;

    public GPTConfig Config => _config;

    public GPT(GPTConfig config, string name = "GPT") : base(name)
    {
        _config = config;
        
        // Token embeddings and transformer blocks
        _wte = Embedding(config.VocabSize, config.NEmbd);
        _blocks = new ModuleList<Block>();
        for (int i = 0; i < config.NLayer; i++)
        {
            _blocks.Add(new Block(config, i, $"block_{i}"));
        }
        
        // Language model head (untied from token embeddings)
        _lmHead = Linear(config.NEmbd, config.VocabSize, hasBias: false);
        
        // Rotary embeddings: over-compute by 10x to allow for longer sequences
        _rotarySeqLen = config.SequenceLen * 10;
        var headDim = config.NEmbd / config.NHead;
        (_cos, _sin) = PrecomputeRotaryEmbeddings(_rotarySeqLen, headDim);

        RegisterComponents();
    }

    public void InitWeights()
    {
        // Initialize all weights
        apply(InitWeightsModule);
        
        // Zero out classifier weights
        init.zeros_(_lmHead.weight);
        
        // Zero out c_proj weights in all blocks
        foreach (var block in _blocks)
        {
            init.zeros_(block._mlp._cProj.weight);
            init.zeros_(block._attn._cProj.weight);
        }
        
        // Re-initialize rotary embeddings
        var headDim = _config.NEmbd / _config.NHead;
        (_cos, _sin) = PrecomputeRotaryEmbeddings(_rotarySeqLen, headDim);
        
        // Cast embeddings to bfloat16 if on CUDA (saves memory)
        if (_wte.weight.device_type == DeviceType.CUDA)
        {
            _wte.to(ScalarType.BFloat16);
        }
    }

    private void InitWeightsModule(Module module)
    {
        if (module is Linear linear)
        {
            // Weight initialization based on https://arxiv.org/pdf/2310.17813
            var fanOut = linear.weight.shape[0];
            var fanIn = linear.weight.shape[1];
            var std = 1.0 / Math.Sqrt(fanIn) * Math.Min(1.0, Math.Sqrt((double)fanOut / fanIn));
            init.normal_(linear.weight, mean: 0.0, std: std);
            
            if (linear.bias is not null)
            {
                init.zeros_(linear.bias);
            }
        }
        else if (module is Embedding embedding)
        {
            init.normal_(embedding.weight, mean: 0.0, std: 1.0);
        }
    }

    private (Tensor cos, Tensor sin) PrecomputeRotaryEmbeddings(int seqLen, int headDim, int baseTheta = 10000, Device? device = null)
    {
        // Auto-detect device from model embeddings
        device ??= _wte.weight.device;
        
        // Stride the channels
        var channelRange = torch.arange(0, headDim, 2, dtype: ScalarType.Float32, device: device);
        var invFreq = 1.0 / torch.pow(baseTheta, channelRange / headDim);
        
        // Stride the time steps
        var t = torch.arange(seqLen, dtype: ScalarType.Float32, device: device);
        
        // Calculate rotation frequencies at each (time, channel) pair
        var freqs = torch.outer(t, invFreq);
        var cos = freqs.cos().to(ScalarType.BFloat16);
        var sin = freqs.sin().to(ScalarType.BFloat16);
        
        // Add batch and head dims for broadcasting: (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
        cos = cos.unsqueeze(0).unsqueeze(2);
        sin = sin.unsqueeze(0).unsqueeze(2);
        
        return (cos, sin);
    }

    public Device GetDevice()
    {
        return _wte.weight.device;
    }

    public long EstimateFlops()
    {
        // Estimated FLOPs per token. Ref: https://arxiv.org/abs/2204.02311
        var nParams = parameters().Sum(p => p.numel());
        var nParamsEmbedding = _wte.weight.numel();
        var (l, h, q, t) = (_config.NLayer, _config.NHead, _config.NEmbd / _config.NHead, _config.SequenceLen);
        var numFlopsPerToken = 6 * (nParams - nParamsEmbedding) + 12 * l * h * q * t;
        return numFlopsPerToken;
    }

    public override Tensor forward(Tensor idx, Tensor? targets = null, object? kvCache = null, string lossReduction = "mean")
    {
        var (B, T) = (idx.shape[0], idx.shape[1]);

        // Ensure rotary embeddings are on the same device as input
        if (_cos is null || _cos.device != idx.device)
        {
            var headDim = _config.NEmbd / _config.NHead;
            (_cos, _sin) = PrecomputeRotaryEmbeddings(_rotarySeqLen, headDim, device: idx.device);
        }

        // Validate rotary embeddings
        if (T > _cos!.shape[1])
            throw new InvalidOperationException($"Sequence length grew beyond rotary embeddings cache: {T} > {_cos.shape[1]}");
        
        if (_cos.dtype != ScalarType.BFloat16)
            throw new InvalidOperationException("Rotary embeddings must be in bfloat16");

        // Get rotary embeddings for current sequence length
        // TODO: Handle KV cache position offset
        var T0 = 0; // kvCache?.GetPos() ?? 0;
        var cos = _cos[.., T0..(int)(T0 + T), .., ..];
        var sin = _sin![.., T0..(int)(T0 + T), .., ..];
        var cosSin = (cos, sin);

        // Forward the trunk of the Transformer
        var x = _wte.forward(idx);
        x = Norm.Forward(x);
        
        foreach (var block in _blocks)
        {
            x = block.forward(x, cosSin, kvCache);
        }
        
        x = Norm.Forward(x);

        // Forward the lm_head (compute logits)
        const float softcap = 15.0f;
        
        if (targets is not null)
        {
            // Training mode: compute and return the loss
            var logits = _lmHead.forward(x);
            logits = softcap * torch.tanh(logits / softcap); // logits softcap
            logits = logits.to(ScalarType.Float32); // use fp32 for logits
            
            var loss = functional.cross_entropy(
                logits.view(-1, logits.shape[^1]),
                targets.view(-1),
                reduction: lossReduction == "mean" ? Reduction.Mean : Reduction.Sum
            );
            
            return loss;
        }
        else
        {
            // Inference mode: compute and return the logits
            var logits = _lmHead.forward(x);
            logits = softcap * torch.tanh(logits / softcap); // logits softcap
            return logits;
        }
    }

    /// <summary>
    /// Naive autoregressive streaming inference.
    /// Assumes batch size of 1 and returns generated token IDs.
    /// </summary>
    public IEnumerable<long> Generate(long[] tokens, int maxTokens, float temperature = 1.0f, int? topK = null, int seed = 42)
    {
        using var _ = torch.no_grad();
        
        var device = GetDevice();
        
        // Set the random seed for sampling
        if (temperature > 0)
        {
            if (cuda.is_available() && device.type == DeviceType.CUDA)
                torch.cuda.manual_seed((long)seed);
            else
                torch.manual_seed((long)seed);
        }
        
        var ids = torch.tensor(tokens, dtype: ScalarType.Int64, device: device).unsqueeze(0); // add batch dim
        
        for (int i = 0; i < maxTokens; i++)
        {
            var logits = forward(ids); // (B, T, vocab_size)
            logits = logits[.., -1, ..]; // (B, vocab_size) - get last token
            
            // Apply top-k filtering if specified
            if (topK.HasValue)
            {
                var (v, _) = torch.topk(logits, Math.Min(topK.Value, (int)logits.shape[^1]));
                logits[logits < v[.., -1]] = float.NegativeInfinity;
            }
            
            Tensor nextIds;
            if (temperature > 0)
            {
                logits = logits / temperature;
                var probs = functional.softmax(logits, dim: -1);
                nextIds = torch.multinomial(probs, num_samples: 1);
            }
            else
            {
                nextIds = torch.argmax(logits, dim: -1, keepdim: true);
            }
            
            ids = torch.cat(new[] { ids, nextIds }, dim: 1);
            var token = nextIds.item<long>();
            yield return token;
        }
    }
}
