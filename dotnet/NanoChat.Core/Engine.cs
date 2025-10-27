using System;
using System.Collections.Generic;
using System.Linq;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core;

/// <summary>
/// KV Cache for efficient inference with Transformer models.
/// Works hand-in-hand with the GPT model to maintain the key-value cache.
/// Note that the .pos advances automatically after the last layer of the Transformer inserts.
/// </summary>
public class KVCache
{
    private readonly long[] _kvShape;
    private Tensor? _kvCache;
    private long _pos;

    public KVCache(long batchSize, long numHeads, long seqLen, long headDim, long numLayers)
    {
        // Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer
        _kvShape = new[] { numLayers, 2, batchSize, numHeads, seqLen, headDim };
        _kvCache = null;
        _pos = 0;
    }

    public long Pos => _pos;

    public void Reset()
    {
        _pos = 0;
    }

    public long GetPos() => _pos;

    /// <summary>
    /// Prefill given another KV cache. Optionally expand along batch dim.
    /// This is used when we do batch 1 prefill and then want to generate
    /// multiple samples in parallel from there.
    /// </summary>
    public void Prefill(KVCache other)
    {
        // 1) Validate the shapes
        if (_kvCache is not null)
            throw new InvalidOperationException("Cannot prefill a non-empty KV cache");
        if (other._kvCache is null)
            throw new InvalidOperationException("Cannot prefill with a None KV cache");

        for (int ix = 0; ix < _kvShape.Length; ix++)
        {
            var dim1 = _kvShape[ix];
            var dim2 = other._kvShape[ix];

            if (ix == 0 || ix == 1 || ix == 3 || ix == 5)
            {
                // num_layers, kv, num_heads, head_dim must match
                if (dim1 != dim2)
                    throw new InvalidOperationException($"Shape mismatch at index {ix}: {dim1} != {dim2}");
            }
            else if (ix == 2)
            {
                // batch_size can be expanded
                if (dim1 != dim2 && dim2 != 1)
                    throw new InvalidOperationException($"Batch dim mismatch: {dim1} != {dim2}");
            }
            else if (ix == 4)
            {
                // seq_len: self must be longer than other
                if (dim1 < dim2)
                    throw new InvalidOperationException($"Seq len mismatch: {dim1} < {dim2}");
            }
        }

        // 2) Initialize the cache
        var dtype = other._kvCache.dtype;
        var device = other._kvCache.device;
        _kvCache = torch.empty(_kvShape, dtype: dtype, device: device);

        // 3) Copy the data over (up to other.pos)
        _kvCache[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, 
                 TensorIndex.Slice(0, other._pos), TensorIndex.Colon] = 
            other._kvCache[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon,
                          TensorIndex.Slice(0, other._pos), TensorIndex.Colon];

        // 4) Update the pos
        _pos = other._pos;
    }

    /// <summary>
    /// Insert new keys/values to the cache and return the full cache so far.
    /// </summary>
    public (Tensor key, Tensor value) InsertKV(long layerIdx, Tensor k, Tensor v)
    {
        // Lazy initialize the cache here because we need to know the dtype/device
        if (_kvCache is null)
        {
            _kvCache = torch.empty(_kvShape, dtype: k.dtype, device: k.device);
        }

        // Insert new keys/values to the cache
        var shape = k.shape;
        long B = shape[0], H = shape[1], T_add = shape[2], D = shape[3];
        long t0 = _pos;
        long t1 = _pos + T_add;

        // Dynamically grow the cache if needed
        if (t1 > _kvCache.shape[4])
        {
            long tNeeded = t1 + 1024; // As much as we need plus buffer of 1024
            tNeeded = (tNeeded + 1023) & ~1023; // Round up to nearest multiple of 1024
            
            var currentShape = _kvCache.shape.ToArray();
            currentShape[4] = tNeeded;
            
            // Create new larger tensor and copy old data
            var newCache = torch.empty(currentShape, dtype: _kvCache.dtype, device: _kvCache.device);
            newCache[TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Colon, 
                     TensorIndex.Slice(0, _kvCache.shape[4])] = _kvCache;
            _kvCache = newCache;
        }

        // Insert k, v into the cache
        _kvCache[layerIdx, 0, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(t0, t1)] = k;
        _kvCache[layerIdx, 1, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(t0, t1)] = v;

        // Return the full cached keys/values up to current position (as a view)
        var keyView = _kvCache[layerIdx, 0, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(0, t1)];
        var valueView = _kvCache[layerIdx, 1, TensorIndex.Colon, TensorIndex.Colon, TensorIndex.Slice(0, t1)];

        // Increment pos after the last layer of the Transformer processes
        if (layerIdx == _kvCache.shape[0] - 1)
        {
            _pos = t1;
        }

        return (keyView, valueView);
    }
}

/// <summary>
/// Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1).
/// Note: This function does NOT set the random seed. The caller must set the seed before calling this.
/// </summary>
public static class Sampling
{
    public static Tensor SampleNextToken(Tensor logits, double temperature = 1.0, int? topK = null)
    {
        if (temperature < 0.0)
            throw new ArgumentException("temperature must be non-negative");

        if (temperature == 0.0)
        {
            return torch.argmax(logits, dim: -1, keepdim: true);
        }

        if (topK.HasValue)
        {
            int k = (int)Math.Min(topK.Value, logits.shape[^1]);
            var (vals, idx) = torch.topk(logits, k, dim: -1);
            vals = vals / temperature;
            var probs = functional.softmax(vals, dim: -1);
            var choice = torch.multinomial(probs, num_samples: 1);
            return idx.gather(1, choice);
        }
        else
        {
            logits = logits / temperature;
            var probs = functional.softmax(logits, dim: -1);
            return torch.multinomial(probs, num_samples: 1);
        }
    }
}

/// <summary>
/// Per-row state tracking during generation
/// </summary>
public class RowState
{
    public List<long> CurrentTokens { get; set; }
    public Queue<long> ForcedTokens { get; set; }
    public bool InPythonBlock { get; set; }
    public List<long> PythonExprTokens { get; set; }
    public bool Completed { get; set; }

    public RowState(List<long>? currentTokens = null)
    {
        CurrentTokens = currentTokens ?? new List<long>();
        ForcedTokens = new Queue<long>();
        InPythonBlock = false;
        PythonExprTokens = new List<long>();
        Completed = false;
    }
}

/// <summary>
/// Engine for efficient inference of language models.
/// Everything works around token sequences:
/// - The user can send token sequences to the engine
/// - The engine returns the next token
/// 
/// Notes:
/// - The engine knows nothing about tokenization, it's purely token id sequences.
/// - The whole thing is made as efficient as possible.
/// </summary>
public class Engine
{
    private readonly GPT _model;
    private readonly ITokenizer _tokenizer;

    public Engine(GPT model, ITokenizer tokenizer)
    {
        _model = model;
        _tokenizer = tokenizer;
    }

    /// <summary>
    /// Generate tokens from a prompt. Does single prefill and then clones the KV cache.
    /// Yields (tokenColumn, tokenMasks) where tokenColumn is a list of tokens (one per sample)
    /// and tokenMasks indicates whether each token was sampled (1) or forced (0).
    /// </summary>
    public IEnumerable<(List<long> tokenColumn, List<int> tokenMasks)> Generate(
        List<long> tokens,
        int numSamples = 1,
        int? maxTokens = null,
        double temperature = 1.0,
        int? topK = null,
        int seed = 42)
    {
        var device = _model.GetDevice();
        
        // Set the random seed for sampling
        if (cuda.is_available() && device.type == DeviceType.CUDA)
            torch.cuda.manual_seed((long)seed);
        else
            torch.manual_seed((long)seed);

        // Get the special tokens we need to coordinate the tool use state machine
        long GetSpecial(string s) => _tokenizer.EncodeSpecial(s);
        var pythonStart = GetSpecial("<|python_start|>");
        var pythonEnd = GetSpecial("<|python_end|>");
        var outputStart = GetSpecial("<|output_start|>");
        var outputEnd = GetSpecial("<|output_end|>");
        var assistantEnd = GetSpecial("<|assistant_end|>"); // if sampled, ends row
        var bos = _tokenizer.BosTokenId; // if sampled, ends row

        // 1) Run a batch 1 prefill of the prompt tokens
        var config = _model.Config;
        var kvModelKwargs = new
        {
            numHeads = config.NKVHead,
            headDim = config.NEmbd / config.NHead,
            numLayers = config.NLayer
        };

        var kvCachePrefill = new KVCache(
            batchSize: 1,
            numHeads: kvModelKwargs.numHeads,
            seqLen: tokens.Count,
            headDim: kvModelKwargs.headDim,
            numLayers: kvModelKwargs.numLayers
        );


        using var ids = torch.tensor(tokens.ToArray(), dtype: ScalarType.Int64, device: device).unsqueeze(0);
        Tensor logits;
        using (no_grad())
        {
            logits = _model.forward(ids, kvCache: kvCachePrefill);
        }
        logits = logits[TensorIndex.Colon, -1, TensorIndex.Colon];
        var nextIds = Sampling.SampleNextToken(logits, temperature, topK); // (B, 1)
        var sampledTokens = nextIds[TensorIndex.Colon, 0].data<long>().ToArray().ToList();

        // 2) Replicate the KV cache for each sample/row
        long kvLengthHint = maxTokens.HasValue ? tokens.Count + maxTokens.Value : _model.Config.SequenceLen;
        var kvCacheDecode = new KVCache(
            batchSize: numSamples,
            numHeads: kvModelKwargs.numHeads,
            seqLen: kvLengthHint,
            headDim: kvModelKwargs.headDim,
            numLayers: kvModelKwargs.numLayers
        );
        kvCacheDecode.Prefill(kvCachePrefill);

        // 3) Initialize states for each sample
        var rowStates = Enumerable.Range(0, numSamples)
            .Select(_ => new RowState(new List<long>(tokens)))
            .ToList();

        // 4) Main generation loop
        int numGenerated = 0;
        bool firstIteration = true;

        while (true)
        {
            // Stop condition: we've reached max tokens
            if (maxTokens.HasValue && numGenerated >= maxTokens.Value)
                break;

            // Stop condition: all rows are completed
            if (rowStates.All(state => state.Completed))
                break;

            // Get sampled tokens - either from prefill or from forward pass
            List<long> currentSampledTokens;
            if (firstIteration)
            {
                // Use the tokens we already sampled from prefill
                currentSampledTokens = Enumerable.Repeat(sampledTokens[0], numSamples).ToList();
                // TODO: we should sample a token for each row instead of broadcasting
                firstIteration = false;
            }
            else
            {
                // Forward the model and get the next token for each row
                var lastTokens = rowStates.Select(s => s.CurrentTokens.Last()).ToArray();
                using var currentIds = torch.tensor(lastTokens, dtype: ScalarType.Int64, device: device).unsqueeze(1);
                
                using (no_grad())
                {
                    logits = _model.forward(currentIds, kvCache: kvCacheDecode); // (B, T, vocab_size)
                }
                logits = logits[TensorIndex.Colon, -1, TensorIndex.Colon]; // (B, vocab_size) at last time step
                nextIds = Sampling.SampleNextToken(logits, temperature, topK); // (B, 1)
                currentSampledTokens = nextIds[TensorIndex.Colon, 0].data<long>().ToArray().ToList();
            }

            // Process each row: choose the next token, update state, optional tool use
            var tokenColumn = new List<long>(); // contains the next token id along each row
            var tokenMasks = new List<int>(); // contains the mask (was it sampled (1) or forced (0)?) along each row

            for (int i = 0; i < rowStates.Count; i++)
            {
                var state = rowStates[i];

                // Select the next token in this row
                bool isForced = state.ForcedTokens.Count > 0; // are there tokens waiting to be forced?
                tokenMasks.Add(isForced ? 0 : 1); // mask is 0 if forced, 1 if sampled
                long nextToken = isForced ? state.ForcedTokens.Dequeue() : currentSampledTokens[i];
                tokenColumn.Add(nextToken);

                // Update the state of this row to include the next token
                state.CurrentTokens.Add(nextToken);

                // On <|assistant_end|> or <|bos|>, mark the row as completed
                if (nextToken == assistantEnd || nextToken == bos)
                {
                    state.Completed = true;
                }

                // Handle tool logic
                if (nextToken == pythonStart)
                {
                    state.InPythonBlock = true;
                    state.PythonExprTokens.Clear();
                }
                else if (nextToken == pythonEnd && state.InPythonBlock)
                {
                    state.InPythonBlock = false;
                    if (state.PythonExprTokens.Count > 0)
                    {
                        var expr = _tokenizer.Decode(state.PythonExprTokens.Select(t => (int)t).ToList());
                        var result = UseCalculator(expr);
                        if (result != null)
                        {
                            var resultTokens = _tokenizer.Encode(result);
                            state.ForcedTokens.Enqueue(outputStart);
                            foreach (var token in resultTokens)
                                state.ForcedTokens.Enqueue(token);
                            state.ForcedTokens.Enqueue(outputEnd);
                        }
                    }
                    state.PythonExprTokens.Clear();
                }
                else if (state.InPythonBlock)
                {
                    state.PythonExprTokens.Add(nextToken);
                }
            }

            // Yield the token column
            yield return (tokenColumn, tokenMasks);
            numGenerated++;
        }
    }

    /// <summary>
    /// Non-streaming batch generation that just returns the final token sequences.
    /// Returns a list of token sequences (list of lists of longs).
    /// Terminal tokens (assistant_end, bos) are not included in the results.
    /// </summary>
    public (List<List<long>> results, List<List<int>> masks) GenerateBatch(
        List<long> tokens,
        int numSamples = 1,
        int? maxTokens = null,
        double temperature = 1.0,
        int? topK = null,
        int seed = 42)
    {
        var assistantEnd = _tokenizer.EncodeSpecial("<|assistant_end|>");
        var bos = _tokenizer.BosTokenId;

        var results = Enumerable.Range(0, numSamples)
            .Select(_ => new List<long>(tokens))
            .ToList();

        var masks = Enumerable.Range(0, numSamples)
            .Select(_ => Enumerable.Repeat(0, tokens.Count).ToList())
            .ToList();

        var completed = new bool[numSamples];

        foreach (var (tokenColumn, tokenMasks) in Generate(tokens, numSamples, maxTokens, temperature, topK, seed))
        {
            for (int i = 0; i < tokenColumn.Count; i++)
            {
                if (!completed[i])
                {
                    var token = tokenColumn[i];
                    var mask = tokenMasks[i];

                    if (token == assistantEnd || token == bos)
                    {
                        completed[i] = true;
                    }
                    else
                    {
                        results[i].Add(token);
                        masks[i].Add(mask);
                    }
                }
            }

            // Stop if all rows are completed
            if (completed.All(c => c))
                break;
        }

        return (results, masks);
    }

    /// <summary>
    /// Stub implementation of calculator tool. In production, this would safely evaluate
    /// Python expressions for the model to use as a tool.
    /// </summary>
    private string? UseCalculator(string expr)
    {
        // TODO: Implement safe expression evaluation
        // For now, return null (no calculator support)
        return null;
    }
}
