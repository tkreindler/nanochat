using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace NanoChat.Core;

/// <summary>
/// Special tokens used in the NanoChat tokenizer
/// </summary>
public static class SpecialTokens
{
    public const string BOS = "<|bos|>";
    public const string UserStart = "<|user_start|>";
    public const string UserEnd = "<|user_end|>";
    public const string AssistantStart = "<|assistant_start|>";
    public const string AssistantEnd = "<|assistant_end|>";
    public const string PythonStart = "<|python_start|>";
    public const string PythonEnd = "<|python_end|>";
    public const string OutputStart = "<|output_start|>";
    public const string OutputEnd = "<|output_end|>";

    public static readonly string[] All = 
    {
        BOS,
        UserStart,
        UserEnd,
        AssistantStart,
        AssistantEnd,
        PythonStart,
        PythonEnd,
        OutputStart,
        OutputEnd
    };
}

/// <summary>
/// Interface for tokenizer implementations.
/// This provides a common interface that can be implemented by different backends
/// (TikToken, HuggingFace, or a direct Rust BPE wrapper).
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Get the vocabulary size
    /// </summary>
    int VocabSize { get; }

    /// <summary>
    /// Get the BOS (beginning of sequence) token ID
    /// </summary>
    int BosTokenId { get; }

    /// <summary>
    /// Encode a special token to its ID
    /// </summary>
    int EncodeSpecial(string token);

    /// <summary>
    /// Encode text to token IDs
    /// </summary>
    List<int> Encode(string text, int? prepend = null, int? append = null);

    /// <summary>
    /// Encode a batch of text to token IDs
    /// </summary>
    List<List<int>> EncodeBatch(List<string> texts, int? prepend = null, int? append = null);

    /// <summary>
    /// Decode token IDs back to text
    /// </summary>
    string Decode(List<int> ids);

    /// <summary>
    /// Get the string representation of a single token ID
    /// </summary>
    string IdToToken(int id);
}

/// <summary>
/// Stub tokenizer implementation for testing purposes.
/// TODO: Replace with actual implementation (TikToken, HuggingFace, or Rust wrapper)
/// </summary>
public class StubTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _specialTokens;
    private readonly int _vocabSize;

    public int VocabSize => _vocabSize;
    public int BosTokenId => _specialTokens[SpecialTokens.BOS];

    public StubTokenizer(int vocabSize = 50304)
    {
        _vocabSize = vocabSize;
        
        // Initialize special tokens at the end of vocab
        _specialTokens = new Dictionary<string, int>();
        int offset = vocabSize - SpecialTokens.All.Length;
        for (int i = 0; i < SpecialTokens.All.Length; i++)
        {
            _specialTokens[SpecialTokens.All[i]] = offset + i;
        }
    }

    public int EncodeSpecial(string token)
    {
        if (_specialTokens.TryGetValue(token, out int id))
            return id;
        throw new ArgumentException($"Unknown special token: {token}");
    }

    public List<int> Encode(string text, int? prepend = null, int? append = null)
    {
        // Stub implementation: just convert to UTF-8 bytes modulo vocab size
        var ids = new List<int>();
        
        if (prepend.HasValue)
            ids.Add(prepend.Value);

        // Very naive encoding - in real implementation this would use BPE
        var bytes = System.Text.Encoding.UTF8.GetBytes(text);
        foreach (var b in bytes)
        {
            ids.Add(b % 256); // Keep in first 256 tokens (byte range)
        }

        if (append.HasValue)
            ids.Add(append.Value);

        return ids;
    }

    public List<List<int>> EncodeBatch(List<string> texts, int? prepend = null, int? append = null)
    {
        var result = new List<List<int>>();
        foreach (var text in texts)
        {
            result.Add(Encode(text, prepend, append));
        }
        return result;
    }

    public string Decode(List<int> ids)
    {
        // Stub implementation: try to decode as UTF-8 bytes
        var bytes = new List<byte>();
        foreach (var id in ids)
        {
            if (id < 256)
                bytes.Add((byte)id);
        }
        return System.Text.Encoding.UTF8.GetString(bytes.ToArray());
    }

    public string IdToToken(int id)
    {
        // Check if it's a special token
        foreach (var kvp in _specialTokens)
        {
            if (kvp.Value == id)
                return kvp.Key;
        }

        // Otherwise return byte representation
        if (id < 256)
            return System.Text.Encoding.UTF8.GetString(new[] { (byte)id });
        
        return $"<{id}>";
    }

    public static StubTokenizer FromDirectory(string tokenizerDir)
    {
        // TODO: Load actual tokenizer from directory
        // For now, just return a stub with default vocab size
        return new StubTokenizer();
    }
}

/// <summary>
/// Factory for creating tokenizer instances
/// </summary>
public static class TokenizerFactory
{
    /// <summary>
    /// Load tokenizer from a directory
    /// </summary>
    public static ITokenizer FromDirectory(string tokenizerDir)
    {
        // TODO: Detect tokenizer type and load appropriately
        // For now, use stub tokenizer
        return StubTokenizer.FromDirectory(tokenizerDir);
    }

    /// <summary>
    /// Get the default tokenizer from the base directory
    /// </summary>
    public static ITokenizer GetTokenizer(string? baseDir = null)
    {
        baseDir ??= Environment.GetEnvironmentVariable("NANOCHAT_BASE_DIR") ?? "./out";
        var tokenizerDir = Path.Combine(baseDir, "tokenizer");
        return FromDirectory(tokenizerDir);
    }
}
