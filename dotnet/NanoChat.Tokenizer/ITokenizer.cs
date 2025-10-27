namespace NanoChat.Tokenizer;

/// <summary>
/// Interface for tokenizer implementations
/// </summary>
public interface ITokenizer
{
    /// <summary>
    /// Gets the vocabulary size
    /// </summary>
    int VocabSize { get; }
    
    /// <summary>
    /// Gets the set of special tokens
    /// </summary>
    IReadOnlySet<string> SpecialTokens { get; }
    
    /// <summary>
    /// Gets the BOS (beginning of sequence) token ID
    /// </summary>
    int BosTokenId { get; }
    
    /// <summary>
    /// Encodes a single string into token IDs
    /// </summary>
    /// <param name="text">Text to encode</param>
    /// <param name="prepend">Optional token ID or special token to prepend</param>
    /// <param name="append">Optional token ID or special token to append</param>
    /// <returns>List of token IDs</returns>
    List<int> Encode(string text, object? prepend = null, object? append = null);
    
    /// <summary>
    /// Encodes multiple strings into token IDs (batch)
    /// </summary>
    /// <param name="texts">List of texts to encode</param>
    /// <param name="prepend">Optional token ID or special token to prepend</param>
    /// <param name="append">Optional token ID or special token to append</param>
    /// <param name="numThreads">Number of threads for parallel encoding</param>
    /// <returns>List of token ID lists</returns>
    List<List<int>> Encode(IEnumerable<string> texts, object? prepend = null, object? append = null, int numThreads = 8);
    
    /// <summary>
    /// Encodes a special token by exact match
    /// </summary>
    /// <param name="token">Special token string</param>
    /// <returns>Token ID</returns>
    int EncodeSpecial(string token);
    
    /// <summary>
    /// Decodes token IDs back into text
    /// </summary>
    /// <param name="ids">Token IDs to decode</param>
    /// <returns>Decoded text</returns>
    string Decode(IEnumerable<int> ids);
    
    /// <summary>
    /// Gets the string representation of a single token ID
    /// </summary>
    /// <param name="id">Token ID</param>
    /// <returns>Token string</returns>
    string IdToToken(int id);
    
    /// <summary>
    /// Saves the tokenizer to disk
    /// </summary>
    /// <param name="directory">Output directory</param>
    void Save(string directory);
}
