namespace NanoChat.Tokenizer;

/// <summary>
/// Special tokens used by the tokenizer
/// Port of SPECIAL_TOKENS from tokenizer.py
/// </summary>
public static class SpecialTokens
{
    /// <summary>
    /// Every document begins with the Beginning of Sequence (BOS) token that delimits documents
    /// </summary>
    public const string Bos = "<|bos|>";
    
    /// <summary>
    /// User message start token (used during finetuning)
    /// </summary>
    public const string UserStart = "<|user_start|>";
    
    /// <summary>
    /// User message end token (used during finetuning)
    /// </summary>
    public const string UserEnd = "<|user_end|>";
    
    /// <summary>
    /// Assistant message start token (used during finetuning)
    /// </summary>
    public const string AssistantStart = "<|assistant_start|>";
    
    /// <summary>
    /// Assistant message end token (used during finetuning)
    /// </summary>
    public const string AssistantEnd = "<|assistant_end|>";
    
    /// <summary>
    /// Python REPL tool invocation start token
    /// </summary>
    public const string PythonStart = "<|python_start|>";
    
    /// <summary>
    /// Python REPL tool invocation end token
    /// </summary>
    public const string PythonEnd = "<|python_end|>";
    
    /// <summary>
    /// Python REPL output start token
    /// </summary>
    public const string OutputStart = "<|output_start|>";
    
    /// <summary>
    /// Python REPL output end token
    /// </summary>
    public const string OutputEnd = "<|output_end|>";
    
    /// <summary>
    /// All special tokens in order
    /// </summary>
    public static readonly string[] All = new[]
    {
        Bos,
        UserStart,
        UserEnd,
        AssistantStart,
        AssistantEnd,
        PythonStart,
        PythonEnd,
        OutputStart,
        OutputEnd
    };
    
    /// <summary>
    /// Split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
    /// This is to avoid "wasting" too many tokens on numbers for smaller vocab sizes
    /// </summary>
    public const string SplitPattern = @"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
}
