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

    /// <summary>
    /// Render a conversation for training (SFT).
    /// Returns token IDs and a mask where 1 indicates tokens the assistant should learn from.
    /// </summary>
    /// <param name="conversation">The conversation to render</param>
    /// <param name="maxTokens">Maximum number of tokens to return</param>
    /// <returns>A tuple of (token IDs, mask values)</returns>
    (List<int> ids, List<int> mask) RenderConversation(Conversation conversation, int maxTokens = 2048);

    /// <summary>
    /// Render a conversation for completion (RL/inference).
    /// Removes the last assistant message and adds assistant_start token.
    /// </summary>
    /// <param name="conversation">The conversation to render</param>
    /// <returns>Token IDs ready for completion</returns>
    List<int> RenderForCompletion(Conversation conversation);

    /// <summary>
    /// Visualize tokenization with color coding (for debugging).
    /// Green = assistant tokens (mask=1), Red = user/system tokens (mask=0).
    /// </summary>
    /// <param name="ids">Token IDs</param>
    /// <param name="mask">Mask values</param>
    /// <param name="withTokenId">Include token IDs in output</param>
    /// <returns>Formatted string with color codes</returns>
    string VisualizeTokenization(List<int> ids, List<int> mask, bool withTokenId = false);
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

    public (List<int> ids, List<int> mask) RenderConversation(Conversation conversation, int maxTokens = 2048)
    {
        conversation.Validate();

        var ids = new List<int>();
        var mask = new List<int>();

        void AddTokens(List<int> tokenIds, int maskValue)
        {
            ids.AddRange(tokenIds);
            mask.AddRange(Enumerable.Repeat(maskValue, tokenIds.Count));
        }

        void AddToken(int tokenId, int maskValue)
        {
            ids.Add(tokenId);
            mask.Add(maskValue);
        }

        // Handle system message by merging with first user message
        var messages = conversation.Messages;
        if (conversation.HasSystemMessage)
        {
            var systemMessage = messages[0];
            var firstUserMessage = messages[1];
            
            if (firstUserMessage.Role != "user")
                throw new InvalidOperationException("System message must be followed by a user message");

            // Create merged content
            string mergedContent = systemMessage.GetStringContent() + "\n\n" + firstUserMessage.GetStringContent();
            
            // Create a new message list starting from index 1 with merged content
            var mergedFirstUser = new Message 
            { 
                Role = "user", 
                Content = mergedContent 
            };
            
            messages = new List<Message> { mergedFirstUser };
            messages.AddRange(conversation.Messages.Skip(2));
        }

        // Get special token IDs
        int bos = BosTokenId;
        int userStart = EncodeSpecial(SpecialTokens.UserStart);
        int userEnd = EncodeSpecial(SpecialTokens.UserEnd);
        int assistantStart = EncodeSpecial(SpecialTokens.AssistantStart);
        int assistantEnd = EncodeSpecial(SpecialTokens.AssistantEnd);
        int pythonStart = EncodeSpecial(SpecialTokens.PythonStart);
        int pythonEnd = EncodeSpecial(SpecialTokens.PythonEnd);
        int outputStart = EncodeSpecial(SpecialTokens.OutputStart);
        int outputEnd = EncodeSpecial(SpecialTokens.OutputEnd);

        // Add BOS token
        AddToken(bos, 0);

        // Process each message
        for (int i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            string expectedRole = i % 2 == 0 ? "user" : "assistant";
            
            if (message.Role != expectedRole)
                throw new InvalidOperationException(
                    $"Message {i} has role '{message.Role}' but should be '{expectedRole}'");

            if (message.Role == "user")
            {
                if (!message.IsSimpleString)
                    throw new InvalidOperationException("User messages must be simple strings");

                var content = message.GetStringContent();
                var valueIds = Encode(content);
                
                AddToken(userStart, 0);
                AddTokens(valueIds, 0);
                AddToken(userEnd, 0);
            }
            else if (message.Role == "assistant")
            {
                AddToken(assistantStart, 0);
                
                if (message.IsSimpleString)
                {
                    // Simple string content
                    var content = message.GetStringContent();
                    var valueIds = Encode(content);
                    AddTokens(valueIds, 1);
                }
                else
                {
                    // Structured content with parts
                    var parts = message.GetStructuredContent();
                    foreach (var part in parts)
                    {
                        var valueIds = Encode(part.Text);
                        
                        switch (part.Type)
                        {
                            case "text":
                                AddTokens(valueIds, 1);
                                break;
                            case "python":
                                AddToken(pythonStart, 1);
                                AddTokens(valueIds, 1);
                                AddToken(pythonEnd, 1);
                                break;
                            case "python_output":
                                // Python output is not supervised (mask = 0)
                                AddToken(outputStart, 0);
                                AddTokens(valueIds, 0);
                                AddToken(outputEnd, 0);
                                break;
                            default:
                                throw new InvalidOperationException($"Unknown part type: {part.Type}");
                        }
                    }
                }
                
                AddToken(assistantEnd, 1);
            }
        }

        // Truncate to max_tokens if necessary
        if (ids.Count > maxTokens)
        {
            ids = ids.Take(maxTokens).ToList();
            mask = mask.Take(maxTokens).ToList();
        }

        return (ids, mask);
    }

    public List<int> RenderForCompletion(Conversation conversation)
    {
        // Remove the last message (must be from assistant)
        if (conversation.Messages.Count == 0 || conversation.Messages[^1].Role != "assistant")
            throw new InvalidOperationException("Conversation must have at least one message and last message must be from assistant");

        // Create a modified conversation without the last message
        var modifiedConversation = new Conversation
        {
            Messages = conversation.Messages.Take(conversation.Messages.Count - 1).ToList()
        };

        // Render the conversation (without the last assistant message)
        var (ids, _) = RenderConversation(modifiedConversation);

        // Add assistant_start token to prime for completion
        int assistantStart = EncodeSpecial(SpecialTokens.AssistantStart);
        ids.Add(assistantStart);

        return ids;
    }

    public string VisualizeTokenization(List<int> ids, List<int> mask, bool withTokenId = false)
    {
        const string Red = "\u001b[91m";
        const string Green = "\u001b[92m";
        const string Reset = "\u001b[0m";
        const string Gray = "\u001b[90m";

        var tokens = new List<string>();
        
        for (int i = 0; i < ids.Count; i++)
        {
            int tokenId = ids[i];
            int maskVal = mask[i];
            
            string tokenStr = Decode(new List<int> { tokenId });
            string color = maskVal == 1 ? Green : Red;
            
            tokens.Add($"{color}{tokenStr}{Reset}");
            
            if (withTokenId)
            {
                tokens.Add($"{Gray}({tokenId}){Reset}");
            }
        }
        
        return string.Join("|", tokens);
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
