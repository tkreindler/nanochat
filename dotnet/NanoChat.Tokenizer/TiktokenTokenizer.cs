using System.Collections.Concurrent;
using SharpToken;

namespace NanoChat.Tokenizer;

/// <summary>
/// Tokenizer implementation using SharpToken (tiktoken port for C#)
/// Port of RustBPETokenizer from tokenizer.py
/// </summary>
public class TiktokenTokenizer : ITokenizer
{
    private readonly GptEncoding _encoding;
    private readonly int _bosTokenId;
    private readonly Dictionary<string, int> _specialTokensCache;
    private int? _vocabSizeCache;
    
    private TiktokenTokenizer(GptEncoding encoding, string bosToken)
    {
        _encoding = encoding;
        _specialTokensCache = new Dictionary<string, int>();
        
        // Try to encode the BOS token
        // For tiktoken encodings, special tokens are encoded differently
        _bosTokenId = TryEncodeSpecialToken(bosToken);
    }
    
    /// <summary>
    /// Train a new tokenizer from an iterator of text
    /// NOTE: This creates a trained BPE model but SharpToken doesn't support custom encodings.
    /// For now, this throws NotImplementedException. Future implementations could:
    /// - Implement a custom tiktoken-compatible encoder
    /// - Use Python interop to create tiktoken files
    /// - Serialize to a format that can be loaded via custom decoder
    /// </summary>
    public static TiktokenTokenizer TrainFromIterator(
        IEnumerable<string> textIterator, 
        int vocabSize,
        string? pattern = null,
        int bufferSize = 8192)
    {
        pattern ??= Tokenizer.SpecialTokens.SplitPattern;
        
        // Step 1: Train using BPETrainer
        var trainer = new BPETrainer(pattern);
        var mergeableRanks = trainer.TrainFromIterator(textIterator, vocabSize - Tokenizer.SpecialTokens.All.Length, bufferSize);
        
        // Step 2: Add special tokens  
        int tokensOffset = mergeableRanks.Count;
        var specialTokens = new Dictionary<string, int>();
        for (int i = 0; i < Tokenizer.SpecialTokens.All.Length; i++)
        {
            specialTokens[Tokenizer.SpecialTokens.All[i]] = tokensOffset + i;
        }
        
        // TODO: SharpToken doesn't expose a constructor to create custom encodings
        // We need to either:
        // 1. Implement our own tiktoken-compatible encoder (~500 lines)
        // 2. Save to Python tiktoken format and load it back
        // 3. Use a different tokenizer library that supports custom models
        
        throw new NotImplementedException(
            "Custom tokenizer training is not yet implemented. " +
            "SharpToken doesn't support creating encodings from custom mergeable ranks. " +
            "Use FromPretrained() with standard models like 'gpt-4' or 'cl100k_base' for now. " +
            $"Trained {mergeableRanks.Count} base tokens + {specialTokens.Count} special tokens.");
    }
    
    /// <summary>
    /// Load tokenizer from directory
    /// </summary>
    public static TiktokenTokenizer FromDirectory(string tokenizerDir)
    {
        string picklePath = Path.Combine(tokenizerDir, "tokenizer.pkl");
        
        if (!File.Exists(picklePath))
            throw new FileNotFoundException($"Tokenizer not found at {picklePath}");
        
        // TODO: Implement pickle loading for C#
        // For now, throw a helpful error
        throw new NotImplementedException(
            "Loading from pickle not yet implemented. " +
            "Use TrainFromIterator to create a new tokenizer, or implement pickle deserialization.");
    }
    
    /// <summary>
    /// Load a pretrained tokenizer (e.g., cl100k_base, gpt-4, gpt-3.5-turbo)
    /// </summary>
    public static TiktokenTokenizer FromPretrained(string modelName)
    {
        // SharpToken supports loading by encoding name or model name
        GptEncoding encoding;
        
        try
        {
            // Try as encoding name first (cl100k_base, p50k_base, etc.)
            encoding = GptEncoding.GetEncoding(modelName);
        }
        catch
        {
            // Fall back to model name (gpt-4, gpt-3.5-turbo, etc.)
            encoding = GptEncoding.GetEncodingForModel(modelName);
        }
        
        // For pretrained models, use "<|endoftext|>" as BOS token
        // Note: Most tiktoken models use this as the special token
        return new TiktokenTokenizer(encoding, "<|endoftext|>");
    }
    
    /// <summary>
    /// Try to encode a special token. Returns a cached or newly encoded token ID.
    /// </summary>
    private int TryEncodeSpecialToken(string token)
    {
        if (_specialTokensCache.TryGetValue(token, out int cachedId))
            return cachedId;
        
        // Encode the special token by allowing it
        var allowedSet = new HashSet<string> { token };
        var ids = _encoding.Encode(token, allowedSet, new HashSet<string>());
        
        if (ids.Count != 1)
            throw new InvalidOperationException(
                $"Special token '{token}' encoded to {ids.Count} tokens instead of 1. " +
                "This tokenizer does not have the required special tokens for conversation rendering. " +
                "Use a tokenizer trained with train_from_iterator() or loaded from a trained directory.");
        
        int tokenId = ids[0];
        _specialTokensCache[token] = tokenId;
        return tokenId;
    }
    
    public int VocabSize
    {
        get
        {
            if (_vocabSizeCache.HasValue)
                return _vocabSizeCache.Value;
            
            // Estimate vocab size by trying to decode sequential integers
            // This is a workaround since SharpToken doesn't expose VocabSize
            // For cl100k_base it's 100277, for gpt-4o it's 100256
            // We can estimate or use known values
            _vocabSizeCache = EstimateVocabSize();
            return _vocabSizeCache.Value;
        }
    }
    
    private int EstimateVocabSize()
    {
        // Binary search for the exact vocab size
        // When SharpToken decodes an invalid token ID, it returns an empty string
        // The vocab size is the smallest ID that returns empty
        int low = 50000;
        int high = 250000;
        
        while (low < high)
        {
            int mid = low + (high - low) / 2;
            var decoded = _encoding.Decode(new List<int> { mid });
            
            if (decoded.Length > 0)
            {
                // Valid token, search higher
                low = mid + 1;
            }
            else
            {
                // Invalid token (empty string), search lower
                high = mid;
            }
        }
        
        return low;
    }
    
    public IReadOnlySet<string> SpecialTokens
    {
        get
        {
            // Return the special tokens we know about from our constants
            // SharpToken doesn't expose the special tokens dict directly
            return new HashSet<string>(Tokenizer.SpecialTokens.All);
        }
    }
    
    public int BosTokenId => _bosTokenId;
    
    /// <summary>
    /// Encode a single string into token IDs
    /// </summary>
    public List<int> Encode(string text, object? prepend = null, object? append = null)
    {
        int? prependId = prepend switch
        {
            int id => id,
            string token => TryEncodeSpecialToken(token),
            _ => null
        };
        
        int? appendId = append switch
        {
            int id => id,
            string token => TryEncodeSpecialToken(token),
            _ => null
        };
        
        // Encode without allowing any special tokens in the text itself
        var ids = _encoding.Encode(text, new HashSet<string>(), new HashSet<string>());
        var result = new List<int>(ids.Count + 2);
        
        if (prependId.HasValue)
            result.Add(prependId.Value);
        
        result.AddRange(ids);
        
        if (appendId.HasValue)
            result.Add(appendId.Value);
        
        return result;
    }
    
    /// <summary>
    /// Encode multiple strings in parallel
    /// </summary>
    public List<List<int>> Encode(
        IEnumerable<string> texts, 
        object? prepend = null, 
        object? append = null, 
        int numThreads = 8)
    {
        int? prependId = prepend switch
        {
            int id => id,
            string token => TryEncodeSpecialToken(token),
            _ => null
        };
        
        int? appendId = append switch
        {
            int id => id,
            string token => TryEncodeSpecialToken(token),
            _ => null
        };
        
        var textList = texts.ToList();
        var results = new ConcurrentBag<(int index, List<int> ids)>();
        
        var emptySet = new HashSet<string>();
        
        // Parallel encoding
        var options = new ParallelOptions { MaxDegreeOfParallelism = numThreads };
        Parallel.For(0, textList.Count, options, i =>
        {
            var ids = _encoding.Encode(textList[i], emptySet, emptySet);
            var result = new List<int>(ids.Count + 2);
            
            if (prependId.HasValue)
                result.Add(prependId.Value);
            
            result.AddRange(ids);
            
            if (appendId.HasValue)
                result.Add(appendId.Value);
            
            results.Add((i, result));
        });
        
        // Return in original order
        return results
            .OrderBy(x => x.index)
            .Select(x => x.ids)
            .ToList();
    }
    
    public int EncodeSpecial(string token)
    {
        return TryEncodeSpecialToken(token);
    }
    
    public string Decode(IEnumerable<int> ids)
    {
        return _encoding.Decode(ids);
    }
    
    public string IdToToken(int id)
    {
        // Decode single token
        return _encoding.Decode(new List<int> { id });
    }
    
    /// <summary>
    /// Save tokenizer to disk
    /// </summary>
    public void Save(string directory)
    {
        Directory.CreateDirectory(directory);
        
        // TODO: Implement proper serialization
        // For now, save as JSON with basic info
        string jsonPath = Path.Combine(directory, "tokenizer.json");
        
        var info = new
        {
            vocab_size = VocabSize,
            special_tokens = SpecialTokens.ToList(),
            bos_token_id = BosTokenId,
            note = "Full serialization not yet implemented. Use Python interop or custom format."
        };
        
        string json = System.Text.Json.JsonSerializer.Serialize(info, new System.Text.Json.JsonSerializerOptions 
        { 
            WriteIndented = true 
        });
        
        File.WriteAllText(jsonPath, json);
        Console.WriteLine($"Saved tokenizer info to {jsonPath}");
        Console.WriteLine("Warning: Full tokenizer state not saved. Implement proper serialization for production use.");
    }
    
    /// <summary>
    /// Render a conversation for training (SFT).
    /// Returns token IDs and a mask where 1 indicates tokens the assistant should learn from.
    /// 
    /// NOTE: This method requires a tokenizer that has been trained with the custom special tokens
    /// (e.g., using train_from_iterator or loaded from a trained directory). Pretrained tokenizers
    /// like GPT-4 do not have these special tokens and will throw an exception.
    /// </summary>
    public (List<int> ids, List<int> mask) RenderConversation(Core.Conversation conversation, int maxTokens = 2048)
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
            var mergedFirstUser = new Core.Message 
            { 
                Role = "user", 
                Content = mergedContent 
            };
            
            messages = new List<Core.Message> { mergedFirstUser };
            messages.AddRange(conversation.Messages.Skip(2));
        }

        // Get special token IDs
        int bos = BosTokenId;
        int userStart = TryEncodeSpecialToken(Core.SpecialTokens.UserStart);
        int userEnd = TryEncodeSpecialToken(Core.SpecialTokens.UserEnd);
        int assistantStart = TryEncodeSpecialToken(Core.SpecialTokens.AssistantStart);
        int assistantEnd = TryEncodeSpecialToken(Core.SpecialTokens.AssistantEnd);
        int pythonStart = TryEncodeSpecialToken(Core.SpecialTokens.PythonStart);
        int pythonEnd = TryEncodeSpecialToken(Core.SpecialTokens.PythonEnd);
        int outputStart = TryEncodeSpecialToken(Core.SpecialTokens.OutputStart);
        int outputEnd = TryEncodeSpecialToken(Core.SpecialTokens.OutputEnd);

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

    /// <summary>
    /// Render a conversation for completion (RL/inference).
    /// Removes the last assistant message and adds assistant_start token.
    /// </summary>
    public List<int> RenderForCompletion(Core.Conversation conversation)
    {
        // Remove the last message (must be from assistant)
        if (conversation.Messages.Count == 0 || conversation.Messages[^1].Role != "assistant")
            throw new InvalidOperationException("Conversation must have at least one message and last message must be from assistant");

        // Create a modified conversation without the last message
        var modifiedConversation = new Core.Conversation
        {
            Messages = conversation.Messages.Take(conversation.Messages.Count - 1).ToList()
        };

        // Render the conversation (without the last assistant message)
        var (ids, _) = RenderConversation(modifiedConversation);

        // Add assistant_start token to prime for completion
        int assistantStart = TryEncodeSpecialToken(Core.SpecialTokens.AssistantStart);
        ids.Add(assistantStart);

        return ids;
    }

    /// <summary>
    /// Visualize tokenization with color coding (for debugging).
    /// Green = assistant tokens (mask=1), Red = user/system tokens (mask=0).
    /// </summary>
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
}
