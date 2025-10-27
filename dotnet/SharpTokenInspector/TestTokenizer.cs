using System;
using NanoChat.Tokenizer;

namespace SharpTokenInspector;

public static class TestTokenizer
{
    public static void Run()
    {
        Console.WriteLine("=== Testing TiktokenTokenizer ===\n");
        
        // Test 1: Load pretrained model
        Console.WriteLine("Test 1: Loading cl100k_base encoding...");
        var tokenizer = TiktokenTokenizer.FromPretrained("cl100k_base");
        Console.WriteLine($"✓ Loaded. Vocab size: {tokenizer.VocabSize}");
        Console.WriteLine($"✓ BOS token ID: {tokenizer.BosTokenId}");
        Console.WriteLine($"✓ Special tokens count: {tokenizer.SpecialTokens.Count}");
        
        // Test 2: Basic encoding/decoding
        Console.WriteLine("\nTest 2: Basic encoding/decoding...");
        string testText = "Hello, world! This is a test.";
        var encoded = tokenizer.Encode(testText);
        Console.WriteLine($"✓ Encoded '{testText}' to {encoded.Count} tokens: [{string.Join(", ", encoded)}]");
        
        var decoded = tokenizer.Decode(encoded);
        Console.WriteLine($"✓ Decoded back: '{decoded}'");
        
        if (decoded == testText)
            Console.WriteLine("✓ Round-trip successful!");
        else
            Console.WriteLine($"✗ Round-trip failed! Got: '{decoded}'");
        
        // Test 3: Encoding with prepend/append
        Console.WriteLine("\nTest 3: Encoding with BOS token...");
        var encodedWithBos = tokenizer.Encode(testText, prepend: tokenizer.BosTokenId);
        Console.WriteLine($"✓ With BOS: {encodedWithBos.Count} tokens: [{string.Join(", ", encodedWithBos)}]");
        
        // Test 4: Special token encoding
        Console.WriteLine("\nTest 4: Special token encoding...");
        try
        {
            int endToken = tokenizer.EncodeSpecial("<|endoftext|>");
            Console.WriteLine($"✓ '<|endoftext|>' encoded to: {endToken}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Failed to encode special token: {ex.Message}");
        }
        
        // Test 5: Batch encoding
        Console.WriteLine("\nTest 5: Batch encoding...");
        var texts = new[] { "First text", "Second text", "Third text" };
        var batchEncoded = tokenizer.Encode(texts);
        Console.WriteLine($"✓ Encoded {texts.Length} texts:");
        for (int i = 0; i < texts.Length; i++)
        {
            Console.WriteLine($"  [{i}] '{texts[i]}' -> {batchEncoded[i].Count} tokens");
        }
        
        // Test 6: Try different model
        Console.WriteLine("\nTest 6: Loading gpt-4 model...");
        try
        {
            var gpt4Tokenizer = TiktokenTokenizer.FromPretrained("gpt-4");
            var gpt4Encoded = gpt4Tokenizer.Encode(testText);
            Console.WriteLine($"✓ GPT-4 encoded to {gpt4Encoded.Count} tokens: [{string.Join(", ", gpt4Encoded)}]");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"✗ Failed to load gpt-4: {ex.Message}");
        }
        
        Console.WriteLine("\n=== All Tests Complete ===");
    }
}
