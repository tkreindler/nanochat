using NanoChat.Core;
using System;
using System.Collections.Generic;
using System.Linq;

namespace TrainingTest;

public static class ConversationTest
{
    public static void TestConversationRendering()
    {
        Console.WriteLine("\n=== Testing Conversation Rendering ===\n");
        
        var tokenizer = new StubTokenizer(vocabSize: 50304);
        
        // Test 1: Simple user/assistant conversation
        TestSimpleConversation(tokenizer);
        
        // Test 2: Conversation with system message
        TestConversationWithSystemMessage(tokenizer);
        
        // Test 3: Conversation with multi-part assistant message
        TestMultiPartConversation(tokenizer);
        
        // Test 4: RenderForCompletion
        TestRenderForCompletion(tokenizer);
        
        // Test 5: VisualizeTokenization
        TestVisualizeTokenization(tokenizer);
        
        Console.WriteLine("✓ All conversation rendering tests passed!\n");
    }
    
    private static void TestSimpleConversation(ITokenizer tokenizer)
    {
        Console.WriteLine("Test 1: Simple conversation");
        
        var conversation = new Conversation
        {
            Messages = new List<Message>
            {
                new Message { Role = "user", Content = "Hello!" },
                new Message { Role = "assistant", Content = "Hi there!" }
            }
        };
        
        conversation.Validate();
        var (ids, mask) = tokenizer.RenderConversation(conversation);
        
        Console.WriteLine($"  Token count: {ids.Count}");
        Console.WriteLine($"  Mask count: {mask.Count}");
        
        // Verify structure
        if (ids.Count != mask.Count)
            throw new Exception("IDs and mask lengths don't match!");
        
        // Should start with BOS token
        if (ids[0] != tokenizer.BosTokenId)
            throw new Exception($"Expected BOS token {tokenizer.BosTokenId}, got {ids[0]}");
        
        // Count mask values
        int userTokens = mask.Count(m => m == 0);
        int assistantTokens = mask.Count(m => m == 1);
        
        Console.WriteLine($"  User tokens (mask=0): {userTokens}");
        Console.WriteLine($"  Assistant tokens (mask=1): {assistantTokens}");
        Console.WriteLine("  ✓ Simple conversation test passed\n");
    }
    
    private static void TestConversationWithSystemMessage(ITokenizer tokenizer)
    {
        Console.WriteLine("Test 2: Conversation with system message");
        
        var conversation = new Conversation
        {
            Messages = new List<Message>
            {
                new Message { Role = "system", Content = "You are a helpful assistant." },
                new Message { Role = "user", Content = "Hello!" },
                new Message { Role = "assistant", Content = "Hi there!" }
            }
        };
        
        conversation.Validate();
        var (ids, mask) = tokenizer.RenderConversation(conversation);
        
        Console.WriteLine($"  Token count: {ids.Count}");
        
        // Should start with BOS token
        if (ids[0] != tokenizer.BosTokenId)
            throw new Exception($"Expected BOS token {tokenizer.BosTokenId}, got {ids[0]}");
        
        Console.WriteLine("  ✓ System message conversation test passed\n");
    }
    
    private static void TestMultiPartConversation(ITokenizer tokenizer)
    {
        Console.WriteLine("Test 3: Multi-part assistant message");
        
        var conversation = new Conversation
        {
            Messages = new List<Message>
            {
                new Message { Role = "user", Content = "Calculate 2+2" },
                new Message 
                { 
                    Role = "assistant", 
                    Content = new List<MessagePart>
                    {
                        new MessagePart { Type = "text", Text = "Let me calculate that:" },
                        new MessagePart { Type = "python", Text = "result = 2 + 2\nprint(result)" },
                        new MessagePart { Type = "python_output", Text = "4" },
                        new MessagePart { Type = "text", Text = "The answer is 4." }
                    }
                }
            }
        };
        
        conversation.Validate();
        var (ids, mask) = tokenizer.RenderConversation(conversation);
        
        Console.WriteLine($"  Token count: {ids.Count}");
        
        // Count special tokens
        int pythonStart = tokenizer.EncodeSpecial(SpecialTokens.PythonStart);
        int pythonEnd = tokenizer.EncodeSpecial(SpecialTokens.PythonEnd);
        int outputStart = tokenizer.EncodeSpecial(SpecialTokens.OutputStart);
        int outputEnd = tokenizer.EncodeSpecial(SpecialTokens.OutputEnd);
        
        int pythonStartCount = ids.Count(id => id == pythonStart);
        int pythonEndCount = ids.Count(id => id == pythonEnd);
        int outputStartCount = ids.Count(id => id == outputStart);
        int outputEndCount = ids.Count(id => id == outputEnd);
        
        Console.WriteLine($"  Python start tokens: {pythonStartCount}");
        Console.WriteLine($"  Python end tokens: {pythonEndCount}");
        Console.WriteLine($"  Output start tokens: {outputStartCount}");
        Console.WriteLine($"  Output end tokens: {outputEndCount}");
        
        if (pythonStartCount != 1 || pythonEndCount != 1)
            throw new Exception("Expected exactly one python_start and python_end token!");
        
        if (outputStartCount != 1 || outputEndCount != 1)
            throw new Exception("Expected exactly one output_start and output_end token!");
        
        Console.WriteLine("  ✓ Multi-part conversation test passed\n");
    }
    
    private static void TestRenderForCompletion(ITokenizer tokenizer)
    {
        Console.WriteLine("Test 4: RenderForCompletion");
        
        var conversation = new Conversation
        {
            Messages = new List<Message>
            {
                new Message { Role = "user", Content = "Hello!" },
                new Message { Role = "assistant", Content = "Hi there!" },
                new Message { Role = "user", Content = "How are you?" },
                new Message { Role = "assistant", Content = "I'm doing well!" }
            }
        };
        
        conversation.Validate();
        var ids = tokenizer.RenderForCompletion(conversation);
        
        Console.WriteLine($"  Token count: {ids.Count}");
        
        // Should end with assistant_start token
        int assistantStart = tokenizer.EncodeSpecial(SpecialTokens.AssistantStart);
        if (ids[^1] != assistantStart)
            throw new Exception($"Expected last token to be assistant_start ({assistantStart}), got {ids[^1]}");
        
        // Should not contain the last assistant message's content tokens
        // (difficult to verify exactly with stub tokenizer, but we can check structure)
        
        Console.WriteLine($"  Last token ID: {ids[^1]} (assistant_start)");
        Console.WriteLine("  ✓ RenderForCompletion test passed\n");
    }
    
    private static void TestVisualizeTokenization(ITokenizer tokenizer)
    {
        Console.WriteLine("Test 5: VisualizeTokenization");
        
        var conversation = new Conversation
        {
            Messages = new List<Message>
            {
                new Message { Role = "user", Content = "Hi" },
                new Message { Role = "assistant", Content = "Hello!" }
            }
        };
        
        conversation.Validate();
        var (ids, mask) = tokenizer.RenderConversation(conversation);
        
        // Test without token IDs
        string visual1 = tokenizer.VisualizeTokenization(ids, mask, withTokenId: false);
        Console.WriteLine($"  Without IDs (length: {visual1.Length})");
        
        // Test with token IDs
        string visual2 = tokenizer.VisualizeTokenization(ids, mask, withTokenId: true);
        Console.WriteLine($"  With IDs (length: {visual2.Length})");
        
        // Verify the visualization contains expected elements
        if (!visual1.Contains("|"))
            throw new Exception("Expected '|' separator in visualization!");
        
        if (!visual2.Contains("(") || !visual2.Contains(")"))
            throw new Exception("Expected token IDs in parentheses when withTokenId=true!");
        
        Console.WriteLine("  ✓ VisualizeTokenization test passed\n");
    }
}
