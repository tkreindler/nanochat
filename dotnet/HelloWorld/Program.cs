using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core;

Console.WriteLine("=== NanoChat Inference Test ===");
Console.WriteLine($"CUDA Available: {cuda.is_available()}");

if (!cuda.is_available())
{
    Console.WriteLine("WARNING: CUDA is not available! Using CPU.");
}

Console.WriteLine($"CUDA Device Count: {cuda.device_count()}");

// Set device
var device = cuda.is_available() ? CUDA : CPU;
Console.WriteLine($"\nUsing device: {device}");

// Create a small GPT model for testing
Console.WriteLine("\nCreating GPT model...");
var config = new GPTConfig
{
    VocabSize = 50304,
    NLayer = 2,        // Small model for testing
    NHead = 4,
    NEmbd = 128,
    NKVHead = 4,
    SequenceLen = 256
};

using var model = new GPT(config);
model.to(device);
model.eval();

Console.WriteLine($"Model created with {config.VocabSize} vocab size, {config.NLayer} layers");

// Create tokenizer
var tokenizer = new StubTokenizer(config.VocabSize);
Console.WriteLine($"Tokenizer created with vocab size: {tokenizer.VocabSize}");

// Create engine
var engine = new Engine(model, tokenizer);
Console.WriteLine("Engine created");

// Test generation with dummy tokens
Console.WriteLine("\nTesting generation...");
var promptTokens = new List<long> { 1, 2, 3, 4, 5 }; // Dummy token IDs
Console.WriteLine($"Input tokens: [{string.Join(", ", promptTokens)}]");

Console.WriteLine("\nGenerating 10 tokens with temperature=0.8, topK=40...");
var results = engine.GenerateBatch(
    tokens: promptTokens,
    numSamples: 2,
    maxTokens: 10,
    temperature: 0.8,
    topK: 40,
    seed: 42
);

Console.WriteLine($"\nGenerated {results.results.Count} samples:");
for (int i = 0; i < results.results.Count; i++)
{
    Console.WriteLine($"  Sample {i + 1}: [{string.Join(", ", results.results[i])}]");
    Console.WriteLine($"    Masks: [{string.Join(", ", results.masks[i])}]");
}

Console.WriteLine("\nInference test completed successfully!");
