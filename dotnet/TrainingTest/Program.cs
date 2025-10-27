using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;
using NanoChat.Core;

Console.WriteLine("=== NanoChat Training Test ===\n");

// Create a tiny model for testing
var config = new GPTConfig
{
    SequenceLen = 128,
    VocabSize = 256,  // tiny vocab
    NLayer = 2,       // tiny model
    NHead = 2,
    NKVHead = 2,
    NEmbd = 128
};

Console.WriteLine("Creating model...");
var model = new GPT(config);

// Move to device
var device = cuda.is_available() ? CUDA : CPU;
Console.WriteLine($"Using device: {device}");
model.to(device);

// Don't call InitWeights - it converts to BFloat16 which causes dtype issues
// We'll use Float32 for this test

// Count parameters
var numParams = model.parameters().Sum(p => p.numel());
Console.WriteLine($"Model has {numParams:N0} parameters\n");

// Store initial weight value for verification
var firstParam = model.parameters().First();
var initialWeightSum = firstParam.sum().item<float>();
Console.WriteLine($"Initial weight sum (first param): {initialWeightSum:F6}");

// Create optimizer - using AdamW for all parameters
Console.WriteLine("Creating AdamW optimizer...");
var optimizer = optim.AdamW(model.parameters(), lr: 0.001, weight_decay: 0.01);

// Training loop with dummy data
const int numSteps = 10;
const int batchSize = 2;
const int seqLen = 64;

Console.WriteLine($"Training for {numSteps} steps...\n");

for (int step = 0; step < numSteps; step++)
{
    // Generate random dummy data
    var x = randint(0, config.VocabSize, new long[] { batchSize, seqLen }, device: device);
    var y = randint(0, config.VocabSize, new long[] { batchSize, seqLen }, device: device);

    // Forward pass
    var loss = model.forward(x, y);

    // Backward pass
    optimizer.zero_grad();
    loss.backward();

    // Optimizer step
    optimizer.step();

    // Log progress
    if (step % 2 == 0 || step == numSteps - 1)
    {
        Console.WriteLine($"Step {step + 1}/{numSteps} | Loss: {loss.item<float>():F6}");
    }

    // Clean up tensors
    x.Dispose();
    y.Dispose();
    loss.Dispose();
}

Console.WriteLine("\n✓ Training test completed successfully!");

// Verify weights actually changed
var finalWeightSum = firstParam.sum().item<float>();
Console.WriteLine($"\nFinal weight sum (first param): {finalWeightSum:F6}");
Console.WriteLine($"Weight change: {Math.Abs(finalWeightSum - initialWeightSum):F6}");
Console.WriteLine($"Weights changed: {Math.Abs(finalWeightSum - initialWeightSum) > 1e-6}");

// Run checkpoint test
TrainingTest.CheckpointTest.TestCheckpointSaveLoad();

Console.WriteLine("\n=== All tests completed successfully! ===");
