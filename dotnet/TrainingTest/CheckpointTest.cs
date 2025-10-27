using TorchSharp;
using static TorchSharp.torch;
using NanoChat.Core;

namespace TrainingTest;

public static class CheckpointTest
{
    public static void TestCheckpointSaveLoad()
    {
        Console.WriteLine("\n=== Testing Checkpoint Save/Load ===\n");

        var device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        Console.WriteLine($"Using device: {device}");

        // Create a tiny model for testing
        var config = new GPTConfig
        {
            SequenceLen = 128,
            VocabSize = 256,
            NLayer = 2,
            NHead = 4,
            NKVHead = 4,
            NEmbd = 128
        };

        Console.WriteLine("Creating model...");
        var model = new GPT(config);
        model.to(device);

        // Extract state dict from model
        var stateDict = new Dictionary<string, Tensor>();
        var paramsBefore = new Dictionary<string, Tensor>();
        foreach (var (name, param) in model.named_parameters())
        {
            stateDict[name] = param.detach().cpu(); // Save to CPU for checkpoint
            paramsBefore[name] = param.clone().detach(); // Keep copy for comparison
        }

        Console.WriteLine($"Model has {stateDict.Count} parameters");
        Console.WriteLine($"First 5 parameter names: {string.Join(", ", stateDict.Keys.Take(5))}");

        // Create checkpoint metadata
        var metadata = new Dictionary<string, object>
        {
            ["step"] = 100,
            ["val_loss"] = 2.5f,
            ["model_config"] = new Dictionary<string, object>
            {
                ["SequenceLen"] = config.SequenceLen,
                ["VocabSize"] = config.VocabSize,
                ["NLayer"] = config.NLayer,
                ["NHead"] = config.NHead,
                ["NKVHead"] = config.NKVHead,
                ["NEmbd"] = config.NEmbd
            },
            ["user_config"] = new Dictionary<string, object>
            {
                ["learning_rate"] = 0.001,
                ["batch_size"] = 32
            }
        };

        // Save checkpoint
        var checkpointDir = "/tmp/checkpoint_test";
        if (Directory.Exists(checkpointDir))
        {
            Directory.Delete(checkpointDir, true);
        }
        
        Console.WriteLine($"\nSaving checkpoint to {checkpointDir}...");
        CheckpointManager.SaveCheckpoint(checkpointDir, 100, stateDict, null, metadata);
        
        // Save a few more checkpoints to test FindLastStep
        metadata["step"] = 50;
        CheckpointManager.SaveCheckpoint(checkpointDir, 50, stateDict, null, metadata);
        metadata["step"] = 200;
        CheckpointManager.SaveCheckpoint(checkpointDir, 200, stateDict, null, metadata);

        // Load checkpoint
        Console.WriteLine("\nLoading checkpoint...");
        var (loadedStateDict, loadedOptimizerData, loadedMetadata) = 
            CheckpointManager.LoadCheckpoint(checkpointDir, 100, device, loadOptimizer: false);

        // Verify metadata
        Console.WriteLine("\nVerifying metadata...");
        Console.WriteLine($"  Step: {loadedMetadata["step"]} (expected: 100)");
        Console.WriteLine($"  ValLoss: {loadedMetadata["val_loss"]} (expected: 2.5)");

        // Create a new model and load the state dict
        Console.WriteLine("\nCreating new model and loading state...");
        var loadedModel = new GPT(config);
        loadedModel.to(device);
        
        // Load state dict into model (use no_grad to avoid autograd issues)
        using (var _ = torch.no_grad())
        {
            var modelParams = loadedModel.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            foreach (var (name, tensor) in loadedStateDict)
            {
                if (modelParams.TryGetValue(name, out var param))
                {
                    param.copy_(tensor);
                }
            }
        }

        // Verify weights match
        Console.WriteLine("\nVerifying weights...");
        var paramsAfter = new Dictionary<string, Tensor>();
        foreach (var (name, param) in loadedModel.named_parameters())
        {
            paramsAfter[name] = param.clone().detach();
        }

        bool allMatch = true;
        int checkedCount = 0;
        foreach (var name in paramsBefore.Keys.Take(5)) // Check first 5 params
        {
            if (!paramsAfter.ContainsKey(name))
            {
                Console.WriteLine($"  ❌ Parameter {name} not found in loaded model!");
                allMatch = false;
                continue;
            }

            var before = paramsBefore[name];
            var after = paramsAfter[name];
            
            var diff = (before - after).abs().max().item<float>();
            if (diff > 1e-6f)
            {
                Console.WriteLine($"  ❌ Parameter {name}: max diff = {diff}");
                allMatch = false;
            }
            checkedCount++;
        }

        if (allMatch)
        {
            Console.WriteLine($"  ✅ All {checkedCount} checked parameters match!");
        }

        // Test FindLastStep
        Console.WriteLine("\nTesting FindLastStep...");
        var lastStep = CheckpointManager.FindLastStep(checkpointDir);
        Console.WriteLine($"  Found last step: {lastStep} (expected: 200)");
        
        Console.WriteLine($"\nCheckpoint files in {checkpointDir}:");
        foreach (var file in Directory.GetFiles(checkpointDir).OrderBy(f => f))
        {
            Console.WriteLine($"  {Path.GetFileName(file)}");
        }

        // Clean up
        if (Directory.Exists(checkpointDir))
        {
            Directory.Delete(checkpointDir, true);
            Console.WriteLine("\nCleaned up test checkpoint directory");
        }

        Console.WriteLine("\n=== Checkpoint Test Complete ===\n");
    }
}
