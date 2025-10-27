using System.Text.Json;
using System.Text.RegularExpressions;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace NanoChat.Core;

/// <summary>
/// Utilities for saving and loading model/optim/state checkpoints.
/// </summary>
public static class CheckpointManager
{
    private static void Log0(string message)
    {
        // Only log on rank 0 (for distributed training compatibility)
        var rank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0");
        if (rank == 0)
        {
            Console.WriteLine(message);
        }
    }

    /// <summary>
    /// Save a checkpoint with model state, optimizer state, and metadata
    /// </summary>
    public static void SaveCheckpoint(
        string checkpointDir,
        int step,
        Dictionary<string, Tensor> modelData,
        Dictionary<string, object>? optimizerData,
        Dictionary<string, object> metaData)
    {
        // Prevent footguns for distributed training
        var rank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0");
        if (rank != 0)
        {
            throw new InvalidOperationException("save_checkpoint should only be called on rank 0");
        }

        Directory.CreateDirectory(checkpointDir);

        // Save the model state (parameters)
        var modelPath = Path.Combine(checkpointDir, $"model_{step:D6}.pt");
        SaveStateDictPt(modelPath, modelData);
        Log0($"Saved model file to: {modelPath}");

        // Save the optimizer state (useful for SFT or any other fine-tuning)
        if (optimizerData != null)
        {
            var optimizerPath = Path.Combine(checkpointDir, $"optim_{step:D6}.pt");
            SaveOptimizerStatePt(optimizerPath, optimizerData);
            Log0($"Saved optimizer file to: {optimizerPath}");
        }

        // Save the metadata dict as json
        var metaPath = Path.Combine(checkpointDir, $"meta_{step:D6}.json");
        var options = new JsonSerializerOptions { WriteIndented = true };
        var json = JsonSerializer.Serialize(metaData, options);
        File.WriteAllText(metaPath, json);
        Log0($"Saved metadata file to: {metaPath}");
    }

    /// <summary>
    /// Load a checkpoint from disk
    /// </summary>
    public static (Dictionary<string, Tensor> modelData, Dictionary<string, object>? optimizerData, Dictionary<string, object> metaData) 
        LoadCheckpoint(
            string checkpointDir,
            int step,
            Device device,
            bool loadOptimizer = false)
    {
        // Load the model state
        var modelPath = Path.Combine(checkpointDir, $"model_{step:D6}.pt");
        var modelData = LoadStateDictPt(modelPath, device);

        // Load the optimizer state if requested
        Dictionary<string, object>? optimizerData = null;
        if (loadOptimizer)
        {
            var optimizerPath = Path.Combine(checkpointDir, $"optim_{step:D6}.pt");
            optimizerData = LoadOptimizerStatePt(optimizerPath, device);
        }

        // Load the metadata
        var metaPath = Path.Combine(checkpointDir, $"meta_{step:D6}.json");
        var json = File.ReadAllText(metaPath);
        var metaData = JsonSerializer.Deserialize<Dictionary<string, object>>(json)
            ?? throw new InvalidOperationException("Failed to deserialize metadata");

        return (modelData, optimizerData, metaData);
    }

    /// <summary>
    /// A bunch of repetitive code to build a model from a given checkpoint.
    /// Returns:
    /// - base model - uncompiled, not wrapped in DDP
    /// - tokenizer (TODO: implement)
    /// - meta data saved during base model training
    /// </summary>
    public static (GPT model, object? tokenizer, Dictionary<string, object> metaData) 
        BuildModel(
            string checkpointDir,
            int step,
            Device device,
            string phase)
    {
        if (phase != "train" && phase != "eval")
        {
            throw new ArgumentException($"Invalid phase: {phase}");
        }

        var (modelData, _, metaData) = LoadCheckpoint(checkpointDir, step, device, loadOptimizer: false);

        // Hack: fix torch compile issue, which prepends all keys with _orig_mod.
        var cleanedModelData = new Dictionary<string, Tensor>();
        foreach (var (key, value) in modelData)
        {
            var cleanKey = key.StartsWith("_orig_mod.") ? key.Substring(10) : key;
            cleanedModelData[cleanKey] = value;
        }

        // Extract model config from metadata
        if (!metaData.TryGetValue("model_config", out var modelConfigObj))
        {
            throw new InvalidOperationException("model_config not found in metadata");
        }

        var modelConfigJson = JsonSerializer.Serialize(modelConfigObj);
        var modelConfig = JsonSerializer.Deserialize<GPTConfig>(modelConfigJson)
            ?? throw new InvalidOperationException("Failed to deserialize model config");

        Log0($"Building model with config: {modelConfigJson}");

        // Create model (TorchSharp doesn't have meta device, so we create on CPU)
        var model = new GPT(modelConfig);

        // Load the model state
        model.to(torch.CPU); // Load on CPU first
        model.InitWeights(); // note: this is dumb, but we need to init the rotary embeddings. TODO: fix model re-init
        
        // Load state dict
        LoadStateDict(model, cleanedModelData);
        
        // Move to target device
        model.to(device);

        // Put the model in the right training phase / mode
        if (phase == "eval")
        {
            model.eval();
        }
        else
        {
            model.train();
        }

        // Load the Tokenizer (TODO: implement tokenizer)
        object? tokenizer = null;

        // Sanity check: compatibility between model and tokenizer (TODO: when tokenizer is implemented)
        // if (tokenizer != null && tokenizer.GetVocabSize() != modelConfig.VocabSize) { ... }

        return (model, tokenizer, metaData);
    }

    /// <summary>
    /// Attempt to guess the model tag: take the biggest model available
    /// </summary>
    public static string FindLargestModel(string checkpointDir)
    {
        var modelTags = Directory.GetDirectories(checkpointDir)
            .Select(Path.GetFileName)
            .Where(name => name != null)
            .Cast<string>()
            .ToList();

        if (modelTags.Count == 0)
        {
            throw new FileNotFoundException($"No checkpoints found in {checkpointDir}");
        }

        // 1) normally all model tags are of the form d<number>, try that first:
        var candidates = new List<(int depth, string tag)>();
        foreach (var modelTag in modelTags)
        {
            var match = Regex.Match(modelTag, @"^d(\d+)$");
            if (match.Success)
            {
                var modelDepth = int.Parse(match.Groups[1].Value);
                candidates.Add((modelDepth, modelTag));
            }
        }

        if (candidates.Count > 0)
        {
            candidates.Sort((a, b) => b.depth.CompareTo(a.depth)); // descending
            return candidates[0].tag;
        }

        // 2) if that failed, take the most recently updated model:
        modelTags.Sort((a, b) =>
        {
            var timeA = Directory.GetLastWriteTime(Path.Combine(checkpointDir, a));
            var timeB = Directory.GetLastWriteTime(Path.Combine(checkpointDir, b));
            return timeB.CompareTo(timeA); // descending
        });

        return modelTags[0];
    }

    /// <summary>
    /// Look into checkpoint_dir and find model_*.pt with the highest step
    /// </summary>
    public static int FindLastStep(string checkpointDir)
    {
        var checkpointFiles = Directory.GetFiles(checkpointDir, "model_*.pt");
        if (checkpointFiles.Length == 0)
        {
            throw new FileNotFoundException($"No checkpoints found in {checkpointDir}");
        }

        var lastStep = checkpointFiles
            .Select(f => Path.GetFileName(f))
            .Select(f => f.Split('_').Last().Split('.').First())
            .Select(int.Parse)
            .Max();

        return lastStep;
    }

    // -----------------------------------------------------------------------------
    // Convenience functions that take into account nanochat's directory structure

    /// <summary>
    /// Load a model from a checkpoints directory with optional model_tag and step
    /// </summary>
    public static (GPT model, object? tokenizer, Dictionary<string, object> metaData) 
        LoadModelFromDir(
            string checkpointsDir,
            Device device,
            string phase,
            string? modelTag = null,
            int? step = null)
    {
        if (modelTag == null)
        {
            // guess the model tag by defaulting to the largest model
            modelTag = FindLargestModel(checkpointsDir);
            Log0($"No model tag provided, guessing model tag: {modelTag}");
        }

        var checkpointDir = Path.Combine(checkpointsDir, modelTag);

        if (step == null)
        {
            // guess the step by defaulting to the last step
            step = FindLastStep(checkpointDir);
        }

        if (step == null)
        {
            throw new InvalidOperationException($"No checkpoints found in {checkpointDir}");
        }

        // build the model
        Log0($"Loading model from {checkpointDir} with step {step}");
        var (model, tokenizer, metaData) = BuildModel(checkpointDir, step.Value, device, phase);
        return (model, tokenizer, metaData);
    }

    /// <summary>
    /// Load a model from a predefined source directory
    /// </summary>
    public static (GPT model, object? tokenizer, Dictionary<string, object> metaData) 
        LoadModel(
            string source,
            Device device,
            string phase,
            string? modelTag = null,
            int? step = null,
            string? baseDir = null)
    {
        var modelDir = source switch
        {
            "base" => "base_checkpoints",
            "mid" => "mid_checkpoints",
            "sft" => "chatsft_checkpoints",
            "rl" => "chatrl_checkpoints",
            _ => throw new ArgumentException($"Unknown source: {source}")
        };

        baseDir ??= Common.GetBaseDir();
        var checkpointsDir = Path.Combine(baseDir, modelDir);
        return LoadModelFromDir(checkpointsDir, device, phase, modelTag, step);
    }

    // -----------------------------------------------------------------------------
    // Helper methods for PyTorch format compatibility

    private static string SanitizeFileName(string name)
    {
        // Replace characters that are problematic in filenames
        return name.Replace("/", "_").Replace("\\", "_").Replace(".", "_");
    }

    private static void SaveStateDictPt(string path, Dictionary<string, Tensor> stateDict)
    {
        // TorchSharp doesn't support PyTorch pickle format
        // Instead, save as a directory with individual tensor files + manifest
        // This maintains compatibility with the API while using TorchSharp's capabilities
        
        // Create a directory to hold the checkpoint
        var checkpointDir = path + ".dir";
        Directory.CreateDirectory(checkpointDir);
        
        // Save each tensor individually
        foreach (var (name, tensor) in stateDict)
        {
            var tensorPath = Path.Combine(checkpointDir, SanitizeFileName(name) + ".dat");
            tensor.save(tensorPath);
        }
        
        // Save a manifest mapping sanitized names to original names
        var manifest = stateDict.Keys.ToDictionary(k => SanitizeFileName(k), k => k);
        var manifestPath = Path.Combine(checkpointDir, "manifest.json");
        var json = JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(manifestPath, json);
        
        // Create a marker file with the original .pt name for compatibility
        File.WriteAllText(path, checkpointDir);
    }

    private static Dictionary<string, Tensor> LoadStateDictPt(string path, Device device)
    {
        // Load from the directory-based checkpoint format
        // Check if path points to a directory marker file
        string checkpointDir;
        if (File.Exists(path) && !Directory.Exists(path))
        {
            // It's a marker file, read the actual directory path
            checkpointDir = File.ReadAllText(path).Trim();
        }
        else if (Directory.Exists(path + ".dir"))
        {
            // Direct directory access
            checkpointDir = path + ".dir";
        }
        else
        {
            throw new FileNotFoundException($"Checkpoint not found at {path}");
        }
        
        // Load the manifest
        var manifestPath = Path.Combine(checkpointDir, "manifest.json");
        var json = File.ReadAllText(manifestPath);
        var manifest = JsonSerializer.Deserialize<Dictionary<string, string>>(json)
            ?? throw new InvalidOperationException("Failed to deserialize manifest");
        
        // Load each tensor
        var stateDict = new Dictionary<string, Tensor>();
        foreach (var (sanitizedName, originalName) in manifest)
        {
            var tensorPath = Path.Combine(checkpointDir, sanitizedName + ".dat");
            var tensor = torch.load(tensorPath).to(device);
            stateDict[originalName] = tensor;
        }
        
        return stateDict;
    }

    private static void SaveOptimizerStatePt(string path, Dictionary<string, object> optimizerState)
    {
        // Optimizer state saving is not supported in TorchSharp yet
        throw new NotImplementedException(
            "Optimizer state saving is not yet supported in TorchSharp. " +
            "You will need to recreate the optimizer when resuming training.");
    }

    private static Dictionary<string, object> LoadOptimizerStatePt(string path, Device device)
    {
        // Optimizer state loading is not supported in TorchSharp yet
        throw new NotImplementedException(
            "Optimizer state loading is not yet supported in TorchSharp. " +
            "You will need to recreate the optimizer when resuming training.");
    }

    private static void LoadStateDict(Module module, Dictionary<string, Tensor> stateDict)
    {
        // TorchSharp's load_state_dict equivalent
        // Use no_grad to avoid autograd issues when copying into parameters
        using (var _ = torch.no_grad())
        {
            var moduleParams = module.named_parameters().ToDictionary(p => p.name, p => p.parameter);
            
            foreach (var (name, tensor) in stateDict)
            {
                if (moduleParams.TryGetValue(name, out var param))
                {
                    param.copy_(tensor);
                }
                else
                {
                    throw new InvalidOperationException($"Parameter {name} not found in model");
                }
            }
        }
    }
}
