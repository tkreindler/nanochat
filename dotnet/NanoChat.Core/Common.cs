using System.Text.RegularExpressions;
using TorchSharp;
using static TorchSharp.torch;

namespace NanoChat.Core;

/// <summary>
/// Common utilities for NanoChat.
/// </summary>
public static class Common
{
    private static readonly object LogLock = new();
    
    /// <summary>
    /// Gets the base directory for nanochat intermediates and cached data.
    /// By default uses ~/.cache/nanochat (Linux/Mac) or %LOCALAPPDATA%/nanochat/cache (Windows).
    /// Can be overridden with NANOCHAT_BASE_DIR environment variable.
    /// </summary>
    public static string GetBaseDir()
    {
        var nanoChatBaseDir = Environment.GetEnvironmentVariable("NANOCHAT_BASE_DIR");
        
        if (!string.IsNullOrEmpty(nanoChatBaseDir))
        {
            Directory.CreateDirectory(nanoChatBaseDir);
            return nanoChatBaseDir;
        }

        var homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        
        // Platform-specific cache directory
        string cacheDir;
        if (OperatingSystem.IsWindows())
        {
            cacheDir = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
        }
        else
        {
            cacheDir = Path.Combine(homeDir, ".cache");
        }
        
        var nanoChatDir = Path.Combine(cacheDir, "nanochat");
        Directory.CreateDirectory(nanoChatDir);
        return nanoChatDir;
    }

    /// <summary>
    /// Downloads a file from a URL to a local path in the base directory.
    /// Uses a lock file to prevent concurrent downloads among multiple ranks.
    /// </summary>
    public static async Task<string> DownloadFileWithLockAsync(string url, string filename)
    {
        var baseDir = GetBaseDir();
        var filePath = Path.Combine(baseDir, filename);
        var lockPath = filePath + ".lock";

        if (File.Exists(filePath))
        {
            return filePath;
        }

        // Use a FileStream with exclusive access as a lock mechanism
        using (var lockFile = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
        {
            // Double-check after acquiring lock
            if (File.Exists(filePath))
            {
                return filePath;
            }

            Console.WriteLine($"Downloading {url}...");
            
            using var client = new HttpClient();
            var content = await client.GetStringAsync(url);
            await File.WriteAllTextAsync(filePath, content);
            
            Console.WriteLine($"Downloaded to {filePath}");
        }

        // Clean up the lock file
        try
        {
            File.Delete(lockPath);
        }
        catch (IOException)
        {
            // Ignore if already removed by another process
        }

        return filePath;
    }

    /// <summary>
    /// Print only from rank 0 in distributed training.
    /// </summary>
    public static void Print0(string message = "")
    {
        var ddpRank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0");
        if (ddpRank == 0)
        {
            Console.WriteLine(message);
        }
    }

    /// <summary>
    /// Prints the nanochat ASCII banner (DOS Rebel font).
    /// </summary>
    public static void PrintBanner()
    {
        var banner = @"
                                                   █████                 █████
                                                  ░░███                 ░░███
 ████████    ██████   ████████    ██████   ██████  ░███████    ██████   ███████
░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███ ░░░███░
 ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████   ░███
 ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███   ░███ ███
 ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░████████  ░░█████
░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░    ░░░░░
";
        Print0(banner);
    }

    /// <summary>
    /// Check if we're running in Distributed Data Parallel (DDP) mode.
    /// </summary>
    public static bool IsDdp()
    {
        var rank = Environment.GetEnvironmentVariable("RANK");
        return !string.IsNullOrEmpty(rank) && int.Parse(rank) != -1;
    }

    /// <summary>
    /// Get distributed training information.
    /// Returns (isDdp, rank, localRank, worldSize).
    /// </summary>
    public static (bool isDdp, int rank, int localRank, int worldSize) GetDistInfo()
    {
        if (IsDdp())
        {
            var rank = int.Parse(Environment.GetEnvironmentVariable("RANK") ?? "0");
            var localRank = int.Parse(Environment.GetEnvironmentVariable("LOCAL_RANK") ?? "0");
            var worldSize = int.Parse(Environment.GetEnvironmentVariable("WORLD_SIZE") ?? "1");
            return (true, rank, localRank, worldSize);
        }
        else
        {
            return (false, 0, 0, 1);
        }
    }

    /// <summary>
    /// Auto-detect the best available device type.
    /// Prefers CUDA if available, otherwise falls back to CPU.
    /// </summary>
    public static DeviceType AutodetectDeviceType()
    {
        DeviceType deviceType;
        
        if (cuda.is_available())
        {
            deviceType = DeviceType.CUDA;
        }
        else
        {
            deviceType = DeviceType.CPU;
        }
        
        Print0($"Autodetected device type: {deviceType}");
        return deviceType;
    }

    /// <summary>
    /// Basic initialization for compute resources.
    /// Sets up random seeds, precision, and distributed training if applicable.
    /// Returns (isDdp, rank, localRank, worldSize, device).
    /// </summary>
    public static (bool isDdp, int rank, int localRank, int worldSize, Device device) ComputeInit(DeviceType deviceType = DeviceType.CUDA)
    {
        if (deviceType == DeviceType.CUDA)
        {
            if (!cuda.is_available())
            {
                throw new InvalidOperationException("TorchSharp is not configured for CUDA but device_type is 'cuda'");
            }
        }

        // Reproducibility
        torch.manual_seed(42);
        if (deviceType == DeviceType.CUDA)
        {
            cuda.manual_seed(42);
        }

        // Precision: use TF32 for CUDA matmuls
        if (deviceType == DeviceType.CUDA)
        {
            // TorchSharp equivalent of torch.set_float32_matmul_precision("high")
            // This uses TF32 instead of FP32 for matrix multiplications
            torch.backends.cuda.matmul.allow_tf32 = true;
            torch.backends.cudnn.allow_tf32 = true;
        }

        // Distributed setup: Distributed Data Parallel (DDP)
        var (isDdp, rank, localRank, worldSize) = GetDistInfo();
        Device device;
        
        if (isDdp && deviceType == DeviceType.CUDA)
        {
            device = torch.device(DeviceType.CUDA, localRank);
            // TODO: Set default CUDA device in TorchSharp
            // torch.cuda.set_device() equivalent needs investigation
            
            // Initialize process group for distributed training
            // Note: In TorchSharp, this might need additional setup
            // distributed.init_process_group(...) equivalent
            Console.WriteLine($"Warning: DDP initialization not fully implemented in TorchSharp port");
        }
        else
        {
            device = torch.device(deviceType);
        }

        if (rank == 0)
        {
            Console.WriteLine($"Distributed world size: {worldSize}");
        }

        return (isDdp, rank, localRank, worldSize, device);
    }

    /// <summary>
    /// Cleanup function to call before script exit.
    /// </summary>
    public static void ComputeCleanup()
    {
        if (IsDdp())
        {
            // distributed.destroy_process_group() equivalent
            Console.WriteLine("Warning: DDP cleanup not fully implemented in TorchSharp port");
        }
    }

    /// <summary>
    /// Simple colored console logger.
    /// </summary>
    public static class Logger
    {
        public static void Debug(string message)
        {
            Log("DEBUG", message, ConsoleColor.Cyan);
        }

        public static void Info(string message)
        {
            Log("INFO", message, ConsoleColor.Green);
        }

        public static void Warning(string message)
        {
            Log("WARNING", message, ConsoleColor.Yellow);
        }

        public static void Error(string message)
        {
            Log("ERROR", message, ConsoleColor.Red);
        }

        public static void Critical(string message)
        {
            Log("CRITICAL", message, ConsoleColor.Magenta);
        }

        private static void Log(string level, string message, ConsoleColor color)
        {
            lock (LogLock)
            {
                var timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                Console.Write($"{timestamp} - NanoChat.Core - ");
                
                var originalColor = Console.ForegroundColor;
                Console.ForegroundColor = color;
                Console.Write($"{level}");
                Console.ForegroundColor = originalColor;
                
                Console.WriteLine($" - {message}");
            }
        }
    }

    /// <summary>
    /// Dummy W&B replacement for when we don't want to use W&B.
    /// </summary>
    public class DummyWandb
    {
        public void Log(object data) { }
        public void Finish() { }
    }
}
