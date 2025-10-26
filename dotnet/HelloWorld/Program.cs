using TorchSharp;
using static TorchSharp.torch;

Console.WriteLine("TorchSharp GPU Test");
Console.WriteLine($"CUDA Available: {cuda.is_available()}");

if (!cuda.is_available())
{
    Console.WriteLine("ERROR: CUDA is not available!");
    return;
}

Console.WriteLine($"CUDA Device Count: {cuda.device_count()}");

// Set default device to CUDA
var device = CUDA;
Console.WriteLine($"\nUsing device: {device}");

// Create tensors on GPU
var x = tensor([1.0f, 2.0f, 3.0f, 4.0f, 5.0f], device: device);
Console.WriteLine($"\nTensor on GPU: {x}");
Console.WriteLine($"Device: {x.device}");

// Perform operations on GPU
var y = x * 2;
Console.WriteLine($"\nTensor * 2: {y}");
Console.WriteLine($"Device: {y.device}");

var sum = x.sum();
Console.WriteLine($"\nSum: {sum}");
Console.WriteLine($"Device: {sum.device}");

// Create a larger matrix for more substantial GPU computation
var matrix1 = randn(1000, 1000, device: device);
var matrix2 = randn(1000, 1000, device: device);

Console.WriteLine($"\nPerforming matrix multiplication on GPU (1000x1000)...");
var startTime = DateTime.Now;
var result = matrix1.matmul(matrix2);
var elapsed = (DateTime.Now - startTime).TotalMilliseconds;

Console.WriteLine($"Matrix multiplication completed in {elapsed:F2} ms");
Console.WriteLine($"Result shape: {result.shape[0]}x{result.shape[1]}");
Console.WriteLine($"Result device: {result.device}");
Console.WriteLine($"First element: {result[0, 0]}");

Console.WriteLine("\nGPU test completed successfully!");
