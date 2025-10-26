using TorchSharp;
using static TorchSharp.torch;

Console.WriteLine("TorchSharp Hello World!");
Console.WriteLine($"CUDA Available: {cuda.is_available()}");

// Create a simple tensor
var x = tensor(new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f });
Console.WriteLine($"\nTensor: {x}");

// Perform some operations
var y = x * 2;
Console.WriteLine($"Tensor * 2: {y}");

var sum = x.sum();
Console.WriteLine($"Sum: {sum}");

// Create a 2D tensor (matrix)
var matrix = randn(3, 3);
Console.WriteLine($"\nRandom 3x3 matrix:\n{matrix}");
