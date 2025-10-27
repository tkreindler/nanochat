using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// Create a simple linear layer
var linear = Linear(10, 5, hasBias: false);

// Check what methods are available
Console.WriteLine("=== Module methods ===");
var moduleMethods = typeof(nn.Module<,>).GetMethods()
    .Where(m => m.Name.Contains("save", StringComparison.OrdinalIgnoreCase) 
             || m.Name.Contains("load", StringComparison.OrdinalIgnoreCase)
             || m.Name.Contains("state", StringComparison.OrdinalIgnoreCase))
    .Select(m => m.Name)
    .Distinct();

foreach (var method in moduleMethods)
{
    Console.WriteLine(method);
}

Console.WriteLine("\n=== Optimizer methods ===");
var opt = optim.AdamW(linear.parameters());
var optimMethods = opt.GetType().GetMethods()
    .Where(m => m.Name.Contains("save", StringComparison.OrdinalIgnoreCase) 
             || m.Name.Contains("load", StringComparison.OrdinalIgnoreCase)
             || m.Name.Contains("state", StringComparison.OrdinalIgnoreCase))
    .Select(m => m.Name)
    .Distinct();

foreach (var method in optimMethods)
{
    Console.WriteLine(method);
}
