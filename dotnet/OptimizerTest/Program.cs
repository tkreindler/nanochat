using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

// Check what optimizers are available in TorchSharp
Console.WriteLine("Available optimizer types in TorchSharp:");
Console.WriteLine("- torch.optim.Adam");
Console.WriteLine("- torch.optim.AdamW"); 
Console.WriteLine("- torch.optim.SGD");
Console.WriteLine("- torch.optim.RMSprop");
Console.WriteLine("- torch.optim.Adagrad");
Console.WriteLine("- torch.optim.LBFGS");

// Test if AdamW exists
try
{
    var model = Linear(10, 10);
    var optimizer = optim.AdamW(model.parameters(), lr: 0.001);
    Console.WriteLine("\n✓ AdamW is available in TorchSharp!");
    Console.WriteLine("  Optimizer created successfully");
    
    // Test optimizer step
    var input = torch.randn(2, 10);
    var output = model.forward(input);
    var loss = output.sum();
    loss.backward();
    optimizer.step();
    Console.WriteLine("  ✓ Optimizer step() works");
    optimizer.zero_grad();
    Console.WriteLine("  ✓ Optimizer zero_grad() works");
}
catch (Exception ex)
{
    Console.WriteLine($"\n✗ AdamW test failed: {ex.Message}");
    Console.WriteLine($"  Stack: {ex.StackTrace}");
}

// Test per-parameter options (needed for lr_mul and wd_mul in Python version)
Console.WriteLine("\n=== Testing Per-Parameter Options ===");
try
{
    var model = Linear(10, 10);
    
    // Check if we can inspect optimizer properties
    var optimizer = optim.AdamW(model.parameters(), lr: 0.01, weight_decay: 0.1);
    Console.WriteLine("✓ Created AdamW with lr=0.01, weight_decay=0.1");
    
    // The Python code uses per-parameter lr_mul and wd_mul
    // We need to see if TorchSharp supports parameter groups
    Console.WriteLine("\nNote: Need to check TorchSharp source or docs for parameter groups API");
    Console.WriteLine("Python nanochat uses lr_mul and wd_mul per parameter");
}
catch (Exception ex)
{
    Console.WriteLine($"✗ Per-parameter options test failed: {ex.Message}");
}
