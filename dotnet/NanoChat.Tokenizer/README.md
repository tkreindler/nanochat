# NanoChat.Tokenizer

This project contains tokenizer implementations for NanoChat.

## Implementation Status

### Current Status
- ✅ `SpecialTokens.cs` - Special token constants
- ✅ `ITokenizer.cs` - Common tokenizer interface
- 🚧 Tokenizer implementations needed

### Required Implementations

1. **RustBPE C# Bindings** (High Priority)
   - The Python code uses a custom Rust tokenizer via PyO3
   - Need to create C# P/Invoke bindings to the Rust library
   - Or create a C# native DLL wrapper
   - Location: `rustbpe/src/lib.rs`

2. **Tiktoken Integration** (Alternative)
   - Port or wrap tiktoken for C# (used for inference)
   - May need to use external library or port the code
   
3. **Pure C# Implementation** (Fallback)
   - Implement BPE training algorithm in C#
   - Port the regex-based text splitting
   - Implement merge table construction

### Dependencies Needed

- **TorchSharp** - Already added (for tensor operations with token bytes)
- **Parquet.NET** or **Apache.Arrow** - For dataset handling (see NanoChat.Dataset)
- **Rust library bindings** - Need to compile rustbpe as a C shared library

### Python Files to Port

- `nanochat/tokenizer.py` (399 lines)
  - `HuggingFaceTokenizer` class
  - `RustBPETokenizer` class
  - `get_tokenizer()` helper
  - `get_token_bytes()` helper
  - Conversation rendering logic for chat

### Key Features to Implement

1. **Training** (`train_from_iterator`)
   - Accept text iterator
   - Build BPE merge table
   - Support GPT-4 style regex splitting
   - Handle special tokens

2. **Encoding**
   - Batch encoding with multi-threading
   - Prepend/append token support
   - Special token handling

3. **Decoding**
   - Convert token IDs back to text
   - Handle special tokens

4. **Conversation Rendering** (for chat fine-tuning)
   - `render_conversation()` - tokenize chat messages with masks
   - `render_for_completion()` - prime assistant for RL
   - `visualize_tokenization()` - debugging helper

5. **Token Bytes Mapping**
   - Cache token ID -> byte count mapping
   - Used for bits-per-byte evaluation
   - Stored as tensor on disk

### Next Steps

1. Decide on tokenizer backend:
   - Option A: Create C# P/Invoke bindings to rustbpe
   - Option B: Use SharpToken (tiktoken port) for inference only
   - Option C: Pure C# BPE implementation

2. Implement training functionality
3. Implement conversation rendering
4. Add token bytes caching
