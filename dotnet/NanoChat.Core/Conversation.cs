namespace NanoChat.Core;

/// <summary>
/// Represents a single message in a conversation
/// </summary>
public class Message
{
    /// <summary>
    /// Role of the message sender (system, user, or assistant)
    /// </summary>
    public required string Role { get; init; }
    
    /// <summary>
    /// Content of the message. Can be either a simple string or structured parts (for tool calls)
    /// </summary>
    public required object Content { get; init; }
    
    /// <summary>
    /// Get content as a simple string (throws if content is structured)
    /// </summary>
    public string GetStringContent()
    {
        if (Content is string str)
            return str;
        throw new InvalidOperationException("Content is not a simple string");
    }
    
    /// <summary>
    /// Get content as structured parts (throws if content is a simple string)
    /// </summary>
    public List<MessagePart> GetStructuredContent()
    {
        if (Content is List<MessagePart> parts)
            return parts;
        throw new InvalidOperationException("Content is not structured (list of parts)");
    }
    
    /// <summary>
    /// Check if content is a simple string
    /// </summary>
    public bool IsSimpleString => Content is string;
}

/// <summary>
/// Represents a part of a multi-part message (e.g., text, tool call, output)
/// </summary>
public class MessagePart
{
    /// <summary>
    /// Type of the part: "text", "python", or "python_output"
    /// </summary>
    public required string Type { get; init; }
    
    /// <summary>
    /// Text content of the part
    /// </summary>
    public required string Text { get; init; }
}

/// <summary>
/// Represents a conversation with multiple messages
/// </summary>
public class Conversation
{
    /// <summary>
    /// List of messages in the conversation
    /// </summary>
    public required List<Message> Messages { get; init; }
    
    /// <summary>
    /// Check if the first message is a system message
    /// </summary>
    public bool HasSystemMessage => Messages.Count > 0 && Messages[0].Role == "system";
    
    /// <summary>
    /// Get all non-system messages
    /// </summary>
    public List<Message> GetNonSystemMessages()
    {
        return HasSystemMessage ? Messages.Skip(1).ToList() : Messages;
    }
    
    /// <summary>
    /// Validate that the conversation follows the expected format:
    /// - Optional system message at the beginning
    /// - Alternating user/assistant messages
    /// </summary>
    public void Validate()
    {
        if (Messages.Count == 0)
            throw new InvalidOperationException("Conversation must have at least one message");
        
        var nonSystemMessages = GetNonSystemMessages();
        
        if (nonSystemMessages.Count < 2)
            throw new InvalidOperationException("Conversation must have at least 2 non-system messages");
        
        for (int i = 0; i < nonSystemMessages.Count; i++)
        {
            var message = nonSystemMessages[i];
            string expectedRole = i % 2 == 0 ? "user" : "assistant";
            
            if (message.Role != expectedRole)
                throw new InvalidOperationException(
                    $"Message {i} has role '{message.Role}' but should be '{expectedRole}'");
            
            // User messages must be simple strings
            if (message.Role == "user" && !message.IsSimpleString)
                throw new InvalidOperationException("User messages must be simple strings");
        }
    }
}
