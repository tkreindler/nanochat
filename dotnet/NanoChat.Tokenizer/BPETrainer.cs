using System.Collections.Concurrent;
using System.Text;
using System.Text.RegularExpressions;

namespace NanoChat.Tokenizer;

/// <summary>
/// BPE Trainer for building token vocabularies from text
/// Port of rustbpe training logic
/// </summary>
public class BPETrainer
{
    private readonly Dictionary<(uint, uint), uint> _merges = new();
    private readonly string _pattern;
    private readonly Regex _compiledPattern;
    
    public BPETrainer(string pattern)
    {
        _pattern = pattern;
        _compiledPattern = new Regex(pattern, RegexOptions.Compiled);
    }
    
    /// <summary>
    /// Train BPE from an iterator of text strings
    /// </summary>
    /// <param name="textIterator">Iterator of text strings</param>
    /// <param name="vocabSize">Target vocabulary size</param>
    /// <param name="bufferSize">Number of texts to process in parallel batches</param>
    /// <returns>Mergeable ranks dictionary (token bytes -> rank)</returns>
    public Dictionary<byte[], int> TrainFromIterator(
        IEnumerable<string> textIterator, 
        int vocabSize,
        int bufferSize = 8192)
    {
        if (vocabSize < 256)
            throw new ArgumentException("vocab_size must be at least 256", nameof(vocabSize));
        
        int numMerges = vocabSize - 256;
        Console.WriteLine($"Starting BPE training: {numMerges} merges to compute");
        
        // Step 1: Count all unique chunks across the dataset
        var chunkCounts = CountChunks(textIterator, bufferSize);
        Console.WriteLine($"Processed sequences, {chunkCounts.Count} unique chunks");
        
        // Step 2: Convert to Word objects
        var words = new List<Word>(chunkCounts.Count);
        var counts = new List<int>(chunkCounts.Count);
        
        foreach (var (chunk, count) in chunkCounts)
        {
            words.Add(new Word(Encoding.UTF8.GetBytes(chunk).Select(b => (uint)b).ToList()));
            counts.Add(count);
        }
        
        // Step 3: Train the BPE merges
        TrainCoreIncremental(words, counts, vocabSize);
        
        // Step 4: Build mergeable ranks
        return BuildMergeableRanks();
    }
    
    /// <summary>
    /// Core incremental BPE training algorithm
    /// </summary>
    private void TrainCoreIncremental(List<Word> words, List<int> counts, int vocabSize)
    {
        int numMerges = vocabSize - 256;
        Console.WriteLine($"Computing initial pair counts from {words.Count} unique sequences");
        
        // Initial pair counting
        var (pairCounts, whereToUpdate) = CountPairsParallel(words, counts);
        
        // Build priority queue (max heap by count)
        Console.WriteLine($"Building heap with {pairCounts.Count} unique pairs");
        var heap = new PriorityQueue<MergeJob, MergeJob>(
            Comparer<MergeJob>.Create((a, b) => b.CompareTo(a))); // reverse for max heap
        
        foreach (var (pair, positions) in whereToUpdate)
        {
            int count = pairCounts.GetValueOrDefault(pair, 0);
            if (count > 0)
            {
                heap.Enqueue(new MergeJob(pair, count, positions), 
                    new MergeJob(pair, count, positions));
            }
        }
        
        // Merge loop
        Console.WriteLine("Starting merge loop");
        int mergesDone = 0;
        int lastLogPercent = 0;
        
        while (mergesDone < numMerges && heap.Count > 0)
        {
            var top = heap.Dequeue();
            
            // Lazy refresh: check if count is still current
            int currentCount = pairCounts.GetValueOrDefault(top.Pair, 0);
            if (top.Count != currentCount)
            {
                if (currentCount > 0)
                {
                    heap.Enqueue(new MergeJob(top.Pair, currentCount, top.Positions), 
                        new MergeJob(top.Pair, currentCount, top.Positions));
                }
                continue;
            }
            
            if (top.Count == 0)
                break;
            
            // Record merge
            uint newId = (uint)(256 + mergesDone);
            _merges[top.Pair] = newId;
            
            // Apply merge to all affected words
            var localPosUpdates = new Dictionary<(uint, uint), HashSet<int>>();
            
            foreach (int wordIdx in top.Positions)
            {
                var changes = words[wordIdx].MergePair(top.Pair, newId);
                
                // Update global pair counts
                foreach (var (pair, delta) in changes)
                {
                    int deltaTotal = delta * counts[wordIdx];
                    if (deltaTotal != 0)
                    {
                        pairCounts[pair] = pairCounts.GetValueOrDefault(pair, 0) + deltaTotal;
                        
                        if (delta > 0)
                        {
                            if (!localPosUpdates.ContainsKey(pair))
                                localPosUpdates[pair] = new HashSet<int>();
                            localPosUpdates[pair].Add(wordIdx);
                        }
                    }
                }
            }
            
            // Add updated pairs back to heap
            foreach (var (pair, positions) in localPosUpdates)
            {
                int cnt = pairCounts.GetValueOrDefault(pair, 0);
                if (cnt > 0)
                {
                    heap.Enqueue(new MergeJob(pair, cnt, positions), 
                        new MergeJob(pair, cnt, positions));
                }
            }
            
            mergesDone++;
            
            // Log progress every 1%
            int currentPercent = (mergesDone * 100) / numMerges;
            if (currentPercent > lastLogPercent)
            {
                Console.WriteLine($"Progress: {currentPercent}% ({mergesDone}/{numMerges} merges) - " +
                    $"Last merge: ({top.Pair.Item1},{top.Pair.Item2}) -> {newId} (frequency: {top.Count})");
                lastLogPercent = currentPercent;
            }
        }
        
        Console.WriteLine($"Finished training: {mergesDone} merges completed");
    }
    
    /// <summary>
    /// Count chunks (regex-split text pieces) across the dataset
    /// </summary>
    private Dictionary<string, int> CountChunks(IEnumerable<string> textIterator, int bufferSize)
    {
        var globalCounts = new ConcurrentDictionary<string, int>();
        var buffer = new List<string>(bufferSize);
        long totalSequences = 0;
        
        foreach (var text in textIterator)
        {
            buffer.Add(text);
            
            if (buffer.Count >= bufferSize)
            {
                ProcessBuffer(buffer, globalCounts);
                totalSequences += buffer.Count;
                buffer.Clear();
            }
        }
        
        // Process remaining
        if (buffer.Count > 0)
        {
            ProcessBuffer(buffer, globalCounts);
            totalSequences += buffer.Count;
        }
        
        Console.WriteLine($"Processed {totalSequences} sequences total");
        return globalCounts.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
    }
    
    private void ProcessBuffer(List<string> buffer, ConcurrentDictionary<string, int> globalCounts)
    {
        // Parallel processing of the buffer
        var localCounts = new ConcurrentDictionary<string, int>();
        
        Parallel.ForEach(buffer, text =>
        {
            foreach (Match match in _compiledPattern.Matches(text))
            {
                localCounts.AddOrUpdate(match.Value, 1, (_, v) => v + 1);
            }
        });
        
        // Merge into global counts
        foreach (var (chunk, count) in localCounts)
        {
            globalCounts.AddOrUpdate(chunk, count, (_, v) => v + count);
        }
    }
    
    /// <summary>
    /// Count pairs across all words (parallel)
    /// </summary>
    private (Dictionary<(uint, uint), int>, Dictionary<(uint, uint), HashSet<int>>) 
        CountPairsParallel(List<Word> words, List<int> counts)
    {
        var pairCounts = new ConcurrentDictionary<(uint, uint), int>();
        var whereToUpdate = new ConcurrentDictionary<(uint, uint), HashSet<int>>();
        
        Parallel.For(0, words.Count, i =>
        {
            if (words[i].Ids.Count >= 2 && counts[i] != 0)
            {
                var localPairs = new HashSet<(uint, uint)>();
                
                for (int j = 0; j < words[i].Ids.Count - 1; j++)
                {
                    var pair = (words[i].Ids[j], words[i].Ids[j + 1]);
                    localPairs.Add(pair);
                }
                
                foreach (var pair in localPairs)
                {
                    pairCounts.AddOrUpdate(pair, counts[i], (_, v) => v + counts[i]);
                    
                    lock (whereToUpdate)
                    {
                        if (!whereToUpdate.ContainsKey(pair))
                            whereToUpdate[pair] = new HashSet<int>();
                        whereToUpdate[pair].Add(i);
                    }
                }
            }
        });
        
        return (pairCounts.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                whereToUpdate.ToDictionary(kvp => kvp.Key, kvp => kvp.Value));
    }
    
    /// <summary>
    /// Build the mergeable ranks dictionary from the trained merges
    /// </summary>
    private Dictionary<byte[], int> BuildMergeableRanks()
    {
        var mergeableRanks = new Dictionary<byte[], int>(new ByteArrayComparer());
        
        // Start with base 256 byte tokens
        var tokenBytes = new List<byte[]>(256);
        for (int i = 0; i < 256; i++)
        {
            tokenBytes.Add(new byte[] { (byte)i });
            mergeableRanks[tokenBytes[i]] = i;
        }
        
        // Sort merges by token ID to reconstruct bytes progressively
        var sortedMerges = _merges.OrderBy(kvp => kvp.Value).ToList();
        
        foreach (var (pair, mergedId) in sortedMerges)
        {
            var (left, right) = pair;
            var merged = tokenBytes[(int)left].Concat(tokenBytes[(int)right]).ToArray();
            
            // Expand tokenBytes if needed
            while (tokenBytes.Count <= mergedId)
                tokenBytes.Add(Array.Empty<byte>());
            
            tokenBytes[(int)mergedId] = merged;
            mergeableRanks[merged] = (int)mergedId;
        }
        
        return mergeableRanks;
    }
    
    public string GetPattern() => _pattern;
    public IReadOnlyDictionary<(uint, uint), uint> GetMerges() => _merges;
    
    /// <summary>
    /// Internal word representation for BPE training
    /// </summary>
    private class Word
    {
        public List<uint> Ids { get; private set; }
        
        public Word(List<uint> ids)
        {
            Ids = ids;
        }
        
        /// <summary>
        /// Merge all non-overlapping occurrences of pair -> newId
        /// Returns pair count deltas for this word
        /// </summary>
        public List<((uint, uint), int)> MergePair((uint, uint) pair, uint newId)
        {
            var (a, b) = pair;
            int n = Ids.Count;
            
            if (n < 2)
                return new List<((uint, uint), int)>();
            
            var output = new List<uint>(n);
            var deltas = new List<((uint, uint), int)>();
            
            int i = 0;
            while (i < n)
            {
                if (i + 1 < n && Ids[i] == a && Ids[i + 1] == b)
                {
                    uint? left = output.Count > 0 ? output[^1] : null;
                    uint? right = i + 2 < n ? Ids[i + 2] : null;
                    
                    // Remove old pairs
                    if (left.HasValue)
                    {
                        deltas.Add(((left.Value, a), -1));
                        deltas.Add(((left.Value, newId), 1));
                    }
                    deltas.Add(((a, b), -1));
                    if (right.HasValue)
                    {
                        deltas.Add(((b, right.Value), -1));
                        deltas.Add(((newId, right.Value), 1));
                    }
                    
                    // Write merged token
                    output.Add(newId);
                    i += 2;
                }
                else
                {
                    output.Add(Ids[i]);
                    i++;
                }
            }
            
            Ids = output;
            return deltas;
        }
    }
    
    /// <summary>
    /// Merge job for priority queue
    /// </summary>
    private class MergeJob : IComparable<MergeJob>
    {
        public (uint, uint) Pair { get; }
        public int Count { get; }
        public HashSet<int> Positions { get; }
        
        public MergeJob((uint, uint) pair, int count, HashSet<int> positions)
        {
            Pair = pair;
            Count = count;
            Positions = positions;
        }
        
        public int CompareTo(MergeJob? other)
        {
            if (other == null) return 1;
            
            // Max-heap by count; tie-break to ascending pair order (deterministic)
            if (Count != other.Count)
                return Count.CompareTo(other.Count);
            
            // Ascending order on pair when counts tie
            int cmp1 = Pair.Item1.CompareTo(other.Pair.Item1);
            if (cmp1 != 0) return -cmp1; // reverse for ascending
            return -Pair.Item2.CompareTo(other.Pair.Item2);
        }
    }
    
    /// <summary>
    /// Byte array comparer for dictionary keys
    /// </summary>
    private class ByteArrayComparer : IEqualityComparer<byte[]>
    {
        public bool Equals(byte[]? x, byte[]? y)
        {
            if (x == null || y == null) return x == y;
            return x.SequenceEqual(y);
        }
        
        public int GetHashCode(byte[] obj)
        {
            if (obj == null) return 0;
            
            int hash = 17;
            foreach (byte b in obj)
                hash = hash * 31 + b;
            return hash;
        }
    }
}
