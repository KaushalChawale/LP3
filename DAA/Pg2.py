# Huffman encoding using greedy strategy

import heapq
from collections import defaultdict, namedtuple

# Define the structure of the Huffman Node
HuffmanNode = namedtuple('HuffmanNode', ['freq', 'char', 'left', 'right'])

# Function to build the Huffman Tree
def build_huffman_tree(frequency):
    # Create heap with (frequency, unique_id, node) tuples for comparison compatibility
    heap = [(freq, id(char), HuffmanNode(freq, char, None, None)) for char, freq in frequency.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # Pop two smallest elements
        left_freq, _, left = heapq.heappop(heap)
        right_freq, _, right = heapq.heappop(heap)
        
        # Merge nodes
        merged = HuffmanNode(left_freq + right_freq, None, left, right)
        
        # Push merged node with a new unique id
        heapq.heappush(heap, (merged.freq, id(merged), merged))
    
    return heap[0][2]  # Return the root node of the Huffman Tree

# Function to build the Huffman codes from the tree
def build_codes(node, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    
    if node.char is not None:  # Leaf node
        codebook[node.char] = prefix
    else:
        build_codes(node.left, prefix + "0", codebook)
        build_codes(node.right, prefix + "1", codebook)
    
    return codebook

# Function to encode the input data using Huffman coding
def huffman_encoding(data):
    # Build frequency map
    frequency = defaultdict(int, {char: data.count(char) for char in set(data)})
    
    # Build Huffman Tree
    root = build_huffman_tree(frequency)
    
    # Generate Huffman codes
    huffman_codes = build_codes(root)
    
    # Encode the data
    encoded_data = "".join(huffman_codes[char] for char in data)
    
    return encoded_data, huffman_codes

# Function to decode the encoded data back to the original string
def huffman_decoding(encoded_data, huffman_codes):
    reverse_codes = {v: k for k, v in huffman_codes.items()}
    current_code = ""
    decoded_data = []
    
    for bit in encoded_data:
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
    
    return "".join(decoded_data)

# Get user input
data = input("Enter a string to encode: ")
encoded_data, huffman_codes = huffman_encoding(data)

# Display results
print("Original Data:", data)
print("Encoded Data:", encoded_data)
print("Huffman Codes:", huffman_codes)
print("Decoded Data:", huffman_decoding(encoded_data, huffman_codes))
