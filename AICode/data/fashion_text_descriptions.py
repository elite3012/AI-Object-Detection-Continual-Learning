"""
Text descriptions for Fashion-MNIST classes
Multi-modal data for vision + language continual learning
"""
import torch
import numpy as np


# Rich text descriptions for each Fashion-MNIST class
CLASS_DESCRIPTIONS = {
    0: [  # T-shirt/top
        "a casual t-shirt with short sleeves",
        "comfortable cotton top for everyday wear",
        "basic crew neck shirt",
        "simple pullover tee",
        "classic short-sleeved top"
    ],
    1: [  # Trouser
        "long pants for formal or casual wear",
        "denim jeans with pockets",
        "comfortable trousers",
        "casual bottom wear with legs",
        "full-length pants"
    ],
    2: [  # Pullover
        "warm knitted sweater",
        "cozy pullover for cold weather",
        "long-sleeved knit top",
        "casual woolen sweater",
        "comfortable knitwear"
    ],
    3: [  # Dress
        "elegant one-piece garment",
        "stylish women's dress",
        "feminine clothing item",
        "fashionable one-piece outfit",
        "graceful dress for special occasions"
    ],
    4: [  # Coat
        "warm outerwear for winter",
        "long jacket for cold weather",
        "protective outer garment",
        "formal or casual overcoat",
        "heavy jacket with buttons"
    ],
    5: [  # Sandal
        "open-toed summer footwear",
        "casual shoes for warm weather",
        "comfortable sandals with straps",
        "lightweight open shoes",
        "breathable summer footwear"
    ],
    6: [  # Shirt
        "formal or casual button-up top",
        "collared shirt with buttons",
        "dress shirt for professional wear",
        "classic button-down clothing",
        "formal top with collar"
    ],
    7: [  # Sneaker
        "athletic sports shoes",
        "comfortable running footwear",
        "casual athletic shoes with laces",
        "sporty sneakers for exercise",
        "cushioned sports footwear"
    ],
    8: [  # Bag
        "handbag or shoulder bag",
        "accessory for carrying items",
        "fashionable purse or tote",
        "portable storage accessory",
        "stylish carrying bag"
    ],
    9: [  # Ankle boot
        "short boots covering the ankle",
        "fashionable ankle-length footwear",
        "stylish boots for fall or winter",
        "short leather boots",
        "trendy ankle-high shoes"
    ]
}


# Simple vocabulary for tokenization
VOCAB = {
    '<PAD>': 0, '<UNK>': 1, '<START>': 2, '<END>': 3,
    'a': 4, 'the': 5, 'for': 6, 'with': 7, 'or': 8, 'and': 9,
    'casual': 10, 'formal': 11, 'comfortable': 12, 'stylish': 13, 'warm': 14,
    'shirt': 15, 'top': 16, 'wear': 17, 'clothing': 18, 'shoes': 19,
    'footwear': 20, 'boots': 21, 'sneaker': 22, 'sandal': 23, 'bag': 24,
    'dress': 25, 'coat': 26, 'jacket': 27, 'pants': 28, 'trousers': 29,
    'sweater': 30, 'pullover': 31, 'tee': 32, 'short': 33, 'long': 34,
    'sleeve': 35, 'sleeves': 36, 'ankle': 37, 'cotton': 38, 'denim': 39,
    'knitted': 40, 'woolen': 41, 'leather': 42, 'women': 43, 'athletic': 44,
    'sports': 45, 'running': 46, 'summer': 47, 'winter': 48, 'weather': 49,
    'everyday': 50, 'basic': 51, 'classic': 52, 'elegant': 53, 'fashionable': 54,
    'cozy': 55, 'protective': 56, 'open': 57, 'toed': 58, 'button': 59,
    'buttons': 60, 'collar': 61, 'collared': 62, 'laces': 63, 'straps': 64,
    'handbag': 65, 'shoulder': 66, 'purse': 67, 'tote': 68, 'accessory': 69,
    'carrying': 70, 'items': 71, 'garment': 72, 'outfit': 73, 'outerwear': 74,
    'knitwear': 75, 'jeans': 76, 'pockets': 77, 'crew': 78, 'neck': 79,
    'one': 80, 'piece': 81, 'feminine': 82, 'special': 83, 'occasions': 84,
    'outer': 85, 'heavy': 86, 'lightweight': 87, 'breathable': 88, 'down': 89,
    'professional': 90, 'exercise': 91, 'cushioned': 92, 'portable': 93,
    'storage': 94, 'fall': 95, 'trendy': 96, 'high': 97, 'covering': 98,
    'length': 99, 'item': 100, 'bottom': 101, 'full': 102, 'cold': 103,
    'knit': 104, 'sleeved': 105, 'graceful': 106, 'overcoat': 107
}

# Reverse vocab for decoding
IDX_TO_WORD = {v: k for k, v in VOCAB.items()}


def simple_tokenize(text, max_len=32):
    """
    Simple word-level tokenization
    
    Args:
        text: string to tokenize
        max_len: maximum sequence length
        
    Returns:
        token_ids: list of token indices
    """
    words = text.lower().replace(',', '').replace('.', '').split()
    tokens = [VOCAB['<START>']]
    
    for word in words[:max_len-2]:  # Reserve space for START and END
        tokens.append(VOCAB.get(word, VOCAB['<UNK>']))
    
    tokens.append(VOCAB['<END>'])
    
    # Pad to max_len
    while len(tokens) < max_len:
        tokens.append(VOCAB['<PAD>'])
    
    return tokens[:max_len]


def get_class_text_descriptions(class_id, num_descriptions=1):
    """
    Get text descriptions for a class
    
    Args:
        class_id: 0-9 Fashion-MNIST class
        num_descriptions: how many descriptions to return
        
    Returns:
        descriptions: list of text strings
    """
    descriptions = CLASS_DESCRIPTIONS.get(class_id, ["unknown clothing item"])
    
    if num_descriptions == 1:
        return [np.random.choice(descriptions)]
    else:
        return list(np.random.choice(descriptions, size=min(num_descriptions, len(descriptions)), replace=False))


def get_tokenized_descriptions(class_id, num_descriptions=1, max_len=32):
    """
    Get tokenized text descriptions for a class
    
    Args:
        class_id: 0-9 Fashion-MNIST class
        num_descriptions: how many descriptions to return
        max_len: max sequence length
        
    Returns:
        tokens: torch.Tensor of shape [num_descriptions, max_len]
    """
    descriptions = get_class_text_descriptions(class_id, num_descriptions)
    tokens = [simple_tokenize(desc, max_len) for desc in descriptions]
    return torch.tensor(tokens, dtype=torch.long)


def create_multimodal_batch(images, labels, max_len=32):
    """
    Create a multi-modal batch with images and corresponding text descriptions
    
    Args:
        images: [batch, 1, 28, 28] image tensor
        labels: [batch] class labels
        max_len: max text sequence length
        
    Returns:
        images: [batch, 1, 28, 28]
        text_tokens: [batch, max_len]
        labels: [batch]
    """
    batch_size = labels.shape[0]
    text_tokens = []
    
    for label in labels:
        # Get one random description for this class
        tokens = get_tokenized_descriptions(int(label.item()), num_descriptions=1, max_len=max_len)
        text_tokens.append(tokens[0])
    
    text_tokens = torch.stack(text_tokens)  # [batch, max_len]
    
    return images, text_tokens, labels


def get_vocab_size():
    """Return vocabulary size"""
    return len(VOCAB)


def decode_tokens(token_ids):
    """
    Decode token IDs back to text
    
    Args:
        token_ids: list or tensor of token indices
        
    Returns:
        text: decoded string
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.cpu().tolist()
    
    words = []
    for idx in token_ids:
        word = IDX_TO_WORD.get(idx, '<UNK>')
        if word in ['<PAD>', '<START>', '<END>']:
            continue
        words.append(word)
    
    return ' '.join(words)


# Template descriptions for zero-shot classification
TEMPLATE_DESCRIPTIONS = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "{} clothing item",
    "wearing {}"
]

CLASS_NAMES_SIMPLE = [
    "t-shirt", "trousers", "pullover", "dress", "coat",
    "sandals", "shirt", "sneakers", "bag", "ankle boots"
]
