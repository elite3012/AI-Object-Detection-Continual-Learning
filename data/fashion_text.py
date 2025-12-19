"""
Fashion-MNIST Text Descriptions
Provides rich text descriptions for each class to enable multi-modal learning
"""

# Class names
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Rich text descriptions for each class (multiple templates for augmentation)
CLASS_DESCRIPTIONS = {
    0: [  # T-shirt/top
        "A casual t-shirt with short sleeves",
        "A comfortable top garment for everyday wear",
        "A basic shirt with round or v-neck collar",
        "A simple upper body clothing item",
        "A casual short-sleeved shirt"
    ],
    1: [  # Trouser
        "Long pants covering both legs separately",
        "A pair of trousers extending to the ankles",
        "Casual or formal leg wear garment",
        "Pants with two separate leg portions",
        "Long bottoms for lower body"
    ],
    2: [  # Pullover
        "A warm knitted sweater for cold weather",
        "A cozy pullover garment worn over clothes",
        "A thick upper body warming clothing",
        "A knit sweater pulled over the head",
        "A warm woolen top garment"
    ],
    3: [  # Dress
        "A one-piece garment for women",
        "A feminine clothing covering torso and legs",
        "An elegant single-piece outfit",
        "A women's garment with skirt portion",
        "A dress combining top and bottom in one piece"
    ],
    4: [  # Coat
        "A long outer garment for protection",
        "A warm jacket extending below the waist",
        "An overcoat worn over other clothes",
        "A thick protective outer layer clothing",
        "A long-sleeved outer garment for cold weather"
    ],
    5: [  # Sandal
        "Open footwear with straps",
        "Casual shoes with exposed toes",
        "Light summer footwear with open design",
        "Comfortable open-toed shoes",
        "Casual foot wear for warm weather"
    ],
    6: [  # Shirt
        "A formal button-up garment",
        "A collared shirt with front buttons",
        "A dress shirt for formal occasions",
        "A button-down upper body garment",
        "A formal top with collar and sleeves"
    ],
    7: [  # Sneaker
        "Comfortable athletic footwear",
        "Casual sports shoes with rubber sole",
        "Running or training shoes",
        "Athletic footwear for daily activities",
        "Comfortable closed-toe sports shoes"
    ],
    8: [  # Bag
        "A carrying accessory with handles",
        "A handbag or shoulder bag",
        "A portable container for carrying items",
        "An accessory for holding belongings",
        "A fashion accessory for carrying things"
    ],
    9: [  # Ankle boot
        "Short boots reaching the ankle",
        "Footwear covering foot and ankle",
        "Low-cut boots ending at ankle height",
        "Stylish boots with ankle-length shaft",
        "Short leather or fabric boots"
    ]
}

# Simple single-sentence descriptions (for efficiency)
SIMPLE_DESCRIPTIONS = {
    0: "a t-shirt or casual top",
    1: "a pair of trousers or pants",
    2: "a pullover sweater",
    3: "a dress for women",
    4: "a long coat or jacket",
    5: "open sandals for feet",
    6: "a formal button-up shirt",
    7: "athletic sneakers or running shoes",
    8: "a handbag or carrying bag",
    9: "ankle-high boots"
}

# Attribute-based descriptions (structured)
CLASS_ATTRIBUTES = {
    0: {
        "type": "upper body clothing",
        "style": "casual",
        "sleeve": "short",
        "formality": "informal"
    },
    1: {
        "type": "lower body clothing",
        "style": "casual or formal",
        "length": "full length",
        "formality": "versatile"
    },
    2: {
        "type": "upper body clothing",
        "style": "casual warm",
        "material": "knitted fabric",
        "season": "cold weather"
    },
    3: {
        "type": "full body clothing",
        "style": "feminine elegant",
        "gender": "women",
        "formality": "formal to casual"
    },
    4: {
        "type": "outer clothing",
        "style": "protective warm",
        "length": "long",
        "season": "cold weather"
    },
    5: {
        "type": "footwear",
        "style": "casual open",
        "coverage": "open toes",
        "season": "warm weather"
    },
    6: {
        "type": "upper body clothing",
        "style": "formal",
        "collar": "yes",
        "formality": "formal"
    },
    7: {
        "type": "footwear",
        "style": "athletic casual",
        "purpose": "sports daily",
        "comfort": "high"
    },
    8: {
        "type": "accessory",
        "style": "carrying item",
        "purpose": "storage",
        "usage": "daily"
    },
    9: {
        "type": "footwear",
        "style": "stylish protective",
        "height": "ankle",
        "season": "all weather"
    }
}

def get_text_description(class_id, template_id=0, mode='rich'):
    """
    Get text description for a class
    
    Args:
        class_id: 0-9 class ID
        template_id: Which template variation to use (0-4)
        mode: 'rich', 'simple', or 'attributes'
    
    Returns:
        Text description string
    """
    if mode == 'simple':
        return SIMPLE_DESCRIPTIONS[class_id]
    elif mode == 'attributes':
        attrs = CLASS_ATTRIBUTES[class_id]
        return " ".join([f"{k}: {v}" for k, v in attrs.items()])
    else:  # rich
        descriptions = CLASS_DESCRIPTIONS[class_id]
        return descriptions[template_id % len(descriptions)]

def get_all_descriptions(mode='simple'):
    """Get descriptions for all classes"""
    return [get_text_description(i, mode=mode) for i in range(10)]

# Task-specific prompts
TASK_PROMPTS = {
    0: "clothing items for upper body: t-shirts and casual tops",
    1: "clothing items: pullovers and dresses",
    2: "outerwear items: coats and sandals",
    3: "clothing items: formal shirts and athletic sneakers",
    4: "accessories and footwear: bags and ankle boots"
}

def get_task_prompt(task_id):
    """Get task-level description"""
    return TASK_PROMPTS.get(task_id, f"Task {task_id}")
