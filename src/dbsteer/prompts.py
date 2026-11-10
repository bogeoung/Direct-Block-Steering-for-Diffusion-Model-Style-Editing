"""Prompt templates and disjoint content splits used by DBsteer."""

EXTRACTION_CONTENTS = [
    "a cat sitting on a windowsill", "a dog running on a beach",
    "a horse in a meadow", "an owl perched on a branch",
    "a red fox in the snow", "a goldfish in a bowl",
    "an elephant at a watering hole", "a parrot on a perch",
    "an old man with a beard", "a young woman reading a book",
    "a child flying a kite", "a chef cooking in a kitchen",
    "a musician playing guitar", "a couple walking in the rain",
    "a mountain lake at sunrise", "a lighthouse on a cliff",
    "a forest path in autumn", "a desert with sand dunes",
    "a waterfall in a jungle", "a field of sunflowers",
    "a city street at night", "a small village by a river",
    "a cathedral interior", "a wooden cabin in the woods",
    "a bridge over a canal", "a bowl of fruit on a table",
    "a vintage car parked on a street", "a sailboat on the sea",
    "a cup of coffee and a croissant", "a vase of flowers",
    "a bicycle leaning against a wall", "a steam train crossing a viaduct",
]

EVALUATION_CONTENTS = [
    "a tabby cat lying on a sofa", "a golden retriever in a park",
    "a deer standing in a forest", "a portrait of a smiling woman",
    "an astronaut standing on the moon", "a fisherman on a boat",
    "a snowy mountain village", "a harbor with fishing boats",
]

NEUTRAL_TEMPLATE = "a photo of {content}, high quality, detailed"

STYLE_TEMPLATES = {
    "watercolor": "a watercolor painting of {content}",
    "sketch": "a pencil sketch of {content}, black and white drawing",
    "anime": "an anime illustration of {content}, cel shading",
    "ukiyoe": "a ukiyo-e japanese woodblock print of {content}",
    "vangogh": "a painting of {content} in the style of Vincent van Gogh, swirling brushstrokes",
    "monet": "an impressionist painting of {content} in the style of Claude Monet",
    "picasso": "a cubist painting of {content} in the style of Pablo Picasso",
}
