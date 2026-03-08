"""
Color Utilities Module
Provides comprehensive color name extraction from hex codes and text.
Uses webcolors library with extended fashion color vocabulary.
"""
import re
from typing import Optional, Tuple, List
import webcolors

from src.core import get_logger

logger = get_logger(__name__)


# Extended fashion color vocabulary with hex approximations
FASHION_COLORS = {
    # Neutrals
    "white": "#FFFFFF",
    "off-white": "#FAF9F6",
    "cream": "#FFFDD0",
    "ivory": "#FFFFF0",
    "eggshell": "#F0EAD6",
    "bone": "#E3DAC9",
    "ecru": "#C2B280",
    "champagne": "#F7E7CE",
    
    "black": "#000000",
    "jet black": "#0A0A0A",
    "charcoal": "#36454F",
    "anthracite": "#293133",
    
    "gray": "#808080",
    "grey": "#808080",
    "silver": "#C0C0C0",
    "slate": "#708090",
    "stone": "#928E85",
    "ash": "#B2BEB5",
    "pewter": "#8E9C9C",
    "gunmetal": "#2C3539",
    "heather gray": "#9AA297",
    "light gray": "#D3D3D3",
    "dark gray": "#A9A9A9",
    
    # Browns & Tans
    "brown": "#964B00",
    "chocolate": "#7B3F00",
    "coffee": "#6F4E37",
    "mocha": "#967969",
    "espresso": "#3C2218",
    "chestnut": "#954535",
    "mahogany": "#C04000",
    "walnut": "#773F1A",
    "cinnamon": "#D2691E",
    "cocoa": "#D2691E",
    
    "tan": "#D2B48C",
    "camel": "#C19A6B",
    "beige": "#F5F5DC",
    "khaki": "#C3B091",
    "sand": "#C2B280",
    "taupe": "#483C32",
    "fawn": "#E5AA70",
    "buff": "#F0DC82",
    "caramel": "#FFD59A",
    "cognac": "#9F381D",
    "tobacco": "#71573C",
    "rust": "#B7410E",
    "terracotta": "#E2725B",
    "sienna": "#A0522D",
    "umber": "#635147",
    
    # Blues
    "blue": "#0000FF",
    "navy": "#000080",
    "navy blue": "#000080",
    "royal blue": "#4169E1",
    "cobalt": "#0047AB",
    "sapphire": "#0F52BA",
    "midnight blue": "#191970",
    "indigo": "#4B0082",
    "denim": "#1560BD",
    "sky blue": "#87CEEB",
    "baby blue": "#89CFF0",
    "powder blue": "#B0E0E6",
    "steel blue": "#4682B4",
    "slate blue": "#6A5ACD",
    "teal": "#008080",
    "turquoise": "#40E0D0",
    "aqua": "#00FFFF",
    "cyan": "#00FFFF",
    "cerulean": "#007BA7",
    "azure": "#007FFF",
    "periwinkle": "#CCCCFF",
    "cornflower": "#6495ED",
    "ocean": "#006994",
    "marine": "#042E60",
    "ice blue": "#99FFFF",
    "french blue": "#0072BB",
    "electric blue": "#7DF9FF",
    "peacock": "#005F69",
    
    # Greens
    "green": "#008000",
    "forest green": "#228B22",
    "emerald": "#50C878",
    "jade": "#00A86B",
    "sage": "#9DC183",
    "olive": "#808000",
    "army green": "#4B5320",
    "hunter green": "#355E3B",
    "moss": "#8A9A5B",
    "mint": "#98FF98",
    "seafoam": "#93E9BE",
    "pistachio": "#93C572",
    "lime": "#32CD32",
    "chartreuse": "#7FFF00",
    "kelly green": "#4CBB17",
    "bottle green": "#006A4E",
    "pine": "#01796F",
    "avocado": "#568203",
    "fern": "#4F7942",
    "khaki green": "#728639",
    
    # Reds & Pinks
    "red": "#FF0000",
    "crimson": "#DC143C",
    "scarlet": "#FF2400",
    "cherry": "#DE3163",
    "ruby": "#E0115F",
    "garnet": "#733635",
    "burgundy": "#800020",
    "maroon": "#800000",
    "wine": "#722F37",
    "oxblood": "#4A0000",
    "bordeaux": "#5C0120",
    "brick red": "#CB4154",
    "vermillion": "#E34234",
    "cardinal": "#C41E3A",
    "raspberry": "#E30B5C",
    "cranberry": "#950714",
    "tomato red": "#FF6347",
    
    "pink": "#FFC0CB",
    "hot pink": "#FF69B4",
    "fuchsia": "#FF00FF",
    "magenta": "#FF00FF",
    "rose": "#FF007F",
    "blush": "#DE5D83",
    "salmon": "#FA8072",
    "coral": "#FF7F50",
    "peach": "#FFCBA4",
    "dusty rose": "#DCAE96",
    "mauve": "#E0B0FF",
    "nude": "#E3BC9A",
    "flesh": "#E8BEAC",
    "millennial pink": "#F3CFC6",
    "ballet pink": "#F7C5CC",
    "bubblegum": "#FFC1CC",
    
    # Purples
    "purple": "#800080",
    "violet": "#EE82EE",
    "lavender": "#E6E6FA",
    "lilac": "#C8A2C8",
    "plum": "#DDA0DD",
    "grape": "#6F2DA8",
    "eggplant": "#614051",
    "aubergine": "#614051",
    "amethyst": "#9966CC",
    "orchid": "#DA70D6",
    "wisteria": "#C9A0DC",
    "mulberry": "#C54B8C",
    "heather": "#B7A5D3",
    "iris": "#5A4FCF",
    "byzantium": "#702963",
    
    # Yellows & Oranges
    "yellow": "#FFFF00",
    "gold": "#FFD700",
    "golden": "#FFD700",
    "mustard": "#FFDB58",
    "lemon": "#FFF44F",
    "canary": "#FFEF00",
    "butter": "#FFFAA0",
    "honey": "#EB9605",
    "amber": "#FFBF00",
    "marigold": "#EAA221",
    "saffron": "#F4C430",
    "ochre": "#CC7722",
    
    "orange": "#FFA500",
    "tangerine": "#FF9966",
    "apricot": "#FBCEB1",
    "peach orange": "#FFCC99",
    "burnt orange": "#CC5500",
    "copper": "#B87333",
    "bronze": "#CD7F32",
    "pumpkin": "#FF7518",
    "paprika": "#8B2500",
    "mango": "#FF8243",
    "sunset": "#FAD6A5",
    "persimmon": "#EC5800",
}

# Add CSS3 standard color names (these will be returned by Gemini)
CSS3_COLORS = {
    # CSS3 standard names that match our prompt instructions
    "aliceblue": "#F0F8FF",
    "antiquewhite": "#FAEBD7",
    "aquamarine": "#7FFFD4",
    "azure": "#F0FFFF",
    "bisque": "#FFE4C4",
    "blanchedalmond": "#FFEBCD",
    "blueviolet": "#8A2BE2",
    "burlywood": "#DEB887",
    "cadetblue": "#5F9EA0",
    "cornflowerblue": "#6495ED",
    "cornsilk": "#FFF8DC",
    "darkblue": "#00008B",
    "darkcyan": "#008B8B",
    "darkgoldenrod": "#B8860B",
    "darkgray": "#A9A9A9",
    "darkgreen": "#006400",
    "darkkhaki": "#BDB76B",
    "darkmagenta": "#8B008B",
    "darkolivegreen": "#556B2F",
    "darkorange": "#FF8C00",
    "darkorchid": "#9932CC",
    "darkred": "#8B0000",
    "darksalmon": "#E9967A",
    "darkseagreen": "#8FBC8F",
    "darkslateblue": "#483D8B",
    "darkslategray": "#2F4F4F",
    "darkturquoise": "#00CED1",
    "darkviolet": "#9400D3",
    "deeppink": "#FF1493",
    "deepskyblue": "#00BFFF",
    "dimgray": "#696969",
    "dodgerblue": "#1E90FF",
    "firebrick": "#B22222",
    "floralwhite": "#FFFAF0",
    "forestgreen": "#228B22",
    "gainsboro": "#DCDCDC",
    "ghostwhite": "#F8F8FF",
    "greenyellow": "#ADFF2F",
    "honeydew": "#F0FFF0",
    "hotpink": "#FF69B4",
    "indianred": "#CD5C5C",
    "khaki": "#F0E68C",
    "lavenderblush": "#FFF0F5",
    "lawngreen": "#7CFC00",
    "lemonchiffon": "#FFFACD",
    "lightblue": "#ADD8E6",
    "lightcoral": "#F08080",
    "lightcyan": "#E0FFFF",
    "lightgoldenrodyellow": "#FAFAD2",
    "lightgray": "#D3D3D3",
    "lightgreen": "#90EE90",
    "lightpink": "#FFB6C1",
    "lightsalmon": "#FFA07A",
    "lightseagreen": "#20B2AA",
    "lightskyblue": "#87CEFA",
    "lightslategray": "#778899",
    "lightsteelblue": "#B0C4DE",
    "lightyellow": "#FFFFE0",
    "limegreen": "#32CD32",
    "linen": "#FAF0E6",
    "mediumaquamarine": "#66CDAA",
    "mediumblue": "#0000CD",
    "mediumorchid": "#BA55D3",
    "mediumpurple": "#9370DB",
    "mediumseagreen": "#3CB371",
    "mediumslateblue": "#7B68EE",
    "mediumspringgreen": "#00FA9A",
    "mediumturquoise": "#48D1CC",
    "mediumvioletred": "#C71585",
    "midnightblue": "#191970",
    "mintcream": "#F5FFFA",
    "mistyrose": "#FFE4E1",
    "moccasin": "#FFE4B5",
    "navajowhite": "#FFDEAD",
    "oldlace": "#FDF5E6",
    "olivedrab": "#6B8E23",
    "orangered": "#FF4500",
    "palegoldenrod": "#EEE8AA",
    "palegreen": "#98FB98",
    "paleturquoise": "#AFEEEE",
    "palevioletred": "#DB7093",
    "papayawhip": "#FFEFD5",
    "peachpuff": "#FFDAB9",
    "peru": "#CD853F",
    "powderblue": "#B0E0E6",
    "rosybrown": "#BC8F8F",
    "royalblue": "#4169E1",
    "saddlebrown": "#8B4513",
    "sandybrown": "#F4A460",
    "seagreen": "#2E8B57",
    "seashell": "#FFF5EE",
    "skyblue": "#87CEEB",
    "slateblue": "#6A5ACD",
    "slategray": "#708090",
    "snow": "#FFFAFA",
    "springgreen": "#00FF7F",
    "steelblue": "#4682B4",
    "thistle": "#D8BFD8",
    "tomato": "#FF6347",
    "wheat": "#F5DEB3",
    "whitesmoke": "#F5F5F5",
    "yellowgreen": "#9ACD32",
}

# Merge CSS3 colors into fashion colors
FASHION_COLORS.update(CSS3_COLORS)

# Reverse lookup: hex to name
HEX_TO_FASHION_COLOR = {v.upper(): k for k, v in FASHION_COLORS.items()}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB to hex string."""
    return f"#{r:02X}{g:02X}{b:02X}"


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Calculate perceptual color distance using weighted Euclidean distance.
    Weights account for human perception of color differences.
    """
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    
    # Weighted distance - humans are more sensitive to green, less to blue
    rmean = (r1 + r2) / 2
    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2
    
    # Redmean color distance formula for better perceptual accuracy
    return (
        (2 + rmean / 256) * dr ** 2 +
        4 * dg ** 2 +
        (2 + (255 - rmean) / 256) * db ** 2
    ) ** 0.5


def get_closest_fashion_color(hex_color: str) -> str:
    """
    Find the closest fashion color name for a hex code.
    
    Args:
        hex_color: Hex color code (e.g., "#FF5733" or "FF5733")
        
    Returns:
        Fashion color name (e.g., "burnt orange")
    """
    hex_color = hex_color.lstrip('#').upper()
    full_hex = f"#{hex_color}"
    
    # Exact match in fashion colors
    if full_hex in HEX_TO_FASHION_COLOR:
        return HEX_TO_FASHION_COLOR[full_hex]
    
    # Try webcolors first for standard CSS colors
    try:
        name = webcolors.hex_to_name(full_hex)
        return name
    except ValueError:
        pass
    
    # Find closest fashion color
    target_rgb = hex_to_rgb(hex_color)
    min_distance = float('inf')
    closest_name = "unknown"
    
    for name, hex_code in FASHION_COLORS.items():
        color_rgb = hex_to_rgb(hex_code.lstrip('#'))
        dist = color_distance(target_rgb, color_rgb)
        if dist < min_distance:
            min_distance = dist
            closest_name = name
    
    return closest_name


def normalize_color_name(color_name: str) -> str:
    """
    Normalize a color name to a standard fashion color.
    
    Args:
        color_name: Raw color name from extraction
        
    Returns:
        Normalized fashion color name
    """
    if not color_name:
        return "unknown"
    
    color_lower = color_name.lower().strip()
    
    # Direct match
    if color_lower in FASHION_COLORS:
        return color_lower
    
    # Remove common prefixes/suffixes
    prefixes = ["light ", "dark ", "deep ", "pale ", "bright ", "muted ", "soft ", "vivid "]
    suffixes = [" color", " colored", " tone", " shade", " hue"]
    
    cleaned = color_lower
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    for suffix in suffixes:
        if cleaned.endswith(suffix):
            cleaned = cleaned[:-len(suffix)]
    
    if cleaned in FASHION_COLORS:
        return cleaned
    
    # Fuzzy matching - check if any fashion color is contained in the name
    for fashion_color in FASHION_COLORS:
        if fashion_color in color_lower or color_lower in fashion_color:
            return fashion_color
    
    # Try to extract color from compound names like "navy-blue", "sky_blue"
    parts = re.split(r'[-_\s]+', color_lower)
    for part in parts:
        if part in FASHION_COLORS:
            return part
    
    # Check for compound colors
    compound = ' '.join(parts)
    if compound in FASHION_COLORS:
        return compound
    
    return color_lower  # Return original if no match found


def extract_colors_from_text(text: str) -> List[str]:
    """
    Extract all color names mentioned in a text.
    
    Args:
        text: Text potentially containing color names
        
    Returns:
        List of detected color names
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found_colors = []
    
    # Sort by length (longest first) to match compound colors before simple ones
    sorted_colors = sorted(FASHION_COLORS.keys(), key=len, reverse=True)
    
    for color in sorted_colors:
        # Use word boundaries to avoid partial matches
        pattern = rf'\b{re.escape(color)}\b'
        if re.search(pattern, text_lower):
            found_colors.append(color)
            # Remove found color to avoid double-matching
            text_lower = re.sub(pattern, '', text_lower)
    
    return found_colors


def extract_hex_codes(text: str) -> List[str]:
    """
    Extract all hex color codes from text.
    
    Args:
        text: Text potentially containing hex codes
        
    Returns:
        List of hex codes found
    """
    pattern = r'#?([0-9a-fA-F]{6})'
    matches = re.findall(pattern, text)
    return [f"#{m.upper()}" for m in matches]


def hex_to_color_name(hex_code: str) -> str:
    """
    Convert a hex code to a descriptive color name.
    
    Args:
        hex_code: Hex color code
        
    Returns:
        Color name
    """
    return get_closest_fashion_color(hex_code)


def get_color_category(color_name: str) -> str:
    """
    Get the broad category of a color (neutral, warm, cool).
    
    Args:
        color_name: Color name
        
    Returns:
        Category string: 'neutral', 'warm', or 'cool'
    """
    neutrals = {
        "white", "off-white", "cream", "ivory", "eggshell", "bone", "ecru",
        "black", "jet black", "charcoal", "anthracite",
        "gray", "grey", "silver", "slate", "stone", "ash", "pewter", "gunmetal",
        "heather gray", "light gray", "dark gray"
    }
    
    warm = {
        "red", "crimson", "scarlet", "cherry", "ruby", "garnet", "burgundy",
        "maroon", "wine", "oxblood", "bordeaux", "brick red", "vermillion",
        "orange", "tangerine", "apricot", "burnt orange", "copper", "bronze",
        "yellow", "gold", "golden", "mustard", "lemon", "honey", "amber",
        "brown", "chocolate", "coffee", "mocha", "chestnut", "mahogany",
        "tan", "camel", "beige", "khaki", "sand", "taupe", "caramel", "cognac",
        "rust", "terracotta", "sienna", "coral", "peach", "salmon"
    }
    
    color_lower = color_name.lower()
    
    if color_lower in neutrals:
        return "neutral"
    elif color_lower in warm:
        return "warm"
    else:
        return "cool"


# Export the main functions and data
__all__ = [
    'FASHION_COLORS',
    'get_closest_fashion_color',
    'normalize_color_name',
    'extract_colors_from_text',
    'extract_hex_codes',
    'hex_to_color_name',
    'hex_to_rgb',
    'rgb_to_hex',
    'get_color_category',
]
