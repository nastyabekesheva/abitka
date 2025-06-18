import cv2

# Load original ears image
img = cv2.imread("/Users/nastyabekesheva/Projects/abitka/abitka/data/вушка (2) (1).png", cv2.IMREAD_UNCHANGED)

# Stretch horizontally (e.g., 1.3x width, same height)
new_width = int(img.shape[1] * 1.7)
new_height = img.shape[0]

# Resize image
resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

# Save new stretched image
cv2.imwrite("/Users/nastyabekesheva/Projects/abitka/abitka/data/ears_wide.png", resized)

# Load the image
img = cv2.imread("/Users/nastyabekesheva/Projects/abitka/abitka/data/спідниця (1).png", cv2.IMREAD_UNCHANGED)
if img is None:
    print("Failed to load image.")
    exit(1)

# Flip horizontally
flipped = cv2.flip(img, 1)  # 1 means horizontal flip

# Save the flipped image
cv2.imwrite("/Users/nastyabekesheva/Projects/abitka/abitka/data/skirt_flipped.png", flipped)

print("Image flipped and saved as scooter_flipped.png")