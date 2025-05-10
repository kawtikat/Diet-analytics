import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000/segment_food/"

CLASS_NAMES = [
    "background", "candy", "egg tart", "french fries", "chocolate", "biscuit", "popcorn", "pudding", "ice cream",
    "cheese butter", "cake", "wine", "milkshake", "coffee", "juice", "milk", "tea", "almond", "red beans", "cashew",
    "dried cranberries", "soy", "walnut", "peanut", "egg", "apple", "date", "apricot", "avocado", "banana",
    "strawberry", "cherry", "blueberry", "raspberry", "mango", "olives", "peach", "lemon", "pear", "fig",
    "pineapple", "grape", "kiwi", "melon", "orange", "watermelon", "steak", "pork", "chicken duck", "sausage",
    "fried meat", "lamb", "sauce", "crab", "fish", "shellfish", "shrimp", "soup", "bread", "corn",
    "hamburg", "pizza", "hanamaki baozi", "wonton dumplings", "pasta", "noodles", "rice", "pie", "tofu", "eggplant",
    "potato", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "spring onion", "rape", "ginger", "okra",
    "lettuce", "pumpkin", "cucumber", "white radish", "carrot", "asparagus", "bamboo shoots", "broccoli", "celery stick", "cilantro mint",
    "snow peas", "cabbage", "bean sprouts", "onion", "pepper", "green beans", "French beans", "king oyster mushroom", "shiitake", "enoki mushroom",
    "oyster mushroom", "white button mushroom", "salad", "other ingredients"
]

def generate_color_map():
    """Generate a color map for visualization."""
    np.random.seed(42)
    color_map = np.random.randint(50, 200, (104, 3), dtype=np.uint8)
    color_map[0] = [0, 0, 0]  # Background in black
    return color_map

def process_segmentation_mask(mask, original_image):
    """Process the segmentation mask with opaque colors and centered ingredient labels."""
    # Generate color map and create a color mask
    color_map = generate_color_map()
    
    # Resize mask to match original image dimensions
    mask_resized = cv2.resize(mask, (original_image.size[0], original_image.size[1]), 
                            interpolation=cv2.INTER_NEAREST)
    
    # Create color mask
    color_mask = color_map[mask_resized]
    
    # Convert original image to numpy array if it's not already
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image.convert("RGB"))
    
    # Create semi-transparent overlay by combining original image and color mask
    overlay = cv2.addWeighted(color_mask, 0, original_image, 0.5, 0)  # Adjusted weights for transparency
    
    # Process each segment in the resized dimensions
    unique_segments = np.unique(mask_resized)
    for segment_id in unique_segments:
        if segment_id == 0:  # Skip background
            continue
            
        # Create mask for current segment in the resized dimensions
        segment_mask = (mask_resized == segment_id).astype(np.uint8)
        
        # Find connected components
        output = cv2.connectedComponentsWithStats(segment_mask, connectivity=8)
        num_labels, labels, stats, centroids = output
        
        # Process each connected component
        for i in range(1, num_labels):  # Skip background (0)
            # Get area of component
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area > 1000:  # Filter small components
                # Get centroid of component
                center_x = int(centroids[i][0])
                center_y = int(centroids[i][1])
                
                # Get ingredient name
                ingredient_name = CLASS_NAMES[segment_id]
                
                # Calculate text size and position
                font_scale = min(0.7, area / 10000)  # Slightly reduced font scale
                thickness = 1  # Reduced thickness for normal font weight
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(ingredient_name, font, font_scale, thickness)[0]
                
                # Center text
                text_x = int(center_x - text_size[0] / 2)
                text_y = int(center_y + text_size[1] / 2)
                
                # Create background rectangle
                padding = 4  # Slightly reduced padding
                bg_rect = [
                    text_x - padding,
                    text_y - text_size[1] - padding,
                    text_x + text_size[0] + padding,
                    text_y + padding
                ]
                
                # Draw semi-transparent white background rectangle
                cv2.rectangle(overlay,
                            (bg_rect[0], bg_rect[1]),
                            (bg_rect[2], bg_rect[3]),
                            (255, 255, 255),
                            -1)
                
                # Draw thinner text
                cv2.putText(overlay,
                           ingredient_name,
                           (text_x, text_y),
                           font,
                           font_scale,
                           (0, 0, 0),
                           thickness,
                           cv2.LINE_AA)
    
    return overlay


def main():
    st.title("Food Image Segmentation with CCNet")
    st.write("Upload a food image to detect ingredients.")

    uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                try:
                    # Prepare image for upload
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr = img_byte_arr.getvalue()

                    # Send request to FastAPI backend
                    files = {"file": ("image.png", img_byte_arr, "image/png")}
                    response = requests.post(BACKEND_URL, files=files)
                    response.raise_for_status()
                    
                    result = response.json()
                    
                    # Convert segmentation mask to array
                    segmentation_mask = np.array(result["segmentation_mask"], dtype=np.uint8)

                    # Process mask with the original image size
                    overlayed_image = process_segmentation_mask(segmentation_mask, image)

                    # Display the segmented image
                    st.image(overlayed_image, caption="Segmented Image with Ingredients", use_column_width=True)
                    
                    # Extract ingredient percentages and create pie chart
                    ingredient_percentages = result["ingredient_percentages"]
                    
                    # Filter out ingredients with very small percentages (less than 1%)
                    filtered_percentages = {k: v for k, v in ingredient_percentages.items() if v > 1.0}
                    
                    if filtered_percentages:
                        st.subheader("Ingredient Composition")
                        
                        # Create pie chart using matplotlib
                        fig, ax = plt.subplots(figsize=(10, 6))
                        wedges, texts, autotexts = ax.pie(
                            filtered_percentages.values(), 
                            labels=filtered_percentages.keys(),
                            autopct='%1.1f%%',
                            textprops={'fontsize': 9},
                            startangle=90
                        )
                        
                        # Equal aspect ratio ensures that pie is drawn as a circle
                        ax.axis('equal')
                        plt.setp(autotexts, size=8, weight='bold')
                        plt.tight_layout()
                        
                        # Display the pie chart
                        st.pyplot(fig)
                        
                        # Also display as a table for accessibility
                        st.subheader("Ingredient Percentages")
                        percentage_data = {"Ingredient": list(filtered_percentages.keys()), 
                                          "Percentage (%)": [f"{v:.1f}%" for v in filtered_percentages.values()]}
                        st.table(percentage_data)
                    else:
                        st.info("No significant ingredients detected for percentage visualization.")

                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to the backend server: {str(e)}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
