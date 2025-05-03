import cv2
import numpy as np
import pandas as pd
import sys

def image_to_csv(image_path, output_csv):
    # Load the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return
    
    # Resize image to a fixed size (optional)
    image_resized = cv2.resize(image, (100, 100))  # Resize to 100x100
    
    # Flatten the image
    image_flattened = image_resized.flatten()
    
    # Convert to DataFrame
    df = pd.DataFrame([image_flattened])
    
    # Save to CSV
    df.to_csv(output_csv, index=False, header=False)
    
    print(f"CSV file saved as: {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python image_to_csv.py <image_path> <output_csv>")
        sys.exit(1)

    image_path = sys.argv[1]
    output_csv = sys.argv[2]
    
    image_to_csv(image_path, output_csv)
