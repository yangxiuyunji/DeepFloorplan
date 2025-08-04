#!/usr/bin/env python
"""最简演示 - 仅使用图像处理部分"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 颜色映射
floorplan_map = {
    0: [255,255,255], # background
    1: [192,192,224], # closet  
    2: [192,255,255], # bathroom
    3: [224,255,192], # living/kitchen/dining
    4: [255,224,128], # bedroom
    5: [255,160, 96], # hall
    6: [255,224,224], # balcony
    9: [255, 60,128], # door & window
    10:[  0,  0,  0]  # wall
}

def create_demo_result():
    print("Creating demo floorplan result...")
    
    # Load image
    img = Image.open('./demo/45719584.jpg').convert('RGB')
    img_resized = img.resize((512, 512))
    
    # Create a simple mock result for demonstration
    # In real usage, this would come from the neural network
    mock_result = np.zeros((512, 512), dtype=np.uint8)
    
    # Create some demo room regions
    mock_result[50:200, 50:200] = 4    # bedroom
    mock_result[50:200, 250:400] = 2   # bathroom
    mock_result[250:400, 50:300] = 3   # living room
    mock_result[250:400, 350:450] = 1  # closet
    mock_result[450:500, 50:450] = 5   # hall
    
    # Add walls and doors
    mock_result[0:50, :] = 10          # top wall
    mock_result[500:512, :] = 10       # bottom wall
    mock_result[:, 0:50] = 10          # left wall
    mock_result[:, 450:512] = 10       # right wall
    mock_result[200:250, 150:200] = 9  # door
    
    # Convert to RGB
    rgb_result = np.zeros((512, 512, 3), dtype=np.uint8)
    for label, color in floorplan_map.items():
        rgb_result[mock_result == label] = color
    
    # Create comparison plot
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(img_resized)
    plt.title('Original Floorplan')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(rgb_result)
    plt.title('Parsed Result (Demo)')
    plt.axis('off')
    
    # Save result
    plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
    print("Demo result saved as 'demo_output.png'")
    
    # Also save the colored floorplan separately
    Image.fromarray(rgb_result).save('floorplan_colored.png')
    print("Colored floorplan saved as 'floorplan_colored.png'")
    
    plt.show()

if __name__ == '__main__':
    create_demo_result()
