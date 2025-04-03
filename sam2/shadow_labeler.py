import os
import json
import numpy as np
import cv2
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ShadowLabeler:
    def __init__(self, model_cfg, checkpoint_path):
        # Initialize SAM model
        self.sam = build_sam2(model_cfg, checkpoint_path).cuda()
        self.mask_generator = SAM2AutomaticMaskGenerator(
            self.sam, 
            stability_score_thresh=0.90,
            points_per_side=128
        )
        
        # Create necessary directories
        self.test_dir = Path("../data/imgs")
        self.test_dir.mkdir(exist_ok=True)
        self.json_dir = Path("../data/labels")
        self.json_dir.mkdir(exist_ok=True)
        
        # Initialize matplotlib figure
        plt.ion()  # Turn on interactive mode

        # Store the current response
        self.current_response = None
        
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key in ['y', 'n', 'q']:
            self.current_response = event.key
            plt.close()
    
    def get_unlabeled_images(self):
        """Get list of images that haven't been labeled yet"""
        image_extensions = {'.jpg', '.jpeg', '.png'}
        labeled_files = {json_file.stem for json_file in self.json_dir.glob('*.json')}
        return [
            img_path for img_path in self.test_dir.glob('*')
            if img_path.suffix.lower() in image_extensions
            and img_path.stem not in labeled_files
        ]
    
    def process_image(self, image_path):
        """Process single image and get user labels"""
        print(f"Processing {image_path}")
        
        # Load and process image
        image = Image.open(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return
            
        # Generate masks
        masks = self.mask_generator.generate(np.array(image))
        print('masks', len(masks))
        
        if not masks:
            print(f"No masks generated for {image_path}")
            return
            
        labels = []
        
        for idx, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.canvas.manager.set_window_title('Shadow Labeler')
            
            # Create colored overlay
            image_array = np.array(image)
            overlay = image_array.copy()
            overlay[mask] = overlay[mask] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
            
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            
            # Display the original image
            ax1.imshow(image_array)
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Display the overlay
            ax2.imshow(overlay)
            ax2.set_title('Overlay with Mask')
            ax2.axis('off')
            
            # Add title with instructions
            fig.suptitle(f"Mask {idx+1}/{len(masks)}\nPress 'y' for shadow, 'n' for not shadow, 'q' to quit",
                            fontsize=12, y=0.95)
            
            # Connect key event handler
            self.current_response = None
            fig.canvas.mpl_connect('key_press_event', self.on_key)
            
            # Show plot and wait for response
            plt.draw()
            plt.pause(0.1)  # Small pause to ensure window updates
            
            while self.current_response is None:
                plt.pause(0.1)
            
            if self.current_response == 'y':
                labels.append(True)
            elif self.current_response == 'n':
                labels.append(False)
            elif self.current_response == 'q':
                plt.close('all')
                return None

            plt.close(fig)
                
        print('finished looking at masks')
        plt.close('all')
        return labels
    
    def save_labels(self, image_path, labels):
        """Save labels to JSON file"""
        json_path = self.json_dir / f"{image_path.stem}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'image_path': str(image_path),
                'labels': labels
            }, f, indent=2)
    
    def run(self):
        """Main labeling loop"""
        unlabeled_images = self.get_unlabeled_images()
        if not unlabeled_images:
            print("No unlabeled images found in test_sam directory")
            return
            
        print(f"Found {len(unlabeled_images)} unlabeled images")
        
        for image_path in unlabeled_images:
            labels = self.process_image(image_path)
            if labels is None:  # User quit
                break
            self.save_labels(image_path, labels)
            print(f"Saved labels for {image_path}")

if __name__ == "__main__":
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint_path = "../checkpoints/sam2.1_hiera_large.pt"
    
    labeler = ShadowLabeler(model_cfg, checkpoint_path)
    labeler.run() 