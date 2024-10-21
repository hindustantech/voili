import argparse
import cv2
from model import Model

def argument_parser():
    parser = argparse.ArgumentParser(description="Violence detection")
    parser.add_argument('--image-path', type=str,
                        default='./data/7.jpg',
                        help='Path to your image')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # Parse command-line arguments
    args = argument_parser()
    
    # Initialize the model
    model = Model()

    # Read the input image
    image = cv2.imread(args.image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        print(f"Error: Unable to load image at '{args.image_path}'. Please check the path and try again.")
    else:
        # Perform prediction
        prediction = model.predict(image=image)
        
        # Retrieve and print the predicted label
        label = prediction['label']
        confidence = prediction['confidence']
        print(f'Predicted label: {label} (Confidence: {confidence:.2f})')
        
        # Display the image with the predicted label
        cv2.imshow(label.title(), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()  # Ensure all OpenCV windows are closed properly
