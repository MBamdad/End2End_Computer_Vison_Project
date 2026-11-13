import gradio as gr  # Create the web interface
import os  # Handle file and folder paths
from core.predict import ImageClassifier  # Import our CNN prediction backend
from PIL import Image  # Open and save images


cwd = os.path.dirname(os.path.abspath(__file__))  # Get the folder containing app.py
model_path = os.path.join(cwd, "model", "cnn_128_model-100.pth")  # Path to trained model
class_name = {0: 'Cat', 1: 'Dog', 2: 'person'}  # Map class indices to names
classifier = ImageClassifier(model_path=model_path, class_name=class_name)  # Load trained CNN and prepare inference

def classify_image(image):  # Function called when user submits an image
    image_path = "uploaded_image.jpg"  # Temporary filename for uploaded image
    image.save(image_path)  # Save uploaded image locally

    label, output_path = classifier.predict(image_path)  # Predict class and generate labeled image

    return label, Image.open(output_path)  # Return class name and labeled image to Gradio
    
demo = gr.Interface(  # Create Gradio web application
    fn=classify_image,  # Function executed when user clicks Submit
    inputs=gr.Image(type='pil'),  # Accept uploaded image as a PIL image
    outputs=[gr.Textbox(label="Prediction"), gr.Image(label="Labeled Image")],  # Show predicted class and labeled image
    title="Image Clascification Gradio app",  # Application title
    description= "Upload an image to classify it as Dog, Cat,or Person"  # Instructions displayed to user
)

if __name__ == "__main__":  # Execute only when running app.py directly
    demo.launch()  # Start local Gradio server and display its URL