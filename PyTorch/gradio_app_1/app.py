
# This is a simple demonstration of Gradio, not a trained CNN model.

import gradio as gr

 # Define a function that receives a class index
def predict(class_idx):
    class_idx = int(class_idx)
    if class_idx == 0:
        return "Cat"
    if class_idx == 1:
        return "Dog"
    if class_idx == 2:
        return "Person"
    else:
        return "No class label found"

# Create a simple web interface    
demo = gr.Interface(
    fn=predict,        # Function executed when the user submits input
    inputs="text",
    outputs="text",
)

if __name__ == "__main__":  # Run only when this file is executed directly
    demo.launch()   # Start the Gradio web application