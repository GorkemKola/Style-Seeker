import numpy as np
import gradio as gr


image = gr.inputs.Image()
gr.Interface(fn=predict_image, inputs=image, outputs=label,interpretation='default').launch(debug='True')

