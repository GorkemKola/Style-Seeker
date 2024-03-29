# Style-Seeker

### A project to recommend products from "https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset"

- This project designed for data mining class.
- The main purpose of the project is to design a system for users that want to find similar fashion products related to a product.
- A simple UI designed using gradio is going to meet the user, then when user uploads of a product image and the application will extract features of this image using PreTrained Vision Transformer (Not Fine Tuned as it is not a large scale application).
- These extracted features will be used to find K (how many wanted) similar products using ANNoy (Approximiate Nearest Neighbors) algorithm (Time Complexity: O(logN))
- The similar products with the features will be returned to the user.

- The project is on its final form. 

### To Run

- First, run preprocess.ipynb
- Then, run python -m feature_extraction.app to pass through the model and obtain feature vectors.
- Then, run python -m tree.tree to get Annoy Tree
- Then you can run python -m app and open the generated link to open Gradio User Interface.
- Finally you can choose how many images will be recommended and upload an image and get results.
