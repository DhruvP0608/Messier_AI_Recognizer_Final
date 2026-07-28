# Messier Astronomical Object Recognition System

This project uses your Messier image dataset to identify an input astronomical image and return the closest matching Messier object along with readable details.
It now also supports training a real ML classifier on `SpaceDataset` and using that model during prediction.

## What It Does

- indexes the 110 reference images in `Dataset/`
- extracts lightweight visual features using Pillow
- compares a query image against the dataset
- returns the best matching Messier object and top alternatives
- can train a supervised classifier on `SpaceDataset/images/train` and evaluate on `SpaceDataset/images/test`
- uses the trained ML model to route predictions to relevant Messier categories before similarity matching

This approach is a prototype-based image recognition system. It is suitable for a lab demo, especially when the query images are from the same dataset or visually similar to it.

## Project Structure

- `main.py`: command-line entry point
- `demo_app.py`: desktop GUI demo built with tkinter
- `astronomy_recognizer/features.py`: image feature extraction
- `astronomy_recognizer/metadata.py`: Messier labels and descriptions
- `astronomy_recognizer/recognizer.py`: index building and prediction logic
- `artifacts/reference_index.json`: generated feature index

## How To Run

Build the reference index once:

```bash
python3 main.py build-index
```

Predict an image:

```bash
python3 main.py predict Dataset/Galaxy/M31.jpg
```

Launch the demo interface:

```bash
python3 demo_app.py
```

Show the top 5 matches:

```bash
python3 main.py predict Dataset/Galaxy/M31.jpg --top-k 5

Train and evaluate the SpaceDataset model:

```bash
python train_space_model.py --dataset "SpaceDataset/images"
```

Run the web server (ML-guided prediction is enabled automatically if `artifacts/space_category_model.pkl` exists):

```bash
python server.py
```
```

## Demo Flow

1. Build the index.
2. Launch `python3 demo_app.py` or run prediction on a test image.
3. Explain that the system compares extracted image features to known Messier references.
4. Show the predicted object name, category, confidence, and similar alternatives.

## Notes

- No heavy ML frameworks are required.
- The Messier detector is still similarity-based for final object matching, but can now be guided by a supervised model trained on SpaceDataset.
- Because the Messier catalog has one reference image per object, similarity retrieval remains useful as a robust fallback.
