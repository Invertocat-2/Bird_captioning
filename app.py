from flask import Flask, render_template, request
import torch
from PIL import Image
import os
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch.nn as nn
import matplotlib.pyplot as plt

# Flask app
app = Flask(__name__)

# Ensure static folder exists
os.makedirs("static", exist_ok=True)

# --- Model Definition ---
class BirdCaptioningModel(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.base_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.hidden_size = self.base_model.decoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)
    
    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        outputs = self.base_model(
            pixel_values=pixel_values,
            decoder_input_ids=input_ids,
            decoder_attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.decoder_hidden_states[-1][:, 0, :]
        class_logits = self.classifier(hidden_states)
        return outputs.logits, class_logits

# --- Helper Functions ---
def load_species_mapping(species_file='species.csv'):
    if os.path.exists(species_file):
        species_idx_to_name = {}
        with open(species_file, 'r') as f:
            for line in f:
                idx, name = line.strip().split(',', 1)
                species_idx_to_name[int(idx)] = name
        return species_idx_to_name
    else:
        return {i: f"Bird_Species_{i}" for i in range(200)}

def load_model(model_path='best_bird_captioning_model.pth', device='cpu'):
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer.pad_token = tokenizer.eos_token

    model = BirdCaptioningModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, image_processor, tokenizer

# Load once
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, image_processor, tokenizer = load_model(device=device)
species_mapping = load_species_mapping()

# --- Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']
    image = Image.open(image_file).convert("RGB")
    image_filename = os.path.join("static", image_file.filename)
    image.save(image_filename)

    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        output_ids = model.base_model.generate(pixel_values, max_length=75, num_beams=4, early_stopping=True)
        _, class_logits = model(pixel_values, input_ids=torch.zeros((1, 1), dtype=torch.long, device=device))

    probs = torch.nn.functional.softmax(class_logits, dim=1)
    confidence, pred_idx = torch.max(probs, dim=1)
    species = species_mapping.get(pred_idx.item(), f"Unknown (Class {pred_idx.item()})")
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return render_template(
        'result.html',
        image_path=image_filename,
        caption=caption,
        species=species,
        confidence=round(confidence.item() * 100, 2)
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
