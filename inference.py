import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import os

class BirdCaptioningModel(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()
        self.base_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.hidden_size = self.base_model.decoder.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        outputs = self.base_model(pixel_values=pixel_values, decoder_input_ids=input_ids,
                                  decoder_attention_mask=attention_mask,
                                  output_hidden_states=True, return_dict=True)
        hidden_states = outputs.decoder_hidden_states[-1][:, 0, :]
        class_logits = self.classifier(hidden_states)
        return outputs.logits, class_logits

def load_species_mapping(species_file=None):
    if species_file and os.path.exists(species_file):
        species_idx_to_name = {}
        with open(species_file, 'r') as f:
            for line in f:
                idx, name = line.strip().split(',', 1)
                species_idx_to_name[int(idx)] = name
        return species_idx_to_name
    else:
        return {i: f"Bird_Species_{i}" for i in range(200)}

def load_model(model_path='best_bird_captioning_model.pth', device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer.pad_token = tokenizer.eos_token
    model = BirdCaptioningModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, image_processor, tokenizer

def generate_caption_and_classify(model, image_path, image_processor, tokenizer, species_idx_to_name, device=None):
    if device is None:
        device = next(model.parameters()).device
    image = Image.open(image_path).convert('RGB')
    pixel_values = image_processor(image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        output_ids = model.base_model.generate(pixel_values, max_length=75, num_beams=4, early_stopping=True)
        _, class_logits = model(pixel_values=pixel_values, input_ids=torch.zeros((1, 1), dtype=torch.long, device=device))
        probs = torch.nn.functional.softmax(class_logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)
        class_idx = class_idx.item()
        confidence = confidence.item() * 100
        species = species_idx_to_name.get(class_idx, f"Unknown (Class {class_idx})")
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption, species, confidence

def predict_bird_image(image_path, model_path='best_bird_captioning_model.pth', species_map_path=None):
    global loaded_model, image_processor, tokenizer, species_mapping
    if 'loaded_model' not in globals():
        loaded_model, image_processor, tokenizer = load_model(model_path)
        species_mapping = load_species_mapping(species_map_path)
    caption, species, confidence = generate_caption_and_classify(loaded_model, image_path, image_processor, tokenizer, species_mapping)
    return caption, species, confidence
