import torch
import os
from medclip import MedCLIPProcessor, MedCLIPModel, MedCLIPVisionModelViT

# --- 1. Define your data and Prompt templates ---

# Skin disease labels (please modify according to your actual situation)
level_1_names = ['benign', 'malignant']
level_2_names = ['benign keratinocytic', 'melanocytic', 'other benign', 'vascular', 'basal cell carcinoma', 'malignant keratinocytic', 'melanoma', 'squamous cell carcinoma']
level_3_names = ['actinic_cheilitis', 'cutaneous_horn', 'lichenoid', 'porokeratosis', 'seborrhoeic', 'solar_lentigo', 'wart', 'acral', 'atypical', 'blue', 'compound', 'congenital', 'dermal', 'halo', 'ink_spot_lentigo', 'involutingregressing',                      'irritated', 'junctional', 'lentigo', 'papillomatous', 'ungual', 'benign_nevus', 'chrondrodermatitis', 'dermatofibroma', 'eczema', 'excoriation', 'nail_dystrophy', 'scar', 'sebaceous_hyperplasia', 'angioma', 'haematoma',                           'other_vascular', 'telangiectasia', 'basal_cell_carcinoma', 'pigmented_basal_cell_carcinoma', 'superficial_basaal_cell_carcinoma', 'actinic', 'lentigo_maligna', 'malignant_melanoma', 'scc_in_situ', 'squamous_cell_carcinoma']

levels_to_process = {
    'order': level_1_names,
    'family': level_2_names,
    'target': level_3_names
}

# Mapping from abbreviation to full name to generate better prompts
abbreviation_map = {
    'bcc': 'Basal Cell Carcinoma',
    'scc': 'Squamous Cell Carcinoma'
}

def generate_prompts(label_name):
    """Generate a set of descriptive prompts for a single label."""
    # If it's an abbreviation, use the full name, otherwise use the label name directly
    full_name = abbreviation_map.get(label_name, label_name).replace('_', ' ')
    
    # Prompt templates
    templates = [
        f"a photo of a {full_name}",
        f"a photo of a skin lesion diagnosed as {full_name}",
        f"dermatoscopic image of {full_name}",
        f"clinical presentation of {full_name}",
        f"this is a case of {full_name}",
        f"an image showing {full_name} on the skin",
        f"{full_name}" # Also include the original label name itself
    ]
    return templates

# --- 2. Setup model and output directory ---
output_dir = './label_embedding/'
os.makedirs(output_dir, exist_ok=True)
final = {}
print("Loading MedCLIP model...")
processor = MedCLIPProcessor()
model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
model.from_pretrained()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded. Using device: {device}")

# --- 3. Generate, pool, and save embeddings for each label ---
# Set the model to evaluation mode and disable gradient calculation
model.eval()
with torch.no_grad():
    for level_key, names_list in levels_to_process.items():
        print(f"\n--- Processing {level_key} ---")
        
        # 1. Initialize a list to collect all embeddings for the current level
        level_embeddings_list = []
        
        # Iterate over each label in the current level
        for name in names_list:
            prompts = generate_prompts(name)
            inputs = processor(
                text=prompts, 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs.pop('token_type_ids', None)
            # outputs = model.get_text_features(**inputs)
            text_outputs = model.text_model(**inputs)
            text_embeds = text_outputs
            pooled_embedding = torch.mean(text_embeds, dim=0)
            
            # 2. Add the generated embedding to the list
            level_embeddings_list.append(pooled_embedding)
            
            print(f"  Generated pooled embedding for '{name}'.")

        # 3. Stack all tensors in the list into a single [N, dim] tensor
        level_tensor = torch.stack(level_embeddings_list).cpu()
        final[level_key] = level_tensor
print(final)
# 4. Save this final tensor
save_path = os.path.join(output_dir, f"skin.pt")
        # Move tensor back to CPU for saving
torch.save(final, save_path)


print("\nAll label embeddings have been generated and saved.")