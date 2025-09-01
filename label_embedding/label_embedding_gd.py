import torch
import os
from medclip import MedCLIPProcessor, MedCLIPModel, MedCLIPVisionModelViT

# --- 1. Define your data and Prompt templates ---

# Skin disease labels (please modify according to your actual situation)
level_2_names = ['Benign and Hyperplastic Lesions', 'Inflammatory Lesions', 'Gallbladder Lithiasis', 'Malignant Lesions']
level_3_names = [
    'Adenomyomatosis',                             # "Adenomyomatosis"
    'Gallbladder Polyps and Cholesterol Crystals', # "Polyps_and_cholesterol_crystals"
    'Extrinsic Gallbladder Lesion',                # "Abdomen_and_retroperitoneum"
    'Cholecystitis',                               # "Cholecystitis"
    'Gangrenous Cholecystitis',                    # "Membranous_and_gangrenous_cholecystitis"
    'Gallbladder Perforation',                     # "Perforation"
    'Gallbladder Wall Thickening',                 # "Various_causes_of_gallbladder_wall_thickening"
    'Gallstones',                                  # "Gallstones"
    'Gallbladder Carcinoma'                        # "Carcinoma"
]

levels_to_process = {
    'family': level_2_names,
    'target': level_3_names
}


def generate_prompts(label_name):
    templates = [
        f"an ultrasound image of {label_name}",
        f"gallbladder ultrasound showing {label_name}",
        f"ultrasound of a case diagnosed as {label_name}",
        f"a case of {label_name} on a gallbladder ultrasound",
        f"ultrasonographic features of {label_name}",
        f"an ultrasound scan showing {label_name}",
        f"{label_name}"  # Also include the label name itself
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
save_path = os.path.join(output_dir, f"gd.pt")
        # Move tensor back to CPU for saving
torch.save(final, save_path)


print("\nAll label embeddings have been generated and saved.")