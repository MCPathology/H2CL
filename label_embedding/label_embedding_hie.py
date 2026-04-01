import torch
from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor
import os
from transformers import AutoProcessor, CLIPProcessor, CLIPModel
from open_clip import create_model_from_pretrained, get_tokenizer
# 1. Define label lists
level_1_names = [
    'Negative for Intraepithelial Lesion or Malignancy',
    'Atypical Squamous Cells',
    'Atypical Glandular Cells',
    'Microorganism'
]

level_2_names = [
    'Negative for Intraepithelial Lesion or Malignancy (Normal)',
    'Endocervical Cells',
    'Reactive Cellular Changes',
    'Metaplastic Squamous Cells',
    'Pregnancy-related Changes',
    'Atrophy',
    'Endometrial Cells',
    'Histiocytes',
    'Atypical Squamous Cells of Undetermined Significance',
    'Low-grade Squamous Intraepithelial Lesion',
    'Atypical Squamous Cells, cannot exclude HSIL',
    'High-grade Squamous Intraepithelial Lesion',
    'Squamous Cell Carcinoma',
    'Atypical Glandular Cells, Not Otherwise Specified',
    'Atypical Glandular Cells, Favor Neoplastic',
    'Adenocarcinoma',
    'Fungal organisms (typically Candida)',
    'Actinomyces',
    'Trichomonas vaginalis',
    'Herpes Simplex Virus',
    'Cellular changes consistent with Chlamydia'
]
level_3_names = ['Normal', 'ECC', 'RPC', 'MPC', 'PG', 'Atrophy', 'EMC', 'HCG', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'AGC-FN', 'AGC-ECC-NOS', 'AGC-EMC-NOS', 'ADC-ECC', 'ADC-EMC', 'FUNGI', 'ACTINO', 'TRI', 'HSV', 'CC']

levels_to_process = {
    'order': level_1_names,
    'family': level_2_names,
    'level_3': level_3_names
}

# 2. Setup model and output directory
def generate_prompts(full_name):
    """Generate a set of descriptive prompts for a single label."""
    # If it's an abbreviation, use the full name, otherwise use the label name directly
    
    # Prompt templates
    templates = [
    f"a microscopic image of {full_name}",
    f"a cytology smear diagnosed as {full_name}",
    f"a cytological image of {full_name}",
    f"cytological features of {full_name}",
    f"this is a case of {full_name}",
    f"microscopic view of {full_name} cells",
    f"{full_name}" # Also include the original label name itself
    ]
    return templates
    
def get_model(clip='MedCLIP'):
    if clip == 'MedCLIP':
        print("Loading MedCLIP model...")
        processor = MedCLIPProcessor()
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained()
    
    elif clip == 'CLIP':
        print("Loading original OpenAI CLIP model...")
        model_id = "openai/clip-vit-base-patch32"
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPModel.from_pretrained(model_id)

    elif clip == 'BioMedCLIP':
        print("Loading BioMedCLIP model...")
        model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        model, img_processor = create_model_from_pretrained(model_id)
        processor = get_tokenizer(model_id)
    else:
        raise ValueError(f"Unsupported model: '{clip}'. Please choose from 'MedCLIP', 'CLIP', 'BioMedCLIP'.")
    
    return model, processor
    
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

save_path = os.path.join(output_dir, f"label_embeddings.pt")
        # Move tensor back to CPU for saving
torch.save(final, save_path)


print("\nAll label embeddings have been generated and saved.")