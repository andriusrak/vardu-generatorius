import streamlit as st
import torch
import torch.nn as nn
import json
import random

# Model class definition (must match your training model)
class GenderAwareTransformer(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, forward_expansion):
        super(GenderAwareTransformer, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gender_embed = nn.Embedding(2, embed_size)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, embed_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        self.output_layer = nn.Linear(embed_size, vocab_size)

    def forward(self, x, gender):
        char_embeddings = self.embed(x)
        gender_embeddings = self.gender_embed(gender).unsqueeze(1).expand(-1, x.size(1), -1)
        x = char_embeddings + gender_embeddings + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x

@st.cache_resource
def load_model():
    # Load mappings
    with open('name_mappings.json', 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    
    # Create model
    model = GenderAwareTransformer(
        vocab_size=mappings['vocab_size'],
        embed_size=128,
        num_heads=8,
        forward_expansion=4
    )
    
    # Load trained weights
    model.load_state_dict(torch.load('name_model.pt'))
    model.eval()
    
    return model, mappings

# Add this with your other UI elements
temperature = st.slider(
    "Temperatūra (Aukštesnė = kūrybiškesni vardai)",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Dėl mažesnių verčių vardai gaunami konservatyvesni, o dėl didesnių - kūrybiškesni."
)

# Modify the generate_name function to use temperature
def generate_name(model, mappings, gender, start_str='', max_length=20, temperature=1.0):
    with torch.no_grad():
        if not start_str:
            # Pick a random letter from the character-to-integer mapping
            start_str = random.choice(list(mappings['char_to_int'].keys()))
        else:
            # Convert start_str to lowercase to ensure compatibility
            start_str = start_str.lower()
            
        char_to_int = mappings['char_to_int']
        int_to_char = {int(k): v for k, v in mappings['int_to_char'].items()}
        
        chars = [char_to_int[c] for c in start_str]
        input_seq = torch.tensor(chars).unsqueeze(0)
        gender_tensor = torch.tensor([gender])
        
        output_name = start_str
        for _ in range(max_length - len(start_str)):
            output = model(input_seq, gender_tensor)
            # Apply temperature to logits
            logits = output[0, -1] / temperature
            probabilities = torch.softmax(logits, dim=0)
            next_char_idx = torch.multinomial(probabilities, 1).item()
            next_char = int_to_char[next_char_idx]
            
            if next_char == ' ':
                break
                
            output_name += next_char
            input_seq = torch.cat([input_seq, torch.tensor([[next_char_idx]])], dim=1)
            
        return output_name.capitalize()
# Streamlit interface
st.title("Lietuviškų vardų generatorius 🎲")

# Load model
model, mappings = load_model()

# Create two columns
col1, col2 = st.columns(2)

with col1:
    gender = st.radio("Vardo lytis:", ["Vyriškas", "Moteriškas"])
    
with col2:
    start_letter = st.text_input("Pradinė raidė (neprivaloma):", "")

# Generate button
if st.button("Generuoti vardą", type="primary"):
    gender_val = 0 if gender == "Vyriškas" else 1
    generated_name = generate_name(model, mappings, gender_val, start_letter, temperature=temperature)
    st.success(f"Sugeneruotas vardas: {generated_name}")

# Add some information
st.markdown("""
---
### Apie
Ši programėlė generuoja lietuviškus vardus naudodama dirbtinį intelektą. Modelis buvo apmokytas naudojant tikrus lietuviškus vardus.
Gali generuoti tiek vyriškus, tiek moteriškus vardus.

#### Kaip naudotis:
1. Pasirinkite lytį (vyrišką / moterišką)
2. Pasirinktinai įveskite pradinę raidę
3. Spustelėkite „Generuoti vardą“.
""")
