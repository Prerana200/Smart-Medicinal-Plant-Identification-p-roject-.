import gradio as gr
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

# ── Load model and class names ────────────────────────────────────────────────
model = tf.keras.models.load_model("plant_classifier_final.keras")
with open("class_names (1).json") as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = 224

# ── Plant info dictionary ─────────────────────────────────────────────────────
PLANT_INFO = {
    'Aloevera':            {'scientific': 'Aloe barbadensis', 'uses': 'Burns, skin care, digestion, wound healing'},
    'Amla':                {'scientific': 'Phyllanthus emblica', 'uses': 'Vitamin C, hair care, immunity, digestion'},
    'Amruthaballi':        {'scientific': 'Tinospora cordifolia', 'uses': 'Immunity booster, fever, jaundice'},
    'Arali':               {'scientific': 'Nerium oleander', 'uses': 'Skin diseases, external use only (toxic internally)'},
    'Ashoka':              {'scientific': 'Saraca asoca', 'uses': 'Menstrual disorders, uterine health, fever'},
    'Ashwagandha':         {'scientific': 'Withania somnifera', 'uses': 'Stress, stamina, adaptogen, sleep aid'},
    'Astma_weed':          {'scientific': 'Euphorbia hirta', 'uses': 'Asthma, bronchitis, respiratory issues'},
    'Avacado':             {'scientific': 'Persea americana', 'uses': 'Heart health, skin nourishment, anti-inflammatory'},
    'Badipala':            {'scientific': 'Wrightia tinctoria', 'uses': 'Skin disorders, psoriasis, hair care'},
    'Balloon_Vine':        {'scientific': 'Cardiospermum halicacabum', 'uses': 'Arthritis, rheumatism, skin diseases'},
    'Bamboo':              {'scientific': 'Bambusa vulgaris', 'uses': 'Respiratory disorders, fever, digestive aid'},
    'Basale':              {'scientific': 'Basella alba', 'uses': 'Constipation, skin health, anaemia'},
    'Beans':               {'scientific': 'Phaseolus vulgaris', 'uses': 'Protein source, diabetes management, heart health'},
    'Betel':               {'scientific': 'Piper betle', 'uses': 'Oral health, digestive aid, antiseptic'},
    'Betel_Nut':           {'scientific': 'Areca catechu', 'uses': 'Digestive stimulant, anthelmintic'},
    'Bhrami':              {'scientific': 'Bacopa monnieri', 'uses': 'Memory, cognition, anxiety, epilepsy'},
    'Brahmi':              {'scientific': 'Bacopa monnieri', 'uses': 'Memory, cognition, anxiety, epilepsy'},
    'Bringaraja':          {'scientific': 'Eclipta prostrata', 'uses': 'Hair growth, liver health, skin diseases'},
    'Camphor':             {'scientific': 'Cinnamomum camphora', 'uses': 'Pain relief, cold, respiratory congestion'},
    'Caricature':          {'scientific': 'Graptophyllum pictum', 'uses': 'Haemorrhoids, skin inflammation, ear infections'},
    'Castor':              {'scientific': 'Ricinus communis', 'uses': 'Laxative, joint pain, skin moisturiser'},
    'Catharanthus':        {'scientific': 'Catharanthus roseus', 'uses': 'Diabetes, anti-cancer properties, blood pressure'},
    'Chakte':              {'scientific': 'Caesalpinia sappan', 'uses': 'Blood purifier, anti-inflammatory'},
    'Chilly':              {'scientific': 'Capsicum annuum', 'uses': 'Digestive stimulant, pain relief, metabolism boost'},
    'Citron_lime':         {'scientific': 'Citrus medica', 'uses': 'Digestion, liver health, vitamin C'},
    'Coffee':              {'scientific': 'Coffea arabica', 'uses': 'Alertness, antioxidant, headache relief'},
    'Common_rue':          {'scientific': 'Ruta graveolens', 'uses': 'Muscle spasms, menstrual aid, insect repellent'},
    'Coriander':           {'scientific': 'Coriandrum sativum', 'uses': 'Digestive aid, anti-inflammatory, blood sugar'},
    'Curry':               {'scientific': 'Murraya koenigii', 'uses': 'Digestive aid, diabetes, hair health'},
    'Curry_Leaf':          {'scientific': 'Murraya koenigii', 'uses': 'Digestive aid, diabetes, hair health'},
    'Doddpathre':          {'scientific': 'Coleus amboinicus', 'uses': 'Cough, cold, respiratory disorders'},
    'Doddapatre':          {'scientific': 'Coleus amboinicus', 'uses': 'Cough, cold, respiratory disorders'},
    'Drumstick':           {'scientific': 'Moringa oleifera', 'uses': 'Nutrition, anti-inflammatory, blood sugar'},
    'Ekka':                {'scientific': 'Calotropis gigantea', 'uses': 'Skin diseases, fever, toothache'},
    'Eucalyptus':          {'scientific': 'Eucalyptus globulus', 'uses': 'Respiratory relief, antiseptic, pain relief'},
    'Ganigale':            {'scientific': 'Notonia grandiflora', 'uses': 'Wounds, skin ailments'},
    'Ganike':              {'scientific': 'Solanum nigrum', 'uses': 'Fever, liver disorders, skin diseases'},
    'Gasagase':            {'scientific': 'Papaver somniferum', 'uses': 'Insomnia, pain relief, cough'},
    'Geranium':            {'scientific': 'Pelargonium graveolens', 'uses': 'Skin care, anxiety relief, anti-inflammatory'},
    'Ginger':              {'scientific': 'Zingiber officinale', 'uses': 'Nausea, digestion, anti-inflammatory'},
    'Globe_Amaranth':      {'scientific': 'Gomphrena globosa', 'uses': 'Cough, bronchitis, urinary issues'},
    'Guava':               {'scientific': 'Psidium guajava', 'uses': 'Diarrhoea, diabetes, oral health'},
    'Henna':               {'scientific': 'Lawsonia inermis', 'uses': 'Hair dye, skin cooling, antimicrobial'},
    'Hibiscus':            {'scientific': 'Hibiscus rosa-sinensis', 'uses': 'Hair care, blood pressure, cholesterol'},
    'Honge':               {'scientific': 'Pongamia pinnata', 'uses': 'Skin diseases, wound healing, anti-inflammatory'},
    'Insulin':             {'scientific': 'Costus pictus', 'uses': 'Diabetes management, blood sugar control'},
    'Jackfruit':           {'scientific': 'Artocarpus heterophyllus', 'uses': 'Immunity, digestive health, energy'},
    'Jasmine':             {'scientific': 'Jasminum officinale', 'uses': 'Stress relief, skin care, antiseptic'},
    'Kambajala':           {'scientific': 'Hardwickia binata', 'uses': 'Rheumatism, skin conditions'},
    'Kamakasturi':         {'scientific': 'Abelmoschus moschatus', 'uses': 'Digestive disorders, nervine tonic'},
    'Kasambruga':          {'scientific': 'Cassia auriculata', 'uses': 'Skin diseases, diabetes, eye disorders'},
    'Kepala':              {'scientific': 'Cocos nucifera', 'uses': 'Hydration, skin nourishment, antimicrobial'},
    'Kohlrabi':            {'scientific': 'Brassica oleracea', 'uses': 'Digestive health, immune support'},
    'Lantana':             {'scientific': 'Lantana camara', 'uses': 'External antiseptic, wound healing, fever'},
    'Lemon':               {'scientific': 'Citrus limon', 'uses': 'Vitamin C, digestion, detox, immunity'},
    'Lemongrass':          {'scientific': 'Cymbopogon citratus', 'uses': 'Digestion, fever, antimicrobial, anxiety'},
    'Lemon_grass':         {'scientific': 'Cymbopogon citratus', 'uses': 'Digestion, fever, antimicrobial, anxiety'},
    'Malabar_Nut':         {'scientific': 'Justicia adhatoda', 'uses': 'Cough, asthma, bronchitis, fever'},
    'Malabar_Spinach':     {'scientific': 'Basella alba', 'uses': 'Constipation, skin health, anaemia'},
    'Mango':               {'scientific': 'Mangifera indica', 'uses': 'Digestion, immunity, skin health, antioxidant'},
    'Marigold':            {'scientific': 'Calendula officinalis', 'uses': 'Wound healing, skin inflammation, antiseptic'},
    'Mint':                {'scientific': 'Mentha spicata', 'uses': 'Digestion, headache, cold, oral health'},
    'Nagadali':            {'scientific': 'Ruta graveolens', 'uses': 'Muscle spasms, menstrual aid, insect repellent'},
    'Neem':                {'scientific': 'Azadirachta indica', 'uses': 'Antibacterial, skin diseases, fever, dental care'},
    'Nelavembu':           {'scientific': 'Andrographis paniculata', 'uses': 'Fever, liver health, immunity, dengue'},
    'Nerale':              {'scientific': 'Syzygium cumini', 'uses': 'Diabetes, digestive disorders, anti-inflammatory'},
    'Nithyapushpa':        {'scientific': 'Catharanthus roseus', 'uses': 'Diabetes, anti-cancer, blood pressure'},
    'Nooni':               {'scientific': 'Morinda citrifolia', 'uses': 'Immunity, pain relief, antioxidant'},
    'Onion':               {'scientific': 'Allium cepa', 'uses': 'Antibacterial, heart health, cold relief'},
    'Padri':               {'scientific': 'Stereospermum suaveolens', 'uses': 'Fever, liver disorders, anti-inflammatory'},
    'Palak':               {'scientific': 'Spinacia oleracea', 'uses': 'Iron deficiency, bone health, antioxidant'},
    'Papaya':              {'scientific': 'Carica papaya', 'uses': 'Dengue, digestion, skin, anti-parasitic'},
    'Parijatha':           {'scientific': 'Nyctanthes arbor-tristis', 'uses': 'Arthritis, fever, skin diseases'},
    'Pea':                 {'scientific': 'Pisum sativum', 'uses': 'Protein source, heart health, blood sugar'},
    'Pepper':              {'scientific': 'Piper nigrum', 'uses': 'Digestion, cold, antimicrobial, metabolism'},
    'Pomegranate':         {'scientific': 'Punica granatum', 'uses': 'Antioxidant, heart health, anti-inflammatory'},
    'Pumpkin':             {'scientific': 'Cucurbita pepo', 'uses': 'Eye health, immunity, digestive aid'},
    'Radish':              {'scientific': 'Raphanus sativus', 'uses': 'Liver detox, digestion, respiratory health'},
    'Raktachandini':       {'scientific': 'Pterocarpus santalinus', 'uses': 'Skin diseases, fever, anti-inflammatory'},
    'Rose':                {'scientific': 'Rosa damascena', 'uses': 'Skin care, stress relief, digestive aid'},
    'Sampige':             {'scientific': 'Michelia champaca', 'uses': 'Fever, skin disorders, aromatic'},
    'Sapota':              {'scientific': 'Manilkara zapota', 'uses': 'Energy, digestion, antioxidant'},
    'Seethaashoka':        {'scientific': 'Saraca asoca', 'uses': 'Menstrual health, uterine disorders'},
    'Seethapala':          {'scientific': 'Annona squamosa', 'uses': 'Antioxidant, anti-tumor, digestive health'},
    'Tamarind':            {'scientific': 'Tamarindus indica', 'uses': 'Digestive aid, anti-inflammatory, fever'},
    'Taro':                {'scientific': 'Colocasia esculenta', 'uses': 'Digestive health, immunity, heart health'},
    'Tecoma':              {'scientific': 'Tecoma stans', 'uses': 'Diabetes, fever, anti-inflammatory'},
    'Thumbe':              {'scientific': 'Leucas aspera', 'uses': 'Cold, cough, skin diseases, anti-parasitic'},
    'Tomato':              {'scientific': 'Solanum lycopersicum', 'uses': 'Antioxidant, heart health, skin health'},
    'Tulsi':               {'scientific': 'Ocimum tenuiflorum', 'uses': 'Respiratory disorders, immunity, stress, fever'},
    'Tulasi':              {'scientific': 'Ocimum tenuiflorum', 'uses': 'Respiratory disorders, immunity, stress, fever'},
    'Turmeric':            {'scientific': 'Curcuma longa', 'uses': 'Anti-inflammatory, antioxidant, wound healing'},
    'Wood_sore':           {'scientific': 'Oxalis corniculata', 'uses': 'Scurvy, fever, skin infections, digestive aid'},
}

def get_info(name):
    if name in PLANT_INFO:
        return PLANT_INFO[name]
    for k, v in PLANT_INFO.items():
        if k.lower() in name.lower():
            return v
    return {
        'scientific': 'Unknown',
        'uses': 'No info available',
        'toxic': ('TOXIC' in name.upper())
    }

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def compute_gradcam(inp, orig_arr):
    try:
        # Find last Conv2D anywhere in the model (including sub-models)
        last_conv_layer = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
            elif hasattr(layer, 'layers'):
                for sub in layer.layers:
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv_layer = sub

        if last_conv_layer is None:
            print("No Conv2D layer found")
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp, training=False)
            class_idx = int(tf.argmax(preds[0]))
            score = preds[:, class_idx]

        grads  = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
        cam    = tf.nn.relu(conv_out[0] @ pooled[..., tf.newaxis])
        cam    = tf.squeeze(cam).numpy()
        cam    = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        heat = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = np.clip(0.55 * orig_arr + 0.45 * heat, 0, 255).astype(np.uint8)
        return Image.fromarray(overlay)
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ── Confidence bar chart ──────────────────────────────────────────────────────
def make_confidence_chart(top3_names, top3_confs):
    fig, ax = plt.subplots(figsize=(5, 3))
    colors = ['#2ecc71', '#3498db', '#95a5a6']
    ax.barh(top3_names[::-1], [c * 100 for c in top3_confs[::-1]],
            color=colors[::-1], edgecolor='white')
    for i, conf in enumerate(top3_confs[::-1]):
        ax.text(conf * 100 + 0.5, i, f'{conf*100:.1f}%',
                va='center', fontsize=10, fontweight='bold', color='white')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)', fontsize=10, color='white')
    ax.set_title('Top-3 Predictions', fontsize=11, fontweight='bold', color='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='white', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#555')
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')
    plt.tight_layout()
    chart_path = "/tmp/chart.png"
    plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    return chart_path

# ── Main prediction function ──────────────────────────────────────────────────
def classify_plant(image):
    if image is None:
        return None, None, "⚠️ Please upload an image first."

    img_pil  = Image.fromarray(image).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    orig_arr = np.array(img_pil, dtype=np.float32)
    inp      = np.expand_dims(preprocess_input(orig_arr.copy()), 0)

    probs      = model.predict(inp, verbose=0)[0]
    top3_idx   = np.argsort(probs)[::-1][:3]
    top3_names = [CLASS_NAMES[i] for i in top3_idx]
    top3_confs = [float(probs[i]) for i in top3_idx]

    best_name = top3_names[0]
    best_conf = top3_confs[0]
    info      = get_info(best_name)

    gradcam_img = compute_gradcam(inp, orig_arr)
    chart_path  = make_confidence_chart(top3_names, top3_confs)

    toxic_icon  = "☠️ Toxic Plant" if info['toxic'] else "🌿 Medicinal Plant"
    result_text = f"""### {toxic_icon}
---
**Identified**  : {best_name}
**Confidence**  : {best_conf*100:.1f}%
**Scientific**  : {info['scientific']}
**Uses**        : {info['uses']}
**Top-3 Predictions:**
1. {top3_names[0]:<30} {top3_confs[0]*100:.1f}%
2. {top3_names[1]:<30} {top3_confs[1]*100:.1f}%
3. {top3_names[2]:<30} {top3_confs[2]*100:.1f}%
"""
    return gradcam_img, chart_path, result_text

# ── Gradio UI ─────────────────────────────────────────────────────────────────
css = """
.gradio-container { background-color: #0f0f1a !important; color: white !important; }
.gr-button-primary { background: linear-gradient(135deg, #2ecc71, #27ae60) !important;
                     color: white !important; font-size: 16px !important;
                     border-radius: 8px !important; }
"""

with gr.Blocks(css=css, title="🌿 Plant Classifier") as demo:
    gr.Markdown("""
    # 🌿 Medicinal & Toxic Plant Classifier
    Upload a leaf image to identify the plant, see **Top-3 predictions**, **Grad-CAM attention map**, and medicinal information.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input  = gr.Image(label="Upload Leaf Image", type="numpy", height=300)
            classify_btn = gr.Button("🔍 Classify Plant", variant="primary", size="lg")

        with gr.Column(scale=1):
            gradcam_output = gr.Image(label="Grad-CAM Attention Map", height=300)

        with gr.Column(scale=1):
            chart_output = gr.Image(label="Confidence Chart", height=300)

    with gr.Row():
        result_output = gr.Markdown(label="Classification Result")

    classify_btn.click(
        fn=classify_plant,
        inputs=image_input,
        outputs=[gradcam_output, chart_output, result_output]
    )

demo.launch()
