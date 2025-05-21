# ✈️ Aircraft Damage Classification and Captioning using Deep Learning & Transformers

##  Overview

This project is a combination of **computer vision** and **generative AI**, focused on analyzing and describing damage in aircraft images. It performs **two main tasks**:

1. **Classifies** aircraft damage as either a **dent** or a **crack** using a fine-tuned VGG16-based CNN.
2. **Generates captions and summaries** of the damage using a fine-tuned BLIP (Bootstrapped Language-Image Pretraining) transformer model.

---

##  Goals

- Detect and classify the type of damage in aircraft components.
- Automatically generate accurate, descriptive captions and summaries of the damaged area.
- Fine-tune a transformer model (BLIP) on a custom dataset of 30 aircraft images and captions.

---

## Techniques Used

###  Image Classification
- **Model**: VGG16 (transfer learning with frozen convolutional base)
- **Layers**: Flatten → Dense (ReLU) → Dropout → Output (sigmoid)
- **Training**: 12 epochs, binary cross-entropy loss
- **Libraries**: TensorFlow / Keras

###  Image Captioning
- **Model**: BLIP (`Salesforce/blip-image-captioning-base`)
- **Technique**: Fine-tuning on a custom dataset of 30 aircraft images with manually written captions
- **Library**: Hugging Face Transformers
- **Prompt style**:
  - Caption: `"This is a picture of..."`
  - Summary: `"This is a detailed photo showing..."`

---

##  Project Structure
aircraft_damage_project/
├── aircraft_classifier.py # CNN classification script
├── blip_finetune.py # Fine-tuning BLIP model on custom dataset
├── test_images/ # Folder with new test images
├── aircraft_caption_dataset/ # Dataset used to fine-tune BLIP
│ ├── images/
│ └── captions.json
├── best_blip_model/ # Fine-tuned BLIP model
├── README.md # Project overview

# aircraft-damage-captioning
Abdulazim haffar - 220208854
