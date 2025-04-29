import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Charger le modèle BERT depuis Hugging Face
model_name = "Muselion/Test_BERT"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertForSequenceClassification.from_pretrained(model_name)

# Fonction pour faire les prédictions
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        logits = outputs.logits

    probabilities = torch.sigmoid(logits)
    threshold = 0.5
    predicted_labels = (probabilities >= threshold).squeeze().cpu().numpy()

    # Liste des noms de classes
    class_names = ['.net', 'actionscript-3', 'ajax', 'algorithm', 'apache-flex',
       'arrays', 'asp.net', 'asp.net-mvc', 'c', 'c#', 'c++', 'cocoa',
       'cocoa-touch', 'css', 'database', 'django', 'flash', 'html',
       'image', 'iphone', 'java', 'javascript', 'jquery', 'linq', 'linux',
       'macos', 'multithreading', 'mysql', 'objective-c', 'performance',
       'php', 'python', 'ruby', 'ruby-on-rails', 'security', 'sql',
       'sql-server', 'sql-server-2005', 'string', 't-sql',
       'user-interface', 'vb.net', 'visual-studio', 'visual-studio-2008',
       'wcf', 'web-services', 'windows', 'winforms', 'wpf', 'xml']

    active_classes = [class_names[i] for i in range(len(predicted_labels)) if predicted_labels[i] == 1]
    
    return ", ".join(active_classes)

# Créer une interface Gradio
interface = gr.Interface(fn=predict, inputs=gr.Textbox(lines=5, placeholder="Enter your text here"), outputs="text")

# Lancer l'interface
interface.launch()