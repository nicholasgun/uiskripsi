import torch
import numpy as np
import json
import re
import pandas as pd
from transformers import DebertaTokenizer, DebertaModel
import torch.nn as nn
import spacy
from spacy.cli import download
import os
import time
from pathlib import Path

class DeBERTaClassifier(nn.Module):
    """
    A classifier model based on DeBERTa for multi-label classification.
    
    This model uses a pre-trained DeBERTa model as the encoder and adds a 
    classification head on top with sigmoid activation for multi-label output.
    
    Args:
        num_labels (int): Number of classes in the multi-label classification task.
    """
    def __init__(self, num_labels):
        super().__init__()
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)
        # Freeze all parameters in DeBERTa
        for param in self.deberta.parameters():
            param.requires_grad = False
        # Unfreeze encoder parameters for fine-tuning
        for layer in self.deberta.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        # Here we'll use the [CLS] token (first token) representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        # Return raw logits for BCEWithLogitsLoss (sigmoid will be applied in the loss function)
        return self.classifier(cls_output)

class DeBERTaCNNClassifier(nn.Module):
    """
    A hybrid classifier model based on DeBERTa and CNN for multi-label classification.
    
    This model uses a pre-trained DeBERTa model as the encoder to generate embeddings,
    followed by a CNN layer for feature extraction, pooling for feature summarization,
    and a fully connected layer for classification.
    
    The architecture consists of:
    1. Embedding layer (DeBERTa)
    2. Convolutional layers with multiple filter sizes
    3. Max pooling layer
    4. Fully connected layer for classification
    
    Args:
        num_labels (int): Number of classes in the multi-label classification task.
        filter_sizes (list): List of filter sizes for CNN layers (default: [2, 4, 6, 8, 10])
        num_filters (int): Number of filters per size (default: 64)
    """
    def __init__(self, num_labels, filter_sizes=[2, 4, 6, 8, 10], num_filters=64):
        super().__init__()
        # Load pre-trained DeBERTa model
        self.deberta = DebertaModel.from_pretrained('microsoft/deberta-base')
        hidden_size = 768  # DeBERTa hidden size
        
        # Freeze all parameters in DeBERTa
        for param in self.deberta.parameters():
            param.requires_grad = False
        # Unfreeze encoder parameters for fine-tuning
        # We'll unfreeze the last 3 encoder layers
        for layer in self.deberta.encoder.layer[-3:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        # CNN layers with multiple filter sizes
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,
                out_channels=num_filters,
                kernel_size=(filter_size, hidden_size),
                stride=1
            )
            for filter_size in filter_sizes
        ])
        
        # Dropout layer
        self.dropout = nn.Dropout(0.1)
        
        # Output layer - concatenate all CNN outputs
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_labels)

    def forward(self, input_ids, attention_mask):
        # Get DeBERTa embeddings
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get the entire sequence output (not just the [CLS] token)
        sequence_output = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Add a channel dimension for CNN
        x = sequence_output.unsqueeze(1)  # Shape: (batch_size, 1, seq_len, hidden_size)
        
        # Apply CNN with different filter sizes and max-over-time pooling
        pooled_outputs = []
        for conv in self.convs:
            # Apply convolution
            # Conv shape: (batch_size, num_filters, seq_len-filter_size+1, 1)
            conv_output = conv(x).squeeze(3)
            
            # Apply ReLU
            conv_output = torch.relu(conv_output)
            
            # Apply max-over-time pooling
            # Pooled shape: (batch_size, num_filters, 1)
            pooled = nn.functional.max_pool1d(
                conv_output, 
                kernel_size=conv_output.shape[2]
            ).squeeze(2)
            
            pooled_outputs.append(pooled)
        
        # Concatenate the outputs from different filter sizes
        # Combined shape: (batch_size, num_filters * len(filter_sizes))
        x = torch.cat(pooled_outputs, dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply the classifier
        return self.classifier(x)

class AreaClassifier:
    """
    A wrapper class for both DeBERTa and DeBERTa-CNN classifiers that handles loading 
    models for both bug and feature requests, with separate models for scenarios with 
    and without filename. Automatically selects the appropriate model based on input.
    
    Model Selection Logic:
    - If filename is provided: Uses DeBERTa model (with filename variant)
    - If no filename: Uses DeBERTa-CNN model (without filename variant)
    """
    def __init__(self, request_type='bug'):
        """
        Initialize the classifier with the specified request type models.
        Other model types will be loaded on-demand when needed.
        
        Args:
            request_type (str): The initial active request type - 'bug' or 'feature'
        """
        # Validate request_type
        if request_type not in ['bug', 'feature']:
            raise ValueError("request_type must be either 'bug' or 'feature'")
        
        self.request_type = request_type
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.TOKENIZER_PATH = 'microsoft/deberta-base'
        
        # Initialize tokenizer and NLP components (shared between models)
        self._load_tokenizer()
        self._init_nlp()
        
        # Dictionary to store models and their associated data
        self.models = {}
        
        # Load only the initial request type models
        print(f"Loading {request_type} models...")
        start_time = time.time()
        
        # Load only the specified model type initially
        self._load_model_data(request_type)
        
        # Set the active model
        self.active_model = request_type
        
        print(f"Initialized AreaClassifier with {request_type} models in {time.time() - start_time:.2f} seconds")

    def _load_tokenizer(self):
        """Load the DeBERTa tokenizer."""
        self.tokenizer = DebertaTokenizer.from_pretrained(self.TOKENIZER_PATH)

    def _init_nlp(self):
        """Initialize spaCy model for text preprocessing."""
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            
    def _load_model_data(self, model_type):
        """
        Load model data for a specific request type (bug or feature).
        Loads both 'with filename' (DeBERTa) and 'without filename' (DeBERTa-CNN) variants.
        
        Args:
            model_type (str): 'bug' or 'feature'
        """
        print(f"Loading {model_type} models...")
        
        # Initialize storage for this model type
        self.models[model_type] = {}
        
        # Load both variants: with filename and without filename
        variants = {
            'with_filename': {
                'folder': 'with filename',
                'model_class': DeBERTaClassifier,
                'model_file': 'DeBERTa augmented {} 10 {}.pt'.format(
                    model_type, 
                    'features' if model_type == 'bug' else 'labels'
                )
            },
            'without_filename': {
                'folder': 'without filename', 
                'model_class': DeBERTaCNNClassifier,
                'model_file': 'DeBERTa-CNN augmented {} 10 labels.pt'.format(model_type)
            }
        }
        
        for variant_name, variant_config in variants.items():
            try:
                # Set paths based on model type and variant
                model_folder = f"classifiers/{model_type}/{variant_config['folder']}"
                model_dir = self.base_dir / model_folder
                
                # Set paths for model loading
                model_path = model_dir / variant_config['model_file']
                label_encoder_path = model_dir / "label_encoder.json"
                selected_labels_path = model_dir / "selected_labels.json"
                vocabulary_path = model_dir / "vocabulary.csv"
                
                # Check if all required files exist
                required_files = [model_path, label_encoder_path, selected_labels_path, vocabulary_path]
                missing_files = [f for f in required_files if not f.exists()]
                
                if missing_files:
                    print(f"Error: Missing files for {model_type} {variant_name} model:")
                    for f in missing_files:
                        print(f"  - {f}")
                    raise FileNotFoundError(f"Required model files are missing for {model_type} {variant_name}")
                
                # Load label encoder data
                with open(label_encoder_path, 'r') as f:
                    label_encoder_data = json.load(f)
                    
                # Load selected labels data
                try:
                    with open(selected_labels_path, 'r') as f:
                        selected_labels_data = json.load(f)
                        print(f"Using {len(selected_labels_data['selected_labels'])} selected labels for {model_type} {variant_name} model")
                except (FileNotFoundError, KeyError) as e:
                    # If file is not found or doesn't contain selected_labels key, use all labels
                    selected_labels_data = {"selected_labels": label_encoder_data["classes"]}
                    print(f"Selected labels file not found or invalid. Using all {len(selected_labels_data['selected_labels'])} labels for {model_type} {variant_name} model")
                    
                # Load vocabulary
                try:
                    vocab_df = pd.read_csv(vocabulary_path)
                    vocabulary_words = set(vocab_df['Word'].tolist())
                    print(f"Loaded {len(vocabulary_words)} words from vocabulary file for {model_type} {variant_name} model")
                except Exception as e:
                    print(f"Error loading vocabulary for {model_type} {variant_name}: {e}")
                    vocabulary_words = set()  # Fallback to empty set
                    
                # Determine number of labels from selected_labels
                num_labels = len(selected_labels_data['selected_labels'])
                    
                # Load the model with appropriate class
                model_class = variant_config['model_class']
                model = model_class(num_labels=num_labels)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                print(f"{model_type} {variant_name} model loaded successfully with {num_labels} output labels")
                    
                # Store all model data in the models dictionary
                self.models[model_type][variant_name] = {
                    'model': model,
                    'label_encoder_data': label_encoder_data,
                    'selected_labels_data': selected_labels_data,
                    'vocabulary_words': vocabulary_words,
                    'num_labels': num_labels
                }
                
            except Exception as e:
                print(f"Error loading {model_type} {variant_name} model: {e}")
                # Continue loading other models even if one fails
                continue

    def switch_model(self, request_type):
        """
        Switch the active model based on request type.
        If the target model type is not loaded, it will be loaded dynamically.
        
        Args:
            request_type (str): The type of request - 'bug' or 'feature'
            
        Returns:
            dict: Result with success status and any loading information
        """
        if request_type == self.active_model:
            # Already using this model
            return {"success": True, "message": f"Already using {request_type} model", "loaded_new": False}
            
        if request_type not in ['bug', 'feature']:
            raise ValueError("request_type must be either 'bug' or 'feature'")
        
        # Check if the target model type is loaded
        if request_type not in self.models:
            # Need to load the model dynamically
            print(f"Loading {request_type} models for the first time...")
            try:
                start_time = time.time()
                self._load_model_data(request_type)
                load_time = time.time() - start_time
                print(f"Successfully loaded {request_type} models in {load_time:.2f} seconds")
                
                # Update active model
                self.active_model = request_type
                self.request_type = request_type  # For backward compatibility
                
                return {
                    "success": True, 
                    "message": f"Loaded and switched to {request_type} model", 
                    "loaded_new": True,
                    "load_time": load_time
                }
            except Exception as e:
                error_msg = f"Error loading {request_type} models: {str(e)}"
                print(error_msg)
                return {
                    "success": False, 
                    "message": error_msg, 
                    "loaded_new": False
                }
        else:
            # Model is already loaded, just switch
            self.active_model = request_type
            self.request_type = request_type  # For backward compatibility
            print(f"Switched to {request_type} model (instant switch)")
            return {"success": True, "message": f"Switched to {request_type} model", "loaded_new": False}

    def get_model_info(self):
        """
        Get information about loaded models.
        
        Returns:
            Dictionary containing information about all loaded models
        """
        info = {
            'active_model': self.active_model,
            'loaded_models': {}
        }
        
        for model_type, variants in self.models.items():
            info['loaded_models'][model_type] = {}
            for variant_name, variant_data in variants.items():
                info['loaded_models'][model_type][variant_name] = {
                    'num_labels': variant_data['num_labels'],
                    'vocab_size': len(variant_data['vocabulary_words']),
                    'model_class': variant_data['model'].__class__.__name__
                }
        
        return info

    def preprocess_text(self, title, description, comments='', filename=''):
        """
        Preprocess text following the steps:
        1. Combine text based on filename presence:
           - If filename provided: title + description (exclude comments)
           - If no filename: title + description + comments
        2. Convert to lowercase
        3. Remove line breaks
        4. Remove non-alphanumeric characters
        
        If no filename provided:
        5. Remove stopwords and Lemmatize text
        6. Filter words based on pre-loaded vocabulary
        
        If filename provided:
        5. Filter words based on pre-loaded vocabulary
        6. Remove stopwords and Lemmatize text
        7. Add filename to the end of description
        
        Args:
            title: The title of the issue
            description: The description of the issue
            comments: Optional comments related to the issue
            filename: Optional filename related to the issue
            
        Returns:
            Preprocessed text ready for model input
        """
        # Determine which model variant to use based on filename presence
        has_filename = filename and filename.strip()
        variant = 'with_filename' if has_filename else 'without_filename'
        
        # Get appropriate model's vocabulary
        vocabulary_words = self.models[self.active_model][variant]['vocabulary_words']
        
        # 1. Combine text based on filename presence
        if has_filename:
            # With filename: use title + description (exclude comments)
            all_text = title + " " + description
        else:
            # Without filename: use title + description + comments
            all_text = title + " " + description + " " + comments
        
        # 2. Convert to lowercase
        all_text = all_text.lower()
        
        # 3. Remove line breaks
        all_text = all_text.replace('\r', ' ')
        all_text = all_text.replace('\n', ' ')
        
        # 4. Remove non-alphanumeric characters
        all_text = all_text.replace(r'[^a-zA-Z0-9 ]', '')
        
        # Process differently based on whether filename is provided
        if not has_filename:
            # No filename: Remove stopwords and lemmatize first, then vocabulary filtering
            # 5. Remove stopwords and Lemmatize text
            doc = self.nlp(all_text)
            all_text = ' '.join([word.lemma_ for word in doc if not word.is_stop])
            
            # 6. Filter words based on the pre-loaded vocabulary file
            words = all_text.split()
            filtered_words = [word for word in words if word in vocabulary_words]
            all_text = ' '.join(filtered_words)
        else:
            # Filename provided: Vocabulary filtering first, then stopwords/lemmatization
            # 5. Filter words based on the pre-loaded vocabulary file
            words = all_text.split()
            filtered_words = [word for word in words if word in vocabulary_words]
            all_text = ' '.join(filtered_words)

            # 6. Remove stopwords and Lemmatize text
            doc = self.nlp(all_text)
            all_text = ' '.join([word.lemma_ for word in doc if not word.is_stop])

            # 7. Add filename to the end of description
            all_text += " " + filename
        
        return all_text

    def predict(self, title, description, comments='', filename='', confidence_threshold=0.5):
        """
        Predict the area labels for a given issue title and description.
        Automatically selects the appropriate model based on filename presence:
        - If filename provided: Uses DeBERTa model (with filename variant)
        - If no filename: Uses DeBERTa-CNN model (without filename variant)
        
        Args:
            title: The title of the issue
            description: The description of the issue
            comments: Optional comments related to the issue
            filename: Optional filename related to the issue
            confidence_threshold: Minimum confidence score to include a label in results
            
        Returns:
            List of dictionaries containing predicted labels and confidence scores
        """
        # Determine which model variant to use based on filename presence
        has_filename = filename and filename.strip()
        variant = 'with_filename' if has_filename else 'without_filename'
        
        # Check if the required model variant exists
        if variant not in self.models[self.active_model]:
            raise ValueError(f"Model variant '{variant}' not available for {self.active_model} type. "
                           f"Available variants: {list(self.models[self.active_model].keys())}")
        
        # Get the appropriate model data
        active_model_data = self.models[self.active_model][variant]
        model = active_model_data['model']
        selected_labels_data = active_model_data['selected_labels_data']
        
        print(f"Using {self.active_model} model with {variant} variant ({'DeBERTa' if has_filename else 'DeBERTa-CNN'})")
        
        # Preprocess the text using the vocabulary-based filtering
        text = self.preprocess_text(title, description, comments, filename)
        
        # Tokenize for the model
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Get model predictions
        with torch.no_grad():
            logits = model(inputs['input_ids'], inputs['attention_mask'])
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get label names and confidences
        label_names = selected_labels_data['selected_labels']
        
        # Since we're training with the selected labels directly, the model outputs
        # probabilities for each selected label in order, no need for indices mapping
        results = [
            {
                "name": label, 
                "confidence": float(prob),
                "color": self._get_color_for_label(label, float(prob))
            }
            for label, prob in zip(label_names, probs) if prob >= confidence_threshold
        ]
        
        # Sort results by confidence score in descending order
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
    
    def _get_color_for_label(self, label, confidence):
        """
        Return a Tailwind CSS color class based on the label and confidence.
        
        Args:
            label: The label name
            confidence: The confidence score (0-1)
            
        Returns:
            String containing Tailwind CSS classes for background and text color
        """
        # Map Kubernetes area labels to specific colors based on label_encoder.json
        # Using more vibrant colors for better visibility
        color_map = {
            "area/api": "bg-purple-500 text-white",
            "area/apiserver": "bg-purple-600 text-white", 
            "area/batch": "bg-orange-500 text-white",
            "area/client-libraries": "bg-cyan-500 text-white",
            "area/cloudprovider": "bg-sky-500 text-white",
            "area/code-generation": "bg-amber-500 text-white",
            "area/code-organization": "bg-neutral-500 text-white",
            "area/conformance": "bg-stone-500 text-white",
            "area/custom-resources": "bg-violet-500 text-white",
            "area/deflake": "bg-rose-500 text-white",
            "area/dependency": "bg-slate-500 text-white",
            "area/e2e-test-framework": "bg-pink-500 text-white",
            "area/etcd": "bg-emerald-500 text-white",
            "area/ipvs": "bg-blue-500 text-white",
            "area/kube-proxy": "bg-indigo-500 text-white",
            "area/kubeadm": "bg-fuchsia-500 text-white",
            "area/kubectl": "bg-teal-500 text-white",
            "area/kubelet": "bg-lime-500 text-white",
            "area/network-policy": "bg-blue-600 text-white",
            "area/provider/aws": "bg-yellow-500 text-white",
            "area/provider/azure": "bg-blue-500 text-white",
            "area/provider/gcp": "bg-red-500 text-white",
            "area/release-eng": "bg-green-500 text-white",
            "area/security": "bg-red-600 text-white",
            "area/test": "bg-pink-500 text-white",
        }
        
        # Check if we have an exact match for this label
        if label in color_map:
            return color_map[label]
            
        # If no exact match, try partial matching
        for key in color_map:
            if key.lower() in label.lower():
                return color_map[key]
        
        # Default colors based on confidence - using more vibrant colors
        if confidence > 0.8:
            return "bg-emerald-500 text-white"
        elif confidence > 0.6:
            return "bg-amber-500 text-white"
        else:
            return "bg-gray-500 text-white"
