import os
import json
import torch
import numpy as np
import pandas as pd
import re
import spacy
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ast  # For safely evaluating string representations of lists
from datetime import datetime
from spacy.cli import download
from classifiers.area_classifier import AreaClassifier

class SimilarityCalculator:
    def __init__(self, request_type='bug'):
        """
        Initialize the enhanced similarity calculator with 4 SBERT models
        
        Args:
            request_type: 'bug' or 'feature'
        """
        self.request_type = request_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        self._set_random_seeds()
        
        # Initialize AreaClassifier for preprocessing and classification
        self.area_classifier = AreaClassifier(request_type=request_type)
        
        # Model storage (load on-demand)
        self.models = {}  # {variant: model}
        self.reference_embeddings = {}  # {variant: embeddings}
        self.reference_data = None
        
        # Set paths based on request type
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(self.base_path)
        
        # Path to reference data with pre-computed embeddings
        self.data_path = os.path.join(
            parent_dir, 
            'data',
            request_type,
            'reference data_with_embeddings.csv'
        )
        
        # Paths to SBERT models
        self.model_paths = {
            'with_filename': os.path.join(
                self.base_path, 'model', request_type, 'with filename'
            ),
            'without_filename': os.path.join(
                self.base_path, 'model', request_type, 'without filename'
            )
        }
        
        print(f"Initializing Enhanced Similarity Calculator for {request_type}")
        print(f"Using device: {self.device}")
        self._initialize()
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility in model operations only"""
        # Only set torch seeds for model reproducibility
        # Don't set numpy seed to avoid affecting pandas sampling
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
            torch.cuda.manual_seed_all(42)
        
        # Set deterministic behavior for CUDA operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _initialize(self):
        """Load reference data and models for current request type"""
        try:
            # Load reference data with pre-computed embeddings
            self._load_reference_data()
            
            # Load SBERT models for current request type
            self._load_models()
            
            print(f"Enhanced Similarity Calculator initialized for {self.request_type}")
        except Exception as e:
            print(f"Error initializing Enhanced Similarity Calculator: {str(e)}")
            raise
            
    def _load_reference_data(self):
        """Load reference data with pre-computed embeddings"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Reference data file not found: {self.data_path}")
        
        print(f"Loading reference data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.reference_data = df
        
        # Load pre-computed embeddings
        print("Loading pre-computed embeddings...")
        
        # Parse string embeddings to numpy arrays
        self.reference_embeddings = {
            'without_filename': self._parse_embeddings(df['without_filename_embeddings']),
            'with_filename': self._parse_embeddings(df['with_filename_embeddings'])
        }
        
        print(f"Loaded {len(df)} reference items with embeddings")
        print(f"Without filename embeddings shape: {self.reference_embeddings['without_filename'].shape}")
        print(f"With filename embeddings shape: {self.reference_embeddings['with_filename'].shape}")
    
    def _parse_embeddings(self, embedding_series):
        """Parse string representations of embeddings to numpy arrays"""
        try:
            embeddings = []
            for emb_str in embedding_series:
                # Convert string representation to list, then to numpy array
                emb_list = ast.literal_eval(emb_str)
                embeddings.append(np.array(emb_list))
            
            return np.array(embeddings)
        except (ValueError, SyntaxError) as e:
            raise RuntimeError(f"Error parsing embeddings: {str(e)}")
    
    def _load_models(self):
        """Load both SBERT model variants for current request type"""
        variants = ['with_filename', 'without_filename']
        
        for variant in variants:
            model_dir = self.model_paths[variant]
            
            # Find the SBERT model file in the directory
            model_files = []
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.pt') and 'sbert' in file.lower():
                        model_files.append(os.path.join(model_dir, file))
            
            if not model_files:
                raise FileNotFoundError(f"No SBERT model file found in {model_dir}")
            
            model_path = model_files[0]  # Use the first matching file
            self.models[variant] = self._load_sbert_model(model_path, variant)
        
        print(f"Loaded {len(self.models)} SBERT models for {self.request_type}")
    
    def _load_sbert_model(self, model_path, variant):
        """Load a single SBERT model from file"""
        print(f"Loading SBERT model for {variant}: {model_path}")
        
        try:
            # Load the model using SentenceTransformer with all-mpnet-base-v2 architecture
            model = SentenceTransformer('all-mpnet-base-v2')
            
            # Load the fine-tuned weights
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Filter out classifier layers that don't belong to SentenceTransformer
            filtered_state_dict = {}
            for key_name, value in state_dict.items():
                # Skip classifier layers
                if not key_name.startswith('classifier.'):
                    filtered_state_dict[key_name] = value
            
            # Load the filtered state dict with strict=False to allow missing keys
            model.load_state_dict(filtered_state_dict, strict=False)
            
            # Move to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            print(f"Successfully loaded SBERT model for {variant}")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Error loading SBERT model from {model_path}: {str(e)}")
    
    def switch_request_type(self, new_type):
        """
        Switch request type and reload models/data
        
        Args:
            new_type: 'bug' or 'feature'
        """
        if new_type == self.request_type:
            print(f"Already using {new_type} request type")
            return
        
        print(f"Switching from {self.request_type} to {new_type}")
        
        try:
            # Clear current models to free memory
            self.models.clear()
            self.reference_embeddings.clear()
            
            # Switch area classifier
            result = self.area_classifier.switch_model(new_type)
            if not result['success']:
                raise RuntimeError(f"Failed to switch area classifier: {result['message']}")
            
            # Update request type and paths
            self.request_type = new_type
            parent_dir = os.path.dirname(self.base_path)
            
            self.data_path = os.path.join(
                parent_dir, 'data', new_type, 'reference data_with_embeddings.csv'
            )
            
            self.model_paths = {
                'with_filename': os.path.join(
                    self.base_path, 'model', new_type, 'with filename'
                ),
                'without_filename': os.path.join(
                    self.base_path, 'model', new_type, 'without filename'
                )
            }
            
            # Load new reference data and models
            self._load_reference_data()
            self._load_models()
            
            print(f"Successfully switched to {new_type}")
            
        except Exception as e:
            raise RuntimeError(f"Error switching to {new_type}: {str(e)}")
    
    def _generate_embedding(self, preprocessed_text, variant):
        """Generate embedding for preprocessed text using specified model variant"""
        if variant not in self.models:
            raise ValueError(f"Model variant {variant} not loaded")
        
        model = self.models[variant]
        
        # Handle empty text
        if not preprocessed_text or not preprocessed_text.strip():
            preprocessed_text = " "
        
        try:
            with torch.no_grad():
                embedding = model.encode(
                    [preprocessed_text],
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False
                )
                
                # Convert to CPU and numpy
                embedding = embedding.cpu().numpy()[0]
                return embedding
                
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {str(e)}")
    
    def find_similar_requests(self, title, description, comments="", filename="", top_k=20):
        """
        Find similar requests using Option A: Full Pipeline Integration
        
        Args:
            title: Title of the request
            description: Description or body of the request
            comments: Optional comments related to the request
            filename: Optional filename related to the request
            top_k: Number of similar items to retrieve
        
        Returns:
            List of dictionaries with similar items and their similarity scores
        """
        try:
            # 1. Get classification results (this also does preprocessing)
            classification_results = self.area_classifier.predict(
                title, description, comments, filename
            )
            
            # 2. Get the preprocessed text that was used in classification
            preprocessed_text = self.area_classifier.preprocess_text(
                title, description, comments, filename
            )
            
            # 3. Determine variant based on filename presence
            has_filename = filename and filename.strip()
            variant = 'with_filename' if has_filename else 'without_filename'
            
            print(f"Using {variant} variant for similarity calculation")
            print(f"Preprocessed text length: {len(preprocessed_text)} characters")
            
            # 4. Generate embedding using the same preprocessed text from classification
            query_embedding = self._generate_embedding(preprocessed_text, variant)
            
            # 5. Calculate similarities with pre-computed embeddings
            reference_embeddings = self.reference_embeddings[variant]
            similarities = cosine_similarity([query_embedding], reference_embeddings)[0]
            
            # 6. Get top-k most similar items
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # 7. Format results with classification info
            results = self._format_results(similarities, top_indices, classification_results)
            
            print(f"Found {len(results)} similar requests")
            return results
            
        except Exception as e:
            print(f"Error finding similar requests: {str(e)}")
            raise
    
    def _format_results(self, similarities, top_indices, classification_results):
        """Format similarity results with classification information"""
        results = []
        
        for i, idx in enumerate(top_indices):
            item_data = self.reference_data.iloc[idx]
            similarity_score = float(similarities[idx])
            
            # Create formatted result
            formatted_result = {
                'similarity': round(similarity_score * 100, 1),  # Convert to percentage
                'title': str(item_data.get('title', '')),
                'description': item_data.get('body', '')[:500] + '...' if len(str(item_data.get('body', ''))) > 500 else str(item_data.get('body', '')),
                'areas': [],
                'date': '',
                'id': str(item_data.get('id', '')),  # Convert to string to avoid numpy int64 issues
                'issue_url': str(item_data.get('issue_url', '')),
                'classification': classification_results if i == 0 else None  # Include classification only for top result
            }
            
            # Process labels
            labels = item_data.get('labels', '')
            if isinstance(labels, str) and labels.strip():
                try:
                    # Try to parse as JSON/list if it's a string representation
                    if labels.startswith('[') and labels.endswith(']'):
                        labels = ast.literal_eval(labels)
                    else:
                        # Split by comma if it's a comma-separated string
                        labels = [label.strip().strip("'\"") for label in labels.split(',')]
                except:
                    # If parsing fails, treat as single label
                    labels = [labels.strip()]
            elif not isinstance(labels, list):
                labels = []
            
            # Convert labels to area format
            for label in labels:
                if isinstance(label, str) and label.strip():
                    formatted_result['areas'].append({
                        'name': label.strip(),
                        'color': self._get_color_for_label(label.strip(), similarity_score),
                        'confidence': round(float(similarity_score * 100), 1)  # Ensure it's a Python float
                    })
            
            # Add date - use various possible date fields
            date_fields = ['created_at', 'updated_at', 'date']
            for date_field in date_fields:
                if date_field in item_data and item_data[date_field]:
                    formatted_result['date'] = str(item_data[date_field])
                    break
            else:
                formatted_result['date'] = datetime.now().strftime('%Y-%m-%d')
            
            results.append(formatted_result)
        
        return results
    
    def _get_color_for_label(self, label, confidence):
        """
        Return a Tailwind CSS color class based on the label and confidence.
        This matches the color mapping used in the area classifier.
        
        Args:
            label: The label name
            confidence: The confidence score (0-1)
            
        Returns:
            String containing Tailwind CSS classes for background and text color
        """
        # Map Kubernetes area labels to specific colors - matching area_classifier.py
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

# Test function
if __name__ == "__main__":
    # Example usage
    sim = SimilarityCalculator()
    results = sim.find_similar_requests(
        "Issue with network policy", 
        "The network policy doesn't apply correctly to pods with multiple labels",
        filename="network-policy.yaml",
        top_k=5
    )
    for i, result in enumerate(results):
        print(f"{i+1}. Similarity: {result['similarity']:.2f}%")
        print(f"   Title: {result['title']}")
        print(f"   Areas: {[a['name'] for a in result['areas']]}")
        if result.get('classification'):
            print(f"   Classification: {[p['label'] for p in result['classification']]}")
        print("")
