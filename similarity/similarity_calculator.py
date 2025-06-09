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

class SimilarityCalculator:
    def __init__(self, request_type='bug'):
        """
        Initialize the similarity calculator for the given request type
        
        Args:
            request_type: 'bug' or 'feature'
        """
        self.request_type = request_type
        self.model = None
        self.reference_embeddings = None
        self.reference_data = None
        self.vocabulary_words = None
        self.nlp = None
        
        # Set paths based on request type
        self.base_path = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(self.base_path)  # Get parent directory
        
        # Path to the data directory in the parent directory
        self.data_path = os.path.join(
            parent_dir, 
            'data',
            request_type,
            f'{request_type}_preprocessed_train_data_with_embeddings.csv'
        )
        
        # Path to vocabulary file based on request type
        self.vocabulary_path = os.path.join(
            parent_dir,
            'classifiers',
            request_type,
            'vocabulary.csv'
        )
        
        # Path to pre-trained SBERT model
        self.model_path = os.path.join(self.base_path, 'model', 'sbert.pt')
        self.base_model = 'all-mpnet-base-v2'  # Base model architecture
        
        print(f"Initializing Similarity Calculator for {request_type}")
        self._initialize()
    
    def _initialize(self):
        """Load the model and reference data"""
        try:
            # Check if CUDA is available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            # Set random seed for reproducibility
            np.random.seed(42)
            torch.manual_seed(42)
            
            # Initialize spaCy for text preprocessing
            self._init_nlp()
            
            # Load vocabulary
            self._load_vocabulary()
            
            # Load model
            self.model = self._load_model()
            self.model = self.model.to(device)
            self.model.eval()
            
            # Load reference data
            self._load_reference_data()
            
            print(f"Similarity Calculator initialized for {self.request_type}")
        except Exception as e:
            print(f"Error initializing Similarity Calculator: {str(e)}")
            raise
            
    def _init_nlp(self):
        """Initialize spaCy model for text preprocessing."""
        try:
            # Try to load the English model
            self.nlp = spacy.load('en_core_web_sm')
            print("Loaded spaCy model 'en_core_web_sm'")
        except OSError:
            # If model is not found, download it
            print("Downloading spaCy model 'en_core_web_sm'...")
            download('en_core_web_sm')
            self.nlp = spacy.load('en_core_web_sm')
            print("Downloaded and loaded spaCy model 'en_core_web_sm'")
    
    def _load_vocabulary(self):
        """Load vocabulary from CSV file."""
        try:
            print(f"Loading vocabulary from: {self.vocabulary_path}")
            vocab_df = pd.read_csv(self.vocabulary_path)
            self.vocabulary_words = set(vocab_df['Word'].tolist())
            print(f"Loaded {len(self.vocabulary_words)} words from vocabulary file")
        except Exception as e:
            print(f"Error loading vocabulary: {str(e)}")
            self.vocabulary_words = set()  # Fallback to empty set
            print("Warning: Using empty vocabulary set")
    
    def _load_model(self):
        """Load the sentence transformer model with pre-trained weights"""
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            if os.path.exists(self.model_path):
                print(f"Loading pre-trained SBERT weights from: {self.model_path}")
                
                # Load the base model
                base_model = SentenceTransformer(self.base_model, device=device)
                print(f"Loaded base SBERT model ({self.base_model})")
                
                # Load the model state dictionary
                state_dict = torch.load(self.model_path, map_location=device)
                
                # Filter out classifier keys that might cause issues
                filtered_state_dict = {k: v for k, v in state_dict.items() 
                                    if not k.startswith('classifier.')}
                
                # Load the filtered state dictionary
                # Use strict=False to ignore missing keys
                missing_keys, unexpected_keys = base_model.load_state_dict(filtered_state_dict, strict=False)
                
                print(f"Loaded SBERT model weights from {self.model_path}")
                if missing_keys:
                    print(f"Warning: Missing keys: {missing_keys}")
                if unexpected_keys:
                    print(f"Warning: Unexpected keys: {unexpected_keys}")
                    
                return base_model
            else:
                print(f"Pre-trained model file not found: {self.model_path}")
                print(f"Falling back to base model: {self.base_model}")
                return SentenceTransformer(self.base_model, device=device)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            # Fallback to standard model
            print("WARNING: Could not load custom model. Using a default pre-trained model instead.")
            print("This is not the model you intended to use!")
            return SentenceTransformer('all-mpnet-base-v2', device=device)
    
    def _load_reference_data(self):
        """Load reference data with embeddings"""
        if not os.path.exists(self.data_path):
            print(f"Reference data file not found: {self.data_path}")
            raise FileNotFoundError(f"Reference data file not found: {self.data_path}")
        
        print(f"Loading reference data from {self.data_path}")
        df = pd.read_csv(self.data_path)
        self.reference_data = df
        
        # Check if we have pre-computed embeddings
        embedding_column = 'embedding'
        have_precomputed_embeddings = embedding_column in df.columns
        
        if have_precomputed_embeddings:
            print(f"Using pre-computed embeddings from '{embedding_column}' column")
            
            # Convert string representations of embeddings to numpy arrays if needed
            if isinstance(df[embedding_column].iloc[0], str):
                try:
                    self.reference_embeddings = np.array([
                        np.array(ast.literal_eval(emb)) for emb in df[embedding_column]
                    ])
                except (ValueError, SyntaxError):
                    print("Error parsing embedding strings. Falling back to generating embeddings.")
                    have_precomputed_embeddings = False
            else:
                self.reference_embeddings = np.array(df[embedding_column].tolist())
        
        # If we don't have usable pre-computed embeddings, generate them
        if not have_precomputed_embeddings:
            # Prepare reference text if not already present
            text_column = 'all_text'
            if text_column not in df.columns:
                print(f"Column {text_column} not found in data. Creating from title, description, and comments...")
                df[text_column] = df.apply(
                    lambda row: self._preprocess_text(
                        row.get('title', ''), 
                        row.get('body', row.get('description', '')), 
                        row.get('comments', row.get('all_comments', ''))
                    ),
                    axis=1
                )
            
            reference_texts = df[text_column].tolist()
            print("Generating embeddings for reference texts...")
            self.reference_embeddings = self._get_embeddings(reference_texts)
    
    def _preprocess_text(self, title, description, comments='', filename='', max_length=512):
        """
        Preprocess text following the same steps as in the AreaClassifier:
        1. Combine title, description and comments
        2. Convert to lowercase
        3. Remove line breaks
        4. Remove non-alphanumeric characters
        5. Filter words based on pre-loaded vocabulary
        6. Remove stopwords
        7. Lemmatize text
        8. Add filename to the end of description if provided
        9. Truncate if necessary
        
        Args:
            title: The title of the issue
            description: The description of the issue
            comments: Optional comments related to the issue
            filename: Optional filename related to the issue
            max_length: Maximum number of tokens to keep
            
        Returns:
            Preprocessed text ready for embedding
        """
        # Combine all text fields
        title = title if title else ""
        description = description if description else ""
        
        # Handle comments - could be string, list, or None
        if isinstance(comments, list):
            comments = " ".join(comments)
        elif not comments:
            comments = ""
        
        # 1. Combine title, description and comments
        all_text = f"{title} {description} {comments}".strip()
        
        # 2. Convert to lowercase
        all_text = all_text.lower()
        
        # 3. Remove line breaks
        all_text = all_text.replace('\r', ' ')
        all_text = all_text.replace('\n', ' ')
        
        # 4. Remove non-alphanumeric characters
        all_text = re.sub(r'[^a-zA-Z0-9 ]', '', all_text)
        
        # 5. Filter words based on the pre-loaded vocabulary file
        if self.vocabulary_words:
            words = all_text.split()
            # Only keep words that exist in our vocabulary
            filtered_words = [word for word in words if word in self.vocabulary_words]
            all_text = ' '.join(filtered_words)

        # 6-7. Remove stopwords and Lemmatize text
        if self.nlp:
            doc = self.nlp(all_text)
            all_text = ' '.join([word.lemma_ for word in doc if not word.is_stop])

        # 8. Add filename to the end of description if provided
        if filename:
            all_text += " " + filename.lower()
        
        # 9. Truncate if necessary (SBERT has a limit, usually 512 tokens)
        words = all_text.split()
        if len(words) > max_length:
            all_text = " ".join(words[:max_length])
        
        return all_text
    
    def _get_embeddings(self, texts, batch_size=32):
        """Generate embeddings for a list of texts using the provided model"""
        if not texts:
            return np.array([])
        
        print("Generating embeddings...")
        
        # Use the model's encode method with batching for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def find_similar_requests(self, title, description, comments="", top_k=20):
        """
        Find similar requests to the given input
        
        Args:
            title: Title of the request
            description: Description or body of the request
            comments: Optional comments related to the request
            top_k: Number of similar items to retrieve
        
        Returns:
            List of dictionaries with similar items and their similarity scores
        """
        # Preprocess query text
        query_text = self._preprocess_text(title, description, comments)
        
        if not query_text:
            print("No query text provided.")
            return []
        
        print("Query text:", query_text[:100] + ("..." if len(query_text) > 100 else ""))
        
        # Generate query embedding
        print("Generating embedding for query text...")
        query_embedding = self._get_embeddings([query_text])[0]
        
        # Ensure query embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.reference_embeddings)[0]
        
        # Get indices of top-k similar items
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Create result list
        results = []
        for idx in top_indices:
            item_data = self.reference_data.iloc[idx].to_dict()
            
            # Format the result to match the expected output
            formatted_result = {
                'similarity': float(similarities[idx]) * 100,  # Convert to percentage
                'title': item_data.get('title', 'No title available'),
                'description': item_data.get('body', item_data.get('description', 'No description available')),
                'areas': []
            }
            
            # Extract labels if available
            labels = item_data.get('labels', [])
            if isinstance(labels, str):
                try:
                    # Try to parse JSON string
                    labels = json.loads(labels.replace("'", '"'))
                except:
                    # If it's a string but not valid JSON, try to convert it
                    if labels.startswith('[') and labels.endswith(']'):
                        labels = labels.strip('[]').split(',')
                        labels = [label.strip().strip("'\"") for label in labels]
                    else:
                        labels = [labels]
            
            # Convert labels to area format
            for label in labels:
                if isinstance(label, str):
                    formatted_result['areas'].append({
                        'name': label,
                        'color': self._get_color_for_label(label, 1.0),
                        'confidence': 100  # Default confidence for existing labels
                    })
            
            # Add issue URL if available
            if 'issue_url' in item_data:
                formatted_result['issue_url'] = item_data.get('issue_url', '')
            
            # Add date - use modified date if available, or current date as fallback
            if 'created_at' in item_data:
                formatted_result['date'] = item_data.get('created_at', '')
            else:
                formatted_result['date'] = datetime.now().strftime('%Y-%m-%d')
            
            # Add ID if available
            if 'id' in item_data:
                formatted_result['id'] = item_data.get('id', '')
            
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
    
    def switch_request_type(self, request_type):
        """Switch to a different request type and reload data"""
        if self.request_type == request_type:
            return
        
        print(f"Switching from {self.request_type} to {request_type}")
        self.request_type = request_type
        parent_dir = os.path.dirname(self.base_path)  # Get parent directory again
        self.data_path = os.path.join(
            parent_dir, 
            'data',
            request_type,
            f'{request_type}_preprocessed_train_data_with_embeddings.csv'
        )
        self._initialize()

# Test function
if __name__ == "__main__":
    # Example usage
    sim = SimilarityCalculator()
    results = sim.find_similar_requests(
        "Issue with network policy", 
        "The network policy doesn't apply correctly to pods with multiple labels",
        top_k=5
    )
    for i, result in enumerate(results):
        print(f"{i+1}. Similarity: {result['similarity']:.2f}%")
        print(f"   Title: {result['title']}")
        print(f"   Areas: {[a['name'] for a in result['areas']]}")
        print("")
