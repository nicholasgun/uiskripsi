# Kubernetes Change Request Classifier

A comprehensive web application that automatically classifies Kubernetes change requests using advanced machine learning. Submit bug reports or feature requests and get instant categorization into Kubernetes subsystem areas with intelligent similarity matching.

## ✨ Features

### Core Functionality

- **Dual-Type Classification**: Separate optimized models for bug reports and feature requests
- **Smart Similarity Search**: Find similar existing requests using 4 specialized SBERT models
- **Filename-Aware Processing**: Enhanced accuracy when filenames are provided
- **Real-time Processing**: Instant classification and similarity results
- **Confidence Scoring**: Detailed confidence scores for all predictions

### Advanced Features

- **Pre-computed Embeddings**: Lightning-fast similarity search using cached embeddings
- **Memory-Efficient Loading**: On-demand model loading (2 models at a time)
- **Background Processing**: Non-blocking model switching and loading

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 12GB+ RAM (recommended for optimal performance)
- CUDA-compatible GPU (optional, for faster processing)

### Installation

1. **Clone and setup**

```bash
git clone https://github.com/nicholasgun/uiskripsi.git
cd uiskripsi
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Prepare embeddings (optional for similarity search)**

```bash
# Generate pre-computed embeddings for faster similarity search
python "similarity/generate data embeddings.py" --request_type bug
python "similarity/generate data embeddings.py" --request_type feature
```

4. **Run the application**

```bash
python app.py
```

5. **Open your browser** to `http://127.0.0.1:5000`

## 📖 Usage

### Basic Classification

1. Select request type (Bug or Feature Request)
2. Enter title and description
3. Add comments and filename (optional, but improves accuracy)
4. Click "Submit" to get classification results and similar requests

### Advanced Features

- **Generate Sample**: Load random samples from the Kubernetes testing dataset
- **Model Switching**: Automatically switches to appropriate model type
- **Similarity Search**: Browse similar requests with confidence scores

## 📁 Project Structure

```
├── app.py                           # Main Flask application with enhanced routing
├── classifiers/                     # ML models and classification logic
│   ├── area_classifier.py          # Multi-model classifier with dynamic loading
│   ├── bug/                         # Bug classification models
│   │   ├── with filename/           # DeBERTa model for bugs with filenames
│   │   └── without filename/        # DeBERTa-CNN model for bugs without filenames
│   └── feature/                     # Feature classification models
│       ├── with filename/           # DeBERTa model for features with filenames
│       └── without filename/        # DeBERTa-CNN model for features without filenames
├── similarity/                      # Advanced similarity search system
│   ├── similarity_calculator.py    # 4-model SBERT similarity engine
│   ├── generate data embeddings.py # Embedding generation script
│   └── model/                       # SBERT models for similarity
│       ├── bug/                     # Bug similarity models
│       │   ├── with filename/       # SBERT model for bugs with filenames
│       │   └── without filename/    # SBERT model for bugs without filenames
│       └── feature/                 # Feature similarity models
│           ├── with filename/       # SBERT model for features with filenames
│           └── without filename/    # SBERT model for features without filenames
├── data/                           # Training and testing datasets
│   ├── bug/                        # Bug-related datasets
│   │   ├── reference data_with_embeddings.csv  # Pre-computed embeddings
│   │   └── testing_data.csv        # Test samples for random generation
│   └── feature/                    # Feature-related datasets
│       ├── reference data_with_embeddings.csv  # Pre-computed embeddings
│       └── testing_data.csv        # Test samples for random generation
├── static/js/main.js               # Enhanced frontend JavaScript
├── templates/index.html            # Modern responsive web interface
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🔧 Technical Architecture

### Classification System

- **Multi-Model Architecture**: 4 specialized classification models
  - Bug with filename (DeBERTa): Optimized for bug reports with file context
  - Bug without filename (DeBERTa-CNN): General bug classification
  - Feature with filename (DeBERTa): Optimized for feature requests with file context
  - Feature without filename (DeBERTa-CNN): General feature classification
- **Dynamic Loading**: Models loaded on-demand to optimize memory usage
- **Preprocessing Pipeline**: Consistent text preprocessing across all models

### Similarity Search System

- **4-Model SBERT Architecture**: Specialized models for each classification variant
- **Pre-computed Embeddings**: Lightning-fast similarity search using cached embeddings
- **Cosine Similarity**: High-quality semantic matching between requests
- **Integrated Pipeline**: Uses same preprocessing as classification for consistency

### Performance Optimizations

- **Memory Management**: Only 2 models loaded simultaneously (classification + similarity)
- **Batch Processing**: Efficient embedding generation for large datasets
- **GPU Acceleration**: CUDA support for faster model inference
- **Caching**: Pre-computed embeddings eliminate runtime computation

## 🛠️ Configuration

### Environment Variables

```bash
MODEL_CONFIDENCE_THRESHOLD=0.5    # Minimum confidence for classification results
SIMILARITY_TOP_K=20               # Number of similar requests to return
```

### Model Files Required

- Classification models: DeBERTa and DeBERTa-CNN variants
- SBERT models: Fine-tuned sentence transformers
- Label encoders: For consistent label mapping
- Vocabulary files: For text preprocessing
- Pre-computed embeddings: For fast similarity search

## 🧪 Testing

### Generate Sample Data

```bash
# Test with random bug samples
curl "http://localhost:5000/random_sample?type=bug"

# Test with random feature samples
curl "http://localhost:5000/random_sample?type=feature"
```

### API Endpoints

- `POST /classify` - Main classification endpoint
- `POST /find_similar` - Dedicated similarity search
- `GET /random_sample` - Generate random test samples
- `GET /model_status` - Check model loading status
- `POST /preload_model` - Preload specific model types

## 👨‍💻 Author

**Nicholas Gunawan**

- Email: c14210099@john.petra.ac.id
- GitHub: [@nicholasgun](https://github.com/nicholasgun)

## 🙏 Acknowledgments

- Kubernetes github issues for providing the training dataset
- Hugging Face for pre-trained models and transformers library
- Sentence Transformers library for semantic similarity capabilities
- Flask and Tailwind CSS communities for excellent documentation

---
