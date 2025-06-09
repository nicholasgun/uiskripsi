# Kubernetes Change Request Classifier

A web application that automatically classifies Kubernetes change requests using machine learning. Submit bug reports or feature requests and get instant categorization into Kubernetes subsystem areas.

## Features

- Classify bug reports and feature requests automatically
- Find similar existing requests
- View classification confidence scores
- Modern responsive web interface
- Template system for quick form filling

## Quick Start

### Prerequisites

- Python 3.8+
- 8GB+ RAM (for loading ML models)

### Installation

1. **Clone and setup**

```bash
git clone https://github.com/YOUR_USERNAME/kubernetes-change-request-classifier.git
cd kubernetes-change-request-classifier
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. **Run the application**

```bash
python app.py
```

4. **Open your browser** to `http://127.0.0.1:5000`

## Usage

1. Select request type (Bug or Feature Request)
2. Enter title and description
3. Click Submit to get classification results
4. Browse similar requests for additional context

## Project Structure

```
├── app.py                    # Main Flask application
├── classifiers/              # ML models and classification logic
├── similarity/               # Similarity search functionality
├── static/js/main.js         # Frontend JavaScript
├── templates/index.html      # Web interface
└── requirements.txt          # Python dependencies
## Technical Details

- **Models**: DeBERTa-based neural networks for text classification
- **Languages**: English text processing with spaCy
- **Similarity**: Sentence-BERT for finding similar requests
- **Frontend**: Vanilla JavaScript with Tailwind CSS
- **Backend**: Flask web framework

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

Nicholas Gunawan - c14210099@john.petra.ac.id
```
