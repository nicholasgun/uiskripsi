from flask import Flask, render_template, request, jsonify
import json
import random
from datetime import datetime, timedelta
import os
import time
import csv
import pandas as pd
from dotenv import load_dotenv
from classifiers.area_classifier import AreaClassifier
from similarity.similarity_calculator import SimilarityCalculator

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize classifiers - one for bug and one for feature
try:
    print("Initializing area classifiers...")
    # Initialize the classifier with bug model as default (only loads bug models)
    area_classifier = AreaClassifier(request_type='bug')
    print("Classifier initialization complete")
except Exception as e:
    print(f"Error initializing classifier: {str(e)}")
    area_classifier = None

# Initialize similarity calculator with bug data as default
try:
    print("Initializing similarity calculator...")
    similarity_calculator = SimilarityCalculator(request_type='bug')
    print("Similarity calculator initialization complete")
except Exception as e:
    print(f"Error initializing similarity calculator: {str(e)}")
    similarity_calculator = None

# Mock data for demo - in real app, this would be in a database
demo_similar_requests = []
for i in range(30):
    demo_similar_requests.append({
        "similarity": (98 - (i % 10)),
        "title": f"Change to ingress rule #{i+1}",
        "description": f"Update ingress to allow traffic from new subnet. This is a longer description for demo purposes.",
        "areas": [
            {"name": "Networking", "color": "bg-blue-100 text-blue-800", "confidence": 90 - (i % 10)},
        ],
        "date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
        "id": i+1
    })
    # Add some variety to the areas
    if i % 3 == 0:
        demo_similar_requests[i]["areas"].append(
            {"name": "Storage", "color": "bg-green-100 text-green-800", "confidence": 60 + (i % 10)}
        )
    if i % 4 == 0:
        demo_similar_requests[i]["areas"].append(
            {"name": "Compute", "color": "bg-purple-100 text-purple-800", "confidence": 40 + (i % 10)}
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Extract data from form
    data = request.form
    title = data.get('title', '')
    description = data.get('description', '')
    comments = data.get('comments', '')
    filename = data.get('filename', '')  # Get filename if provided
    request_type = data.get('kind', 'bug')  # Get the request type (bug or feature)
    
    # Get confidence threshold from environment variable or use default
    confidence_threshold = float(os.getenv('MODEL_CONFIDENCE_THRESHOLD', 0.5))
    
    # Check if we have a classifier
    if not area_classifier:
        # Fallback to mock data if classifier initialization failed
        areas = [
            {"name": "Networking", "confidence": 92, "color": "bg-blue-500 text-white"},
            {"name": "Storage", "confidence": 67, "color": "bg-green-500 text-white"},
            {"name": "Compute", "confidence": 41, "color": "bg-purple-500 text-white"}
        ]
        reasoning = "Classifier not available. Using mock data."
        model_type_info = f"Using <b>{request_type}</b> model for classification (mock data)"
    else:
        # Use the real classifier
        start_time = time.time()
        try:
            # Switch to the appropriate model based on request type
            if request_type not in ['bug', 'feature']:
                request_type = 'bug'  # Default to bug if invalid type
                
            # Switch model (may involve dynamic loading)
            switch_result = {"success": True, "loaded_new": False, "message": ""}
            if area_classifier.active_model != request_type:
                switch_result = area_classifier.switch_model(request_type)
                
            # Check if model switch was successful
            if not switch_result["success"]:
                # Model loading failed, return error
                areas = []
                reasoning = f"Error switching to {request_type} model: {switch_result['message']}"
                model_type_info = f"Failed to load <b>{request_type}</b> model"
            else:
                # Get predictions using the appropriate model
                areas = area_classifier.predict(title, description, comments, filename, confidence_threshold=confidence_threshold)

                # Format confidence scores for display
                for area in areas:
                    area["confidence"] = int(area["confidence"] * 100)  # Convert from 0-1 to 0-100
                
                # Generate reasoning text for each area but NOT the model type
                reasoning = ""
                for area in areas: 
                    reasoning += f"This request is classified as <b>{area['name']}</b> ({area['confidence']}%) based on the model prediction.<br>"
                    
                reasoning += f"<br>Processing time: {time.time() - start_time:.2f} seconds."
                
                # Add loading information if a new model was loaded
                if switch_result["loaded_new"]:
                    reasoning += f"<br>Model loading time: {switch_result.get('load_time', 0):.2f} seconds."
                
                # Store the model type separately
                model_type_info = f"Using <b>{request_type}</b> model for classification"
                if switch_result["loaded_new"]:
                    model_type_info += " (newly loaded)"
        except Exception as e:
            areas = []
            reasoning = f"Error in classification: {str(e)}"
    
    # Create classification result
    classification_result = {
        "areas": areas,
        "components": [],  # This would need a separate classifier
        "reasoning": reasoning,
        "model_type": model_type_info,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M')
    }
    
    # Get similar requests based on input
    top_k = int(os.getenv('SIMILARITY_TOP_K', 20))
    similar_requests = []
    if similarity_calculator:
        try:
            # Switch to the appropriate model based on request type if needed
            if similarity_calculator.request_type != request_type:
                try:
                    similarity_calculator.switch_request_type(request_type)
                except Exception as e:
                    print(f"Error switching similarity model: {str(e)}")
                    # If we can't switch to the requested type, use what we have
            
            # Get similar requests
            similar_time_start = time.time()
            top_k = int(os.getenv('SIMILARITY_TOP_K', 20))
            similar_requests = similarity_calculator.find_similar_requests(title, description, comments, filename, top_k=top_k)
            
            reasoning += f"<br>Similarity calculation time: {time.time() - similar_time_start:.2f} seconds."
        except Exception as e:
            print(f"Error finding similar requests: {str(e)}")
            similar_requests = demo_similar_requests  # Fallback to demo data
    else:
        similar_requests = demo_similar_requests  # Use demo data if calculator not available
    
    return jsonify({
        "success": True,
        "classification": classification_result,
        "similar_requests": similar_requests
    })

@app.route('/similar_requests', methods=['GET'])
def get_similar_requests():
    # Get pagination parameters
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('page_size', 5))
    area_filter = request.args.get('area', '')
    
    # Filter by area if provided
    filtered_requests = demo_similar_requests
    if area_filter:
        filtered_requests = [req for req in demo_similar_requests 
                            if any(area["name"] == area_filter for area in req["areas"])]
    
    # Calculate pagination
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered_requests))
    
    return jsonify({
        "success": True,
        "requests": filtered_requests[start_idx:end_idx],
        "total": len(filtered_requests),
        "page": page,
        "page_size": page_size
    })

@app.route('/export_csv', methods=['GET'])
def export_csv():
    # In a real application, this would generate and return a CSV file
    return jsonify({
        "success": True,
        "message": "CSV export feature would be implemented here"
    })

@app.route('/random_sample', methods=['GET'])
def random_sample():
    try:
        # Ensure fresh randomness for each request
        import time
        import numpy as np
        # Reset numpy random state to ensure different samples each time
        np.random.seed(int(time.time() * 1000000) % 2**32)
        
        # Get request type (bug or feature) from the query parameters
        request_type = request.args.get('type', 'bug')
        # Only allow 'bug' or 'feature' as valid types
        if request_type not in ['bug', 'feature']:
            request_type = 'bug'  # Default to bug if invalid type
            
        # Path to testing data based on request type
        data_path = os.path.join(os.path.dirname(__file__), 'data', request_type, 'testing_data.csv')
        
        # Get filename parameter
        requested_filename = request.args.get('filename', '')
        
        if not os.path.exists(data_path):
            print(f"CSV file not found at: {data_path}")
            return jsonify({
                "success": False,
                "error": f"CSV file for {request_type} not found at: {data_path}"
            })
        
        try:
            # Load the full dataset for maximum variety in random sampling
            df = pd.read_csv(data_path)
        except ValueError as e:
            # If columns don't match, fall back to reading all columns
            print(f"Falling back to reading all columns: {str(e)}")
            df = pd.read_csv(data_path)
        
        if df.empty:
            return jsonify({
                "success": False,
                "error": "CSV file is empty or no matching records found"
            })
        
        # Select a random row with proper randomization
        # Use None as random_state to get truly random sampling
        # This ensures each call produces different results
        random_row = df.sample(1).iloc[0]
        
        # Extract title and body with fallbacks if columns don't exist
        title = random_row.get('title', 'No title available') if 'title' in random_row else 'No title available'
        description = random_row.get('body', 'No description available') if 'body' in random_row else 'No description available'
        
        # Get comments from the data if it exists
        comments = random_row.get('all_comments', '') if 'all_comments' in random_row else ''
        
        # Get filename from the data if it exists
        filename = random_row.get('filename', '') if 'filename' in random_row else ''
        
        # If filename is not in the dataset but was requested, use the requested filename
        if not filename and requested_filename:
            filename = requested_filename
        
        # Use the request_type as the kind
        kind = request_type
        
        return jsonify({
            "success": True,
            "sample": {
                "title": title,
                "description": description,
                "comments": comments,
                "filename": filename,
                "kind": kind
            }
        })
    except Exception as e:
        import traceback
        print(f"Error in random_sample: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/find_similar', methods=['POST'])
def find_similar():
    # Extract data from form
    data = request.form
    title = data.get('title', '')
    description = data.get('description', '')
    comments = data.get('comments', '')
    filename = data.get('filename', '')
    request_type = data.get('kind', 'bug')  # Get the request type (bug or feature)
    
    # Get top_k from form data or environment variable
    top_k_from_request = data.get('top_k', None)
    top_k = int(top_k_from_request) if top_k_from_request else int(os.getenv('SIMILARITY_TOP_K', 20))
    
    # Check if we have a similarity calculator
    if not similarity_calculator:
        return jsonify({
            "success": False,
            "error": "Similarity calculator not available",
            "similar_requests": demo_similar_requests  # Fallback to demo data
        })
    
    try:
        # Switch to the appropriate model based on request type if needed
        if similarity_calculator.request_type != request_type:
            try:
                similarity_calculator.switch_request_type(request_type)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Error switching to {request_type} model: {str(e)}",
                    "similar_requests": demo_similar_requests  # Fallback to demo data
                })
        
        # Get similar requests
        start_time = time.time()
        similar_requests = similarity_calculator.find_similar_requests(
            title, description, comments, filename, top_k=top_k
        )
        processing_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "similar_requests": similar_requests,
            "processing_time": f"{processing_time:.2f} seconds",
            "model_type": request_type
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "similar_requests": demo_similar_requests  # Fallback to demo data
        })

@app.route('/model_status')
def model_status():
    """Get the current model loading status."""
    if not area_classifier:
        return jsonify({"error": "Classifier not available"})
    
    try:
        model_info = area_classifier.get_model_info()
        loaded_models = list(model_info['loaded_models'].keys())
        
        return jsonify({
            "success": True,
            "active_model": model_info['active_model'],
            "loaded_models": loaded_models,
            "available_models": ["bug", "feature"]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/preload_model', methods=['POST'])
def preload_model():
    """Preload a model type if not already loaded."""
    if not area_classifier:
        return jsonify({"error": "Classifier not available"})
    
    data = request.get_json()
    model_type = data.get('model_type', 'feature')
    
    if model_type not in ['bug', 'feature']:
        return jsonify({"success": False, "error": "Invalid model type"})
    
    try:
        # Check if already loaded
        if model_type in area_classifier.models:
            return jsonify({
                "success": True, 
                "message": f"{model_type} model already loaded",
                "loaded_new": False
            })
        
        # Load the model
        print(f"Preloading {model_type} model...")
        start_time = time.time()
        area_classifier._load_model_data(model_type)
        load_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "message": f"{model_type} model loaded successfully",
            "loaded_new": True,
            "load_time": load_time
        })
        
    except Exception as e:
        error_msg = f"Error loading {model_type} model: {str(e)}"
        print(error_msg)
        return jsonify({"success": False, "error": error_msg})

if __name__ == '__main__':
    app.run(debug=True)
