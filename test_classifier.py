#!/usr/bin/env python3
"""
Test script for the updated AreaClassifier with DeBERTa and DeBERTa-CNN models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classifiers.area_classifier import AreaClassifier

def test_classifier():
    """Test the updated area classifier with different scenarios."""
    
    print("=" * 80)
    print("TESTING UPDATED AREA CLASSIFIER")
    print("=" * 80)
    
    try:
        # Initialize classifier (now only loads bug models initially)
        print("\n1. Initializing AreaClassifier with bug models only...")
        classifier = AreaClassifier(request_type='bug')
        
        # Get model info
        print("\n2. Model Information:")
        model_info = classifier.get_model_info()
        print(f"Active model: {model_info['active_model']}")
        
        for model_type, variants in model_info['loaded_models'].items():
            print(f"\n{model_type.upper()} Model:")
            for variant_name, variant_info in variants.items():
                print(f"  {variant_name}:")
                print(f"    - Model class: {variant_info['model_class']}")
                print(f"    - Number of labels: {variant_info['num_labels']}")
                print(f"    - Vocabulary size: {variant_info['vocab_size']}")
        
        # Test scenarios
        test_cases = [
            {
                "name": "Bug with filename (should use DeBERTa)",
                "title": "API authentication fails",
                "description": "The authentication service returns 401 error when making API calls",
                "comments": "This affects all users",
                "filename": "auth_service.py",
                "request_type": "bug"
            },
            {
                "name": "Bug without filename (should use DeBERTa-CNN)",
                "title": "Memory leak in pod",
                "description": "Kubernetes pod is consuming increasing amounts of memory over time",
                "comments": "Happens after 24 hours of runtime",
                "filename": "",
                "request_type": "bug"
            },
            {
                "name": "Feature with filename (should use DeBERTa)",
                "title": "Add new dashboard widget",
                "description": "Create a new widget for displaying real-time metrics",
                "comments": "Should support both line and bar charts",
                "filename": "dashboard.js",
                "request_type": "feature"
            },
            {
                "name": "Feature without filename (should use DeBERTa-CNN)",
                "title": "Implement auto-scaling",
                "description": "Add horizontal pod autoscaler to handle traffic spikes",
                "comments": "Should scale based on CPU and memory usage",
                "filename": "",
                "request_type": "feature"
            }
        ]
        
        print(f"\n3. Testing {len(test_cases)} scenarios:")
        print("=" * 80)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print("-" * 60)
            
            # Switch model if needed (this will trigger lazy loading for feature)
            if classifier.active_model != test_case['request_type']:
                print(f"Switching to {test_case['request_type']} model...")
                switch_result = classifier.switch_model(test_case['request_type'])
                if switch_result['loaded_new']:
                    print(f"  -> Loaded new model in {switch_result.get('load_time', 0):.2f} seconds")
            
            # Make prediction
            try:
                results = classifier.predict(
                    title=test_case['title'],
                    description=test_case['description'],
                    comments=test_case['comments'],
                    filename=test_case['filename'],
                    confidence_threshold=0.3  # Lower threshold for testing
                )
                
                print(f"Input: {test_case['title']}")
                print(f"Filename: '{test_case['filename']}'")
                print(f"Results ({len(results)} predictions):")
                
                if results:
                    for j, result in enumerate(results[:5], 1):  # Show top 5
                        print(f"  {j}. {result['name']}: {result['confidence']:.3f}")
                else:
                    print("  No predictions above threshold")
                    
            except Exception as e:
                print(f"Error in prediction: {e}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_classifier()
