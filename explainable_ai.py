import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
# Assuming TriageModel is defined in model_architecture.py
from model_architecture import TriageModel

# Import SHAP and permutation importance libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

try:
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score
    PERMUTATION_AVAILABLE = True
except ImportError:
    PERMUTATION_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")

# Import for LLM API integration
try:
    import requests
    import os
    import json
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available. Install with: pip install openai")

class OpenRouterClient:
    """Enhanced OpenRouter API client with multiple model support and fallback options"""
    
    # Available models with their capabilities and costs
    AVAILABLE_MODELS = {
        "anthropic/claude-3-sonnet": {
            "name": "Claude 3 Sonnet",
            "context_length": 200000,
            "cost_per_1k_tokens": 0.003,
            "strengths": ["reasoning", "analysis", "medical knowledge"],
            "recommended_for": "complex medical explanations"
        },
        "anthropic/claude-3-haiku": {
            "name": "Claude 3 Haiku",
            "context_length": 200000,
            "cost_per_1k_tokens": 0.00025,
            "strengths": ["speed", "efficiency"],
            "recommended_for": "quick explanations"
        },
        "openai/gpt-4-turbo": {
            "name": "GPT-4 Turbo",
            "context_length": 128000,
            "cost_per_1k_tokens": 0.01,
            "strengths": ["general knowledge", "reasoning"],
            "recommended_for": "comprehensive explanations"
        },
        "openai/gpt-3.5-turbo": {
            "name": "GPT-3.5 Turbo",
            "context_length": 16385,
            "cost_per_1k_tokens": 0.0005,
            "strengths": ["speed", "cost-effectiveness"],
            "recommended_for": "basic explanations"
        },
        "meta-llama/llama-3.1-70b-instruct": {
            "name": "Llama 3.1 70B",
            "context_length": 131072,
            "cost_per_1k_tokens": 0.0009,
            "strengths": ["open source", "reasoning"],
            "recommended_for": "privacy-conscious deployments"
        },
        "google/gemini-pro": {
            "name": "Gemini Pro",
            "context_length": 32768,
            "cost_per_1k_tokens": 0.0005,
            "strengths": ["multimodal", "reasoning"],
            "recommended_for": "diverse input types"
        }
    }
    
    def __init__(self, api_key=None, model="anthropic/claude-3-sonnet", fallback_models=None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.fallback_models = fallback_models or [
            "anthropic/claude-3-haiku",
            "openai/gpt-3.5-turbo",
            "meta-llama/llama-3.1-70b-instruct"
        ]
        self.base_url = "https://openrouter.ai/api/v1"
        self.request_count = 0
        self.total_cost = 0.0
        
    def get_model_info(self, model_name=None):
        """Get information about a specific model or all available models"""
        if model_name:
            return self.AVAILABLE_MODELS.get(model_name, {})
        return self.AVAILABLE_MODELS
    
    def select_optimal_model(self, prompt_length, complexity="medium", budget_conscious=False):
        """Select the optimal model based on prompt characteristics and requirements"""
        if budget_conscious:
            # Sort by cost, prefer cheaper models
            sorted_models = sorted(
                self.AVAILABLE_MODELS.items(),
                key=lambda x: x[1]["cost_per_1k_tokens"]
            )
        else:
            # Default ordering by capability for medical use
            model_priority = [
                "anthropic/claude-3-sonnet",
                "openai/gpt-4-turbo",
                "anthropic/claude-3-haiku",
                "meta-llama/llama-3.1-70b-instruct",
                "google/gemini-pro",
                "openai/gpt-3.5-turbo"
            ]
            sorted_models = [(m, self.AVAILABLE_MODELS[m]) for m in model_priority if m in self.AVAILABLE_MODELS]
        
        # Filter by context length requirements
        suitable_models = [
            (model, info) for model, info in sorted_models
            if info["context_length"] >= prompt_length
        ]
        
        if suitable_models:
            selected_model = suitable_models[0][0]
            print(f"Selected model: {self.AVAILABLE_MODELS[selected_model]['name']} for {complexity} complexity task")
            return selected_model
        else:
            print(f"Warning: No model found with sufficient context length ({prompt_length}). Using default.")
            return self.model
    
    def generate_explanation(self, prompt, max_tokens=200, temperature=0.7, model_override=None, auto_select=True):
        """Generate explanation using OpenRouter API with enhanced features"""
        if not self.api_key:
            raise ValueError("OpenRouter API key not provided. Set OPENROUTER_API_KEY environment variable.")
        
        # Auto-select model if enabled
        if auto_select and not model_override:
            prompt_length = len(prompt.split())
            complexity = "high" if prompt_length > 500 else "medium" if prompt_length > 200 else "low"
            selected_model = self.select_optimal_model(prompt_length, complexity)
        else:
            selected_model = model_override or self.model
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/triaj-research",
            "X-Title": "FairTriEdge-FL Medical Triage System"
        }
        
        # Enhanced system prompt for medical context
        system_prompt = """You are an expert medical AI assistant specializing in emergency triage explanations.
        Your role is to explain AI-driven triage decisions in a way that is:
        1. Medically accurate and evidence-based
        2. Clear and understandable to both healthcare professionals and patients
        3. Empathetic and reassuring while maintaining clinical objectivity
        4. Transparent about AI limitations and the need for human medical judgment
        5. Culturally sensitive and appropriate for diverse patient populations
        
        Always emphasize that AI triage is a decision support tool and final medical decisions should involve qualified healthcare professionals."""
        
        data = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
        
        # Try primary model, then fallbacks
        models_to_try = [selected_model] + [m for m in self.fallback_models if m != selected_model]
        
        for attempt, model_name in enumerate(models_to_try):
            try:
                data["model"] = model_name
                print(f"Attempting explanation generation with {self.AVAILABLE_MODELS.get(model_name, {}).get('name', model_name)}...")
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60  # Increased timeout for complex requests
                )
                response.raise_for_status()
                result = response.json()
                
                # Track usage and costs
                self.request_count += 1
                if "usage" in result:
                    tokens_used = result["usage"].get("total_tokens", 0)
                    model_cost = self.AVAILABLE_MODELS.get(model_name, {}).get("cost_per_1k_tokens", 0)
                    request_cost = (tokens_used / 1000) * model_cost
                    self.total_cost += request_cost
                    print(f"Request completed. Tokens used: {tokens_used}, Cost: ${request_cost:.4f}")
                
                explanation = result["choices"][0]["message"]["content"]
                
                # Add model attribution
                model_info = self.AVAILABLE_MODELS.get(model_name, {})
                attribution = f"\n\n[Generated by {model_info.get('name', model_name)}]"
                
                return explanation + attribution
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limit
                    print(f"Rate limit hit for {model_name}, trying next model...")
                    continue
                elif e.response.status_code == 402:  # Insufficient credits
                    print(f"Insufficient credits for {model_name}, trying next model...")
                    continue
                else:
                    print(f"HTTP error {e.response.status_code} for {model_name}: {e}")
                    if attempt == len(models_to_try) - 1:  # Last attempt
                        raise Exception(f"All models failed. Last error: {e}")
                    continue
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                if attempt == len(models_to_try) - 1:  # Last attempt
                    raise Exception(f"All models failed. Last error: {e}")
                continue
        
        raise Exception("All available models failed to generate explanation")
    
    def get_usage_stats(self):
        """Get usage statistics"""
        return {
            "total_requests": self.request_count,
            "total_cost": self.total_cost,
            "average_cost_per_request": self.total_cost / max(1, self.request_count)
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics"""
        self.request_count = 0
        self.total_cost = 0.0

# Placeholder for SHAP or LIME integration
# In a real scenario, you would install and use the shap or lime library.
# For this demonstration, we'll simulate SHAP-like output.
def generate_feature_importance(model, data_point, feature_names, method="random", background_data=None, test_data=None, test_labels=None):
    """
    Generates feature importance scores using the specified method.
    Args:
        model (nn.Module): The trained PyTorch model.
        data_point (torch.Tensor): A single input data point (concatenated numerical, boolean, temporal).
        feature_names (list): List of names for all features.
        method (str): The feature importance method to use ('random', 'shap', 'permutation').
        background_data (torch.Tensor, optional): Background data for SHAP explainer.
        test_data (torch.Tensor, optional): Test data for permutation importance.
        test_labels (torch.Tensor, optional): Test labels for permutation importance.
    Returns:
        dict: A dictionary mapping feature names to their importance scores.
    """
    if method == "shap" and SHAP_AVAILABLE:
        print("Generating feature importance (method: SHAP).")
        try:
            # Create a wrapper function for the model that works with SHAP
            def model_wrapper(x):
                if isinstance(x, np.ndarray):
                    x = torch.tensor(x, dtype=torch.float32)
                with torch.no_grad():
                    # Split the input based on the model's expected structure
                    num_numerical = 7  # From the model architecture
                    num_boolean = 268
                    num_temporal = 3
                    
                    if x.dim() == 1:
                        x = x.unsqueeze(0)
                    
                    numerical_data = x[:, :num_numerical]
                    boolean_data = x[:, num_numerical:num_numerical + num_boolean]
                    temporal_data = x[:, num_numerical + num_boolean:]
                    
                    logits = model(numerical_data, boolean_data, temporal_data)
                    return torch.softmax(logits, dim=1).numpy()
            
            # Use background data if provided, otherwise create a simple background
            if background_data is not None:
                background = background_data.numpy() if isinstance(background_data, torch.Tensor) else background_data
            else:
                # Create a simple background with zeros
                background = np.zeros((10, len(feature_names)))
            
            # Initialize SHAP explainer
            explainer = shap.Explainer(model_wrapper, background)
            
            # Calculate SHAP values
            data_numpy = data_point.numpy() if isinstance(data_point, torch.Tensor) else data_point
            if data_numpy.ndim == 1:
                data_numpy = data_numpy.reshape(1, -1)
            
            shap_values = explainer(data_numpy)
            
            # Get importance scores (mean absolute SHAP values across classes)
            if hasattr(shap_values, 'values'):
                if shap_values.values.ndim == 3:  # Multi-class case
                    importance_scores = np.abs(shap_values.values).mean(axis=(0, 2))
                else:
                    importance_scores = np.abs(shap_values.values).mean(axis=0)
            else:
                importance_scores = np.abs(shap_values).mean(axis=0)
                
        except Exception as e:
            print(f"Error in SHAP calculation: {e}. Falling back to random values.")
            importance_scores = np.random.rand(len(feature_names))
            
    elif method == "permutation" and PERMUTATION_AVAILABLE:
        print("Generating feature importance (method: Permutation Importance).")
        try:
            if test_data is None or test_labels is None:
                print("Warning: test_data and test_labels required for permutation importance. Using random values.")
                importance_scores = np.random.rand(len(feature_names))
            else:
                # Create a sklearn-compatible wrapper
                class SklearnModelWrapper:
                    def __init__(self, pytorch_model):
                        self.model = pytorch_model
                        
                    def predict(self, X):
                        if isinstance(X, np.ndarray):
                            X = torch.tensor(X, dtype=torch.float32)
                        
                        with torch.no_grad():
                            num_numerical = 7
                            num_boolean = 268
                            num_temporal = 3
                            
                            if X.dim() == 1:
                                X = X.unsqueeze(0)
                            
                            numerical_data = X[:, :num_numerical]
                            boolean_data = X[:, num_numerical:num_numerical + num_boolean]
                            temporal_data = X[:, num_numerical + num_boolean:]
                            
                            logits = self.model(numerical_data, boolean_data, temporal_data)
                            return torch.argmax(logits, dim=1).numpy()
                
                # Wrap the model
                wrapped_model = SklearnModelWrapper(model)
                
                # Convert test data to numpy if needed
                X_test = test_data.numpy() if isinstance(test_data, torch.Tensor) else test_data
                y_test = test_labels.numpy() if isinstance(test_labels, torch.Tensor) else test_labels
                
                # Calculate permutation importance
                result = permutation_importance(
                    wrapped_model, X_test, y_test,
                    n_repeats=10, random_state=42,
                    scoring='accuracy'
                )
                importance_scores = result.importances_mean
                
        except Exception as e:
            print(f"Error in permutation importance calculation: {e}. Falling back to random values.")
            importance_scores = np.random.rand(len(feature_names))
            
    elif method == "random":
        print("Generating feature importance (method: Random values - placeholder).")
        importance_scores = np.random.rand(len(feature_names))
    else:
        if method == "shap" and not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap. Using random values.")
        elif method == "permutation" and not PERMUTATION_AVAILABLE:
            print("sklearn not available. Install with: pip install scikit-learn. Using random values.")
        else:
            print(f"Unknown feature importance method: {method}. Returning random values.")
        importance_scores = np.random.rand(len(feature_names))
        
    # Normalize importance scores
    if np.sum(np.abs(importance_scores)) > 0:
        importance_scores = importance_scores / np.sum(np.abs(importance_scores))
    
    feature_importance = dict(zip(feature_names, importance_scores))
    
    # Sort by absolute importance for better readability
    sorted_importance = sorted(feature_importance.items(), key=lambda item: abs(item[1]), reverse=True)
    return sorted_importance

# Placeholder for LLM-Enhanced Explanation Module
# In a real scenario, this would interact with an LLM API or a local LLM.
def generate_llm_explanation(prediction, top_features, patient_context, method="placeholder", api_key=None):
    """
    Generates natural language explanations using an LLM based on the specified method.
    Args:
        prediction (int): The predicted triage level (0, 1, or 2).
        top_features (list): A list of (feature_name, importance_score) tuples for key features.
        patient_context (dict): A dictionary of relevant patient information (e.g., age, symptoms).
        method (str): The LLM explanation method to use ('placeholder', 'openrouter', 'openai_gpt').
        api_key (str, optional): API key for the chosen method.
    Returns:
        str: A natural language explanation for the triage decision.
    """
    triage_levels = {
        0: "Yeşil Alan (Green Area - Non-urgent)",
        1: "Sarı Alan (Yellow Area - Urgent)",
        2: "Kırmızı Alan (Red Area - Emergency)"
    }
    predicted_level_text = triage_levels.get(prediction, "Unknown")
    
    explanation = f"The AI system triaged the patient to: {predicted_level_text}.\n"
    explanation += "Key factors influencing this decision include:\n"
    
    for feature, score in top_features[:5]: # Show top 5 features
        explanation += f"- {feature} (Importance: {score:.3f})\n"
    
    if method == "openrouter" and REQUESTS_AVAILABLE:
        print("Generating LLM explanation (method: OpenRouter API).")
        try:
            client = OpenRouterClient(api_key=api_key)
            
            # Construct detailed prompt
            prompt = f"""Based on the following patient context and AI prediction, explain the triage decision in a clear, empathetic manner:

Patient Context: {patient_context}
AI Predicted Triage Level: {predicted_level_text}

Key influencing factors:
"""
            for feature, score in top_features[:5]:
                prompt += f"- {feature} (Importance: {score:.3f})\n"
            
            prompt += """
Please provide a concise, clear, and empathetic explanation that:
1. Explains why this triage level was chosen
2. Relates the key factors to medical reasoning
3. Is understandable to both medical professionals and patients
4. Maintains appropriate medical caution and recommendations for professional consultation
"""
            
            llm_explanation = client.generate_explanation(prompt)
            explanation += f"\nLLM Explanation: {llm_explanation}"
            
        except Exception as e:
            explanation += f"\nError generating OpenRouter explanation: {e}"
            
    elif method == "openai_gpt" and OPENAI_AVAILABLE:
        print("Generating LLM explanation (method: OpenAI GPT-4 API).")
        try:
            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
            
            prompt = f"""Based on the following patient context and AI prediction, explain the triage decision:

Patient Context: {patient_context}
AI Predicted Triage Level: {predicted_level_text}

Key influencing factors:
"""
            for feature, score in top_features[:5]:
                prompt += f"- {feature} (Importance: {score:.3f})\n"
            
            prompt += "\nPlease provide a concise, clear, and empathetic explanation."
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that explains medical triage decisions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            llm_explanation = response.choices[0].message.content
            explanation += f"\nLLM Explanation: {llm_explanation}"
            
        except Exception as e:
            explanation += f"\nError generating OpenAI explanation: {e}"
            
    elif method == "placeholder":
        explanation += "\n(This explanation is a placeholder and would be generated by a sophisticated LLM based on medical knowledge and the identified features.)"
    else:
        if method == "openrouter" and not REQUESTS_AVAILABLE:
            print("requests library not available. Install with: pip install requests")
        elif method == "openai_gpt" and not OPENAI_AVAILABLE:
            print("OpenAI library not available. Install with: pip install openai")
        else:
            print(f"Unknown LLM explanation method: {method}. Using placeholder explanation.")
        explanation += "\n(This explanation is a placeholder and would be generated by a sophisticated LLM based on medical knowledge and the identified features.)"
    
    return explanation

# Placeholder for Real-Time Interpretability of Boolean Rule Chains
def extract_boolean_rules(model, feature_names, num_rules=3):
    """
    Placeholder for extracting simplified Boolean rule chains.
    In a real scenario, this would involve rule-mining algorithms.
    """
    print("Extracting Boolean Rule Chains (placeholder: dummy rules).")
    
    dummy_rules = [
        f"IF (feature_goguste_baski_hissi == 1) AND (feature_solunum_sayisi > 0.8) THEN Triage = Kırmızı Alan",
        f"IF (feature_ates > 0.5) AND (feature_ishal == 1) THEN Triage = Sarı Alan",
        f"IF (feature_bas_agrisi == 1) AND (feature_yorgunluk == 0) THEN Triage = Yeşil Alan"
    ]
    
    return dummy_rules
class LLMExplanationEngine:
    """Enhanced LLM explanation engine with multiple providers and fallback support"""
    
    def __init__(self, openrouter_key=None, openai_key=None, preferred_provider="openrouter"):
        self.openrouter_key = openrouter_key or os.getenv("OPENROUTER_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self.preferred_provider = preferred_provider
        
        # Initialize clients
        self.openrouter_client = None
        self.openai_client = None
        
        if self.openrouter_key and REQUESTS_AVAILABLE:
            self.openrouter_client = OpenRouterClient(api_key=self.openrouter_key)
        
        if self.openai_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_key)
    
    def generate_explanation(self, prediction, top_features, patient_context, 
                           complexity="medium", max_tokens=300, use_fallback=True):
        """
        Generate comprehensive medical explanation with automatic fallback
        """
        triage_levels = {
            0: "Yeşil Alan (Green Area - Non-urgent)",
            1: "Sarı Alan (Yellow Area - Urgent)", 
            2: "Kırmızı Alan (Red Area - Emergency)"
        }
        predicted_level_text = triage_levels.get(prediction, "Unknown")
        
        # Build comprehensive prompt
        prompt = self._build_medical_prompt(predicted_level_text, top_features, patient_context, complexity)
        
        # Try preferred provider first, then fallback
        providers_to_try = []
        if self.preferred_provider == "openrouter" and self.openrouter_client:
            providers_to_try.append(("openrouter", self.openrouter_client))
        if self.openai_client:
            providers_to_try.append(("openai", self.openai_client))
        if self.preferred_provider == "openrouter" and self.openrouter_client:
            # Add openrouter again if it wasn't first
            if providers_to_try[0][0] != "openrouter":
                providers_to_try.append(("openrouter", self.openrouter_client))
        
        explanation = f"AI Triage Decision: {predicted_level_text}\n\n"
        explanation += "Key Contributing Factors:\n"
        for i, (feature, score) in enumerate(top_features[:5], 1):
            explanation += f"{i}. {feature} (Importance: {score:.3f})\n"
        explanation += "\n"
        
        # Try each provider
        for provider_name, client in providers_to_try:
            try:
                print(f"Generating explanation using {provider_name}...")
                
                if provider_name == "openrouter":
                    llm_response = client.generate_explanation(
                        prompt, 
                        max_tokens=max_tokens,
                        auto_select=True
                    )
                elif provider_name == "openai":
                    llm_response = self._generate_openai_explanation(
                        client, prompt, max_tokens
                    )
                
                explanation += f"Medical Analysis:\n{llm_response}"
                return explanation
                
            except Exception as e:
                print(f"Error with {provider_name}: {e}")
                if not use_fallback or provider_name == providers_to_try[-1][0]:
                    # Last provider or fallback disabled
                    explanation += f"\nNote: Unable to generate detailed explanation ({e}). "
                    explanation += "Please consult with medical professionals for comprehensive assessment."
                    return explanation
                else:
                    print(f"Trying fallback provider...")
                    continue
        
        # If all providers fail
        explanation += self._generate_fallback_explanation(predicted_level_text, top_features, patient_context)
        return explanation
    
    def _build_medical_prompt(self, predicted_level, top_features, patient_context, complexity):
        """Build a comprehensive medical prompt"""
        prompt = f"""You are providing a medical explanation for an AI triage decision. 

PATIENT CONTEXT:
{self._format_patient_context(patient_context)}

AI TRIAGE DECISION: {predicted_level}

KEY FACTORS IDENTIFIED BY AI:
"""
        for i, (feature, score) in enumerate(top_features[:5], 1):
            prompt += f"{i}. {feature} (Importance Score: {score:.3f})\n"
        
        if complexity == "high":
            prompt += """
Please provide a comprehensive medical explanation that includes:
1. Clinical reasoning for the triage level assignment
2. How each key factor contributes to the decision
3. Potential differential diagnoses to consider
4. Recommended immediate actions and monitoring
5. When to escalate or reassess the patient
6. Patient/family communication points
7. Documentation requirements

Maintain clinical accuracy while being accessible to both healthcare providers and patients.
"""
        elif complexity == "medium":
            prompt += """
Please provide a clear medical explanation that includes:
1. Why this triage level was assigned
2. How the key factors support this decision
3. Immediate care priorities
4. When to reassess or escalate
5. Key points for patient communication

Balance clinical detail with accessibility.
"""
        else:  # low complexity
            prompt += """
Please provide a concise explanation that:
1. Explains the triage decision in simple terms
2. Highlights the most important factors
3. Gives clear next steps
4. Reassures while maintaining clinical accuracy
"""
        
        prompt += "\nRemember: AI triage is a decision support tool. Final medical decisions require qualified healthcare professional judgment."
        
        return prompt
    
    def _format_patient_context(self, context):
        """Format patient context for the prompt"""
        if isinstance(context, dict):
            formatted = []
            for key, value in context.items():
                formatted.append(f"- {key.replace('_', ' ').title()}: {value}")
            return "\n".join(formatted)
        return str(context)
    
    def _generate_openai_explanation(self, client, prompt, max_tokens):
        """Generate explanation using OpenAI API"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system", 
                    "content": "You are an expert emergency medicine physician providing clear, empathetic explanations of AI-assisted triage decisions."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )
        return response.choices[0].message.content
    
    def _generate_fallback_explanation(self, predicted_level, top_features, patient_context):
        """Generate a structured fallback explanation when APIs are unavailable"""
        explanation = "Medical Assessment Summary:\n\n"
        
        # Basic triage reasoning
        if "Emergency" in predicted_level:
            explanation += "This patient requires immediate medical attention based on critical indicators. "
            explanation += "The AI identified high-priority factors that suggest potential life-threatening conditions.\n\n"
        elif "Urgent" in predicted_level:
            explanation += "This patient should be seen promptly by medical staff. "
            explanation += "The AI detected concerning factors that require timely evaluation and intervention.\n\n"
        else:
            explanation += "This patient can be managed with standard priority. "
            explanation += "The AI assessment suggests stable condition with routine care needs.\n\n"
        
        # Feature interpretation
        explanation += "Key Clinical Indicators:\n"
        for feature, score in top_features[:3]:
            if "vital" in feature.lower() or "bp" in feature.lower() or "hr" in feature.lower():
                explanation += f"• Vital signs abnormality detected ({feature})\n"
            elif "pain" in feature.lower() or "ache" in feature.lower():
                explanation += f"• Pain-related symptom identified ({feature})\n"
            elif "age" in feature.lower():
                explanation += f"• Age-related risk factor considered ({feature})\n"
            else:
                explanation += f"• Clinical factor: {feature}\n"
        
        explanation += "\nRecommendations:\n"
        explanation += "• Verify AI assessment with clinical judgment\n"
        explanation += "• Monitor for changes in patient condition\n"
        explanation += "• Follow institutional triage protocols\n"
        explanation += "• Document assessment and rationale\n\n"
        
        explanation += "Note: This is a simplified explanation. AI triage decisions should always be validated by qualified healthcare professionals."
        
        return explanation
    
    def get_provider_status(self):
        """Get status of available providers"""
        status = {
            "openrouter": {
                "available": self.openrouter_client is not None,
                "key_configured": self.openrouter_key is not None,
                "library_available": REQUESTS_AVAILABLE
            },
            "openai": {
                "available": self.openai_client is not None,
                "key_configured": self.openai_key is not None,
                "library_available": OPENAI_AVAILABLE
            }
        }
        return status

if __name__ == "__main__":
    # Dummy setup for demonstration
    num_numerical = 7
    num_boolean = 268
    num_temporal = 3
    num_classes = 3
    
    # Create a dummy model instance
    model = TriageModel(num_numerical, num_boolean, num_temporal, num_classes)
    
    # Create a dummy data point (batch size 1)
    dummy_numerical_data = torch.randn(1, num_numerical)
    dummy_boolean_data = torch.randint(0, 2, (1, num_boolean)).float()
    dummy_temporal_data = torch.randn(1, num_temporal)
    
    # Simulate a prediction
    with torch.no_grad():
        logits = model(dummy_numerical_data, dummy_boolean_data, dummy_temporal_data)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # Dummy feature names (replace with actual feature names from feature_engineering.py)
    dummy_feature_names = [f"feature_{i}" for i in range(num_numerical + num_boolean + num_temporal)]
    
    print("\n--- Phase 4: Explainable AI (XAI) and LLM Integration ---")
    
    print("\nStep 4.1: Built-in XAI (Feature Importance)...")
    feature_importance_scores = generate_feature_importance(
        model, 
        torch.cat((dummy_numerical_data, dummy_boolean_data, dummy_temporal_data), dim=1), 
        dummy_feature_names,
        method="random" # Change to "shap" or "permutation" to test
    )
    print("Top 5 Feature Importance Scores:")
    for feature, score in feature_importance_scores[:5]:
        print(f"- {feature}: {score:.4f}")

    print("\nStep 4.2: LLM-Enhanced Explanation Module...")
    patient_context = {
        "age": 65,
        "gender": "Male",
        "symptoms": "chest pain, shortness of breath",
        "vitals": "BP 140/90, HR 95, RR 20, Temp 37.0, Sat 95%"
    }
    llm_explanation = generate_llm_explanation(
        predicted_class, 
        feature_importance_scores[:5], 
        patient_context,
        method="placeholder" # Change to "openai_gpt" to test
    )
    print(llm_explanation)

    print("\nStep 4.3: Synthetic Rare Case Generation (Conceptual - not implemented here)...")
    print("This would involve using LLMs (e.g., GPT-4o) to generate new synthetic patient records for stress-testing.")

    print("\nStep 4.4: Real-Time Interpretability of Boolean Rule Chains...")
    boolean_rules = extract_boolean_rules(model, dummy_feature_names)
    print("Extracted Boolean Rules:")
    for rule in boolean_rules:
        print(f"- {rule}")