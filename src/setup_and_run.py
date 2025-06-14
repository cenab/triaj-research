#!/usr/bin/env python3
"""
FairTriEdge-FL Setup and Run Script
This script helps you set up and run the federated learning experiment.
"""

import os
import sys
from dotenv import load_dotenv
from config_loader import load_experiment_config
from config_templates import create_config_file, list_templates

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    print("🔧 Loading environment configuration...")
    
    # Check for API keys
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    openai_key = os.getenv('OPENAI_API_KEY')
    
    if openrouter_key and openrouter_key != 'your_openrouter_api_key_here':
        print("✅ OpenRouter API key found")
    elif openai_key and openai_key != 'your_openai_api_key_here':
        print("✅ OpenAI API key found")
    else:
        print("⚠️  No API keys configured. LLM explanations will be limited.")
        print("   Please edit the .env file to add your API keys.")
    
    # Display current configuration
    print(f"📊 Configuration:")
    print(f"   - LLM Provider: {os.getenv('LLM_PROVIDER', 'openrouter')}")
    print(f"   - OpenRouter Model: {os.getenv('OPENROUTER_MODEL', 'anthropic/claude-3-sonnet')}")
    print(f"   - Privacy Epsilon: {os.getenv('PRIVACY_EPSILON', '1.0')}")
    print(f"   - FL Rounds: {os.getenv('FL_NUM_ROUNDS', '5')}")
    print(f"   - FL Clients: {os.getenv('FL_NUM_CLIENTS', '3')}")
    print()

def check_data_file():
    """Check if the required data file exists"""
    data_file = 'triaj_data.csv'
    if os.path.exists(data_file):
        print(f"✅ Data file '{data_file}' found")
        return True
    else:
        print(f"❌ Data file '{data_file}' not found")
        print("   Please ensure your medical triage data is saved as 'triaj_data.csv'")
        return False

def run_experiment():
    """Run the main experiment"""
    print("🚀 Starting FairTriEdge-FL experiment...")
    print("=" * 60)
    
    # Import and run main experiment
    try:
        from main import main
        main()
    except Exception as e:
        print(f"❌ Error running experiment: {e}")
        return False
    
    return True

def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    print("=" * 60)
    
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'run_tests.py'], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def create_config_from_template():
    """Create configuration from template"""
    print("\n📋 Available Configuration Templates:")
    list_templates()
    
    template_name = input("\nEnter template name (or 'cancel' to go back): ").strip().lower()
    
    if template_name == 'cancel':
        return
    
    output_file = input("Output file name (default: config.json): ").strip()
    if not output_file:
        output_file = "config.json"
    
    if create_config_file(template_name, output_file):
        print(f"✅ Configuration created successfully!")
        
        # Load and display the new config
        try:
            config_loader = load_experiment_config(output_file)
            config_loader.print_config_summary()
        except Exception as e:
            print(f"⚠️  Could not load new config: {e}")

def validate_current_config():
    """Validate and display current configuration"""
    try:
        config_loader = load_experiment_config()
        config_loader.print_config_summary()
        config_loader.validate_config()
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")

def main():
    """Main setup and run function"""
    print("🏥 FairTriEdge-FL: Federated Learning for Medical Triage")
    print("=" * 60)
    
    # Load environment
    load_environment()
    
    # Check data file
    data_exists = check_data_file()
    
    # Show menu
    while True:
        print("\nWhat would you like to do?")
        print("1. Run the full experiment")
        print("2. Run tests only")
        print("3. Check configuration")
        print("4. Create config from template")
        print("5. Validate current config")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            if not data_exists:
                print("⚠️  Cannot run experiment without data file. Please add 'triaj_data.csv'")
                continue
            run_experiment()
            break
        elif choice == '2':
            run_tests()
            break
        elif choice == '3':
            load_environment()
            check_data_file()
        elif choice == '4':
            create_config_from_template()
        elif choice == '5':
            validate_current_config()
        elif choice == '6':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()