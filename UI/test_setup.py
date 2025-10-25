"""
Test script to verify Streamlit UI setup and Azure Function connectivity
Run this before starting the main app to ensure everything is configured correctly
"""

import sys
import requests
from pathlib import Path

def check_python_version():
    """Check Python version"""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\n2. Checking dependencies...")
    required = ['streamlit', 'requests', 'PIL', 'plotly']
    missing = []
    
    for package in required:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Not installed")
            missing.append(package)
    
    if missing:
        print(f"\n   Install missing packages: pip install -r requirements.txt")
        return False
    return True

def check_secrets_file():
    """Check if secrets.toml exists"""
    print("\n3. Checking configuration...")
    secrets_path = Path(".streamlit/secrets.toml")
    
    if secrets_path.exists():
        print(f"   ✅ secrets.toml found")
        with open(secrets_path, 'r') as f:
            content = f.read()
            if 'AZURE_FUNCTION_URL' in content:
                print(f"   ✅ AZURE_FUNCTION_URL configured")
                return True
            else:
                print(f"   ⚠️  AZURE_FUNCTION_URL not found in secrets.toml")
                return True
    else:
        print(f"   ⚠️  secrets.toml not found (will use default)")
        return True

def check_azure_function():
    """Check if Azure Function is accessible"""
    print("\n4. Checking Azure Function connectivity...")
    
    # Try to read endpoint from secrets
    endpoint = "http://localhost:7071/api"
    secrets_path = Path(".streamlit/secrets.toml")
    
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                if 'AZURE_FUNCTION_URL' in line and '=' in line:
                    endpoint = line.split('=')[1].strip().strip('"')
                    break
    
    print(f"   Testing endpoint: {endpoint}")
    
    try:
        # Test health endpoint
        health_url = f"{endpoint}/health"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            print(f"   ✅ Azure Function is running")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"   ⚠️  Health check returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to Azure Function")
        print(f"   Make sure it's running: func host start")
        return False
    except requests.exceptions.Timeout:
        print(f"   ❌ Connection timeout")
        return False
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False

def check_warmup_endpoint():
    """Check if warmup endpoint is accessible"""
    print("\n5. Checking warmup endpoint...")
    
    endpoint = "http://localhost:7071/api"
    secrets_path = Path(".streamlit/secrets.toml")
    
    if secrets_path.exists():
        with open(secrets_path, 'r') as f:
            for line in f:
                if 'AZURE_FUNCTION_URL' in line and '=' in line:
                    endpoint = line.split('=')[1].strip().strip('"')
                    break
    
    try:
        warmup_url = f"{endpoint}/warmup"
        print(f"   Testing: {warmup_url}")
        print(f"   (This may take a while for first-time model loading...)")
        
        response = requests.get(warmup_url, timeout=10)
        
        if response.status_code in [200, 202]:
            result = response.json()
            print(f"   ✅ Warmup endpoint accessible")
            print(f"   Status: {result.get('status', 'unknown')}")
            return True
        else:
            print(f"   ⚠️  Warmup returned status {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"   ⚠️  Warmup timeout (models may be loading)")
        print(f"   This is normal for first-time setup")
        return True
    except Exception as e:
        print(f"   ⚠️  Error: {str(e)}")
        return True

def main():
    """Run all checks"""
    print("=" * 60)
    print("Deepfake Detection UI - Setup Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("Configuration", check_secrets_file()))
    results.append(("Azure Function", check_azure_function()))
    results.append(("Warmup Endpoint", check_warmup_endpoint()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    all_passed = all(result[1] for result in results[:4])  # Warmup can be slow
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed! Ready to run: streamlit run app.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Start Azure Function: func host start")
        print("  - Configure endpoint in .streamlit/secrets.toml")
    
    print("\n")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
