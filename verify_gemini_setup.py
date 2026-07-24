#!/usr/bin/env python3
"""Quick verification script for Gemini LLM setup."""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def verify_imports():
    """Check if all required packages can be imported."""
    print("[*] Checking imports...")
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("  [+] langchain_google_genai.ChatGoogleGenerativeAI")
    except ImportError as e:
        print(f"  [-] langchain_google_genai: {e}")
        return False
    
    try:
        import google.generativeai
        print("  [+] google.generativeai")
    except ImportError as e:
        print(f"  [-] google.generativeai: {e}")
        return False
    
    return True

def verify_config():
    """Check if config.py has Gemini fields."""
    print("\n[*] Checking config.py...")
    
    try:
        from backend.app.config import Settings
        
        # Check if Gemini fields exist
        settings = Settings()
        
        assert hasattr(settings, 'gemini_api_key'), "Missing gemini_api_key field"
        assert hasattr(settings, 'gemini_model'), "Missing gemini_model field"
        print("  [+] Gemini configuration fields present")
        print(f"    - gemini_model default: {settings.gemini_model}")
        
        return True
    except Exception as e:
        print(f"  [-] Config verification failed: {e}")
        return False

def verify_llm_service():
    """Check if LLMService supports Gemini."""
    print("\n[*] Checking LLMService...")
    
    try:
        from backend.app.rag.llm import LLMService
        import inspect
        
        # Check if _build_llm has gemini support
        source = inspect.getsource(LLMService._build_llm)
        if 'gemini' in source.lower() and 'ChatGoogleGenerativeAI' in source:
            print("  [+] LLMService._build_llm() has Gemini support")
        else:
            print("  [-] LLMService._build_llm() missing Gemini support")
            return False
        
        # Check chat model detection
        source = inspect.getsource(LLMService._use_chat_model)
        if 'gemini' in source:
            print("  [+] LLMService._use_chat_model() includes Gemini")
        else:
            print("  [-] LLMService._use_chat_model() missing Gemini")
            return False
        
        return True
    except Exception as e:
        print(f"  [-] LLMService verification failed: {e}")
        return False

def main():
    """Run all verification checks."""
    print("=" * 60)
    print("Gemini LLM Integration Verification")
    print("=" * 60)
    
    checks = [
        ("Imports", verify_imports),
        ("Configuration", verify_config),
        ("LLM Service", verify_llm_service),
    ]
    
    results = []
    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[-] {name} check failed: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Verification Summary:")
    print("=" * 60)
    
    for name, result in results:
        status = "[+] PASS" if result else "[-] FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("[+] All checks passed! Gemini setup is complete.")
        print("\nNext steps:")
        print("1. Get your API key from: https://aistudio.google.com/app/apikey")
        print("2. Update .env with:")
        print("   LLM_PROVIDER=gemini")
        print("   GEMINI_API_KEY=your-key-here")
        print("   GEMINI_MODEL=gemini-2.0-flash")
        print("3. Restart your application")
    else:
        print("[-] Some checks failed. Review the output above.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
