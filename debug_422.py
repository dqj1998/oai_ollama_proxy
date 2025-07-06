#!/usr/bin/env python3
"""
Debug script to test different request formats and identify validation issues
"""

import httpx
import json
import asyncio

BASE_URL = "http://localhost:8889"

async def test_request_formats():
    """Test various request formats to identify what causes 422 errors"""
    
    test_cases = [
        {
            "name": "Basic valid request",
            "data": {
                "model": "gemma3:12b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        },
        {
            "name": "Request with extra fields",
            "data": {
                "model": "gemma3:12b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 100,
                "temperature": 0.7,
                "unknown_field": "should be ignored"
            }
        },
        {
            "name": "Request with system message",
            "data": {
                "model": "gemma3:12b",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ]
            }
        },
        {
            "name": "Empty messages (should fail)",
            "data": {
                "model": "gemma3:12b",
                "messages": []
            }
        },
        {
            "name": "Missing model (should fail)",
            "data": {
                "messages": [
                    {"role": "user", "content": "Hello"}
                ]
            }
        },
        {
            "name": "Invalid message role (might fail)",
            "data": {
                "model": "gemma3:12b",
                "messages": [
                    {"role": "invalid_role", "content": "Hello"}
                ]
            }
        },
        {
            "name": "Missing content in message (should fail)",
            "data": {
                "model": "gemma3:12b",
                "messages": [
                    {"role": "user"}
                ]
            }
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for test_case in test_cases:
            print(f"\n🧪 Testing: {test_case['name']}")
            print("-" * 50)
            
            try:
                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json=test_case['data'],
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    print("✅ SUCCESS - Request accepted")
                    data = response.json()
                    if 'choices' in data:
                        print(f"   Response: {data['choices'][0]['message']['content'][:50]}...")
                elif response.status_code == 422:
                    print("❌ VALIDATION ERROR (422)")
                    try:
                        error_data = response.json()
                        print(f"   Error: {json.dumps(error_data, indent=2)}")
                    except Exception:
                        print(f"   Raw error: {response.text}")
                else:
                    print(f"❌ ERROR {response.status_code}")
                    print(f"   Response: {response.text}")
                    
            except Exception as e:
                print(f"❌ EXCEPTION: {e}")

async def test_your_exact_request():
    """Test with the exact format you might be using"""
    print("\n🔍 Testing common request formats that might cause issues:")
    print("=" * 60)
    
    # Common problematic patterns
    problematic_requests = [
        {
            "name": "Request with null values",
            "data": {
                "model": "gemma3:12b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": None,
                "temperature": None
            }
        },
        {
            "name": "Request with string numbers",
            "data": {
                "model": "gemma3:12b", 
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": "100",  # Should be int
                "temperature": "0.7"  # Should be float
            }
        },
        {
            "name": "Request with wrong message structure",
            "data": {
                "model": "gemma3:12b",
                "messages": "Hello"  # Should be array
            }
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for test_case in problematic_requests:
            print(f"\n🧪 {test_case['name']}")
            print("-" * 30)
            
            try:
                response = await client.post(
                    f"{BASE_URL}/v1/chat/completions",
                    json=test_case['data'],
                    timeout=5.0
                )
                
                if response.status_code == 422:
                    print("❌ VALIDATION ERROR (as expected)")
                    error_data = response.json()
                    print("   Details:")
                    for detail in error_data.get('details', []):
                        print(f"     - {detail['field']}: {detail['message']}")
                else:
                    print(f"   Status: {response.status_code}")
                    
            except Exception as e:
                print(f"   Exception: {e}")

async def main():
    print("🐛 OpenAI to Ollama Proxy - Debug Tool")
    print("====================================")
    
    # Check if proxy is running
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                print("✅ Proxy is running")
            else:
                print(f"⚠️  Proxy returned status {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to proxy: {e}")
        print("Make sure the proxy is running on http://localhost:8889")
        return
    
    await test_request_formats()
    await test_your_exact_request()
    
    print("\n" + "=" * 60)
    print("💡 Tips to fix 422 errors:")
    print("1. Ensure 'model' field is a string")
    print("2. Ensure 'messages' is an array of objects")
    print("3. Each message must have 'role' and 'content' fields")
    print("4. Use proper data types (int for max_tokens, float for temperature)")
    print("5. Check the detailed error message above for specific issues")

if __name__ == "__main__":
    asyncio.run(main())
