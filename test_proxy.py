#!/usr/bin/env python3
"""
Test script for the OpenAI to Ollama Proxy
"""

import asyncio
import httpx
import json
import sys
import os

BASE_URL = os.getenv("PROXY_URL", "http://localhost:8889")

async def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False

async def test_list_models():
    """Test listing available models"""
    print("\nTesting model listing...")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Models endpoint working: Found {len(data['data'])} models")
                for model in data['data'][:3]:  # Show first 3 models
                    print(f"  - {model['id']}")
                return True
            else:
                print(f"❌ Models endpoint failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Models endpoint error: {e}")
            return False

async def test_chat_completion():
    """Test non-streaming chat completion"""
    print("\nTesting chat completion (non-streaming)...")
    
    request_data = {
        "model": "llama2",  # Default model
        "messages": [
            {"role": "user", "content": "Hello! Please respond with just 'Hi there!' and nothing else."}
        ],
        "max_tokens": 10,
        "temperature": 0.1
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/v1/chat/completions",
                json=request_data
            )
            
            if response.status_code == 200:
                data = response.json()
                print("✅ Chat completion successful!")
                print(f"  Model: {data['model']}")
                print(f"  Response: {data['choices'][0]['message']['content']}")
                print(f"  Tokens used: {data['usage']['total_tokens']}")
                return True
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                print(f"  Error: {response.text}")
                return False
        except Exception as e:
            print(f"❌ Chat completion error: {e}")
            return False

async def test_streaming_chat():
    """Test streaming chat completion"""
    print("\nTesting streaming chat completion...")
    
    request_data = {
        "model": "llama2",
        "messages": [
            {"role": "user", "content": "Count from 1 to 3, one number per line."}
        ],
        "stream": True,
        "max_tokens": 20
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{BASE_URL}/v1/chat/completions",
                json=request_data
            ) as response:
                
                if response.status_code != 200:
                    print(f"❌ Streaming failed: {response.status_code}")
                    return False
                
                return await process_streaming_response(response)
                    
        except Exception as e:
            print(f"❌ Streaming error: {e}")
            return False

async def process_streaming_response(response):
    """Process the streaming response and return success status"""
    print("✅ Streaming response:")
    chunks_received = 0
    
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
            
        data_str = line[6:]  # Remove "data: " prefix
        
        if data_str.strip() == "[DONE]":
            print("\n  Stream completed successfully!")
            break
        
        content = extract_content_from_chunk(data_str)
        if content:
            print(content, end="", flush=True)
            chunks_received += 1
    
    if chunks_received > 0:
        print(f"\n  Received {chunks_received} content chunks")
        return True
    else:
        print("❌ No content chunks received")
        return False

def extract_content_from_chunk(data_str):
    """Extract content from a streaming chunk"""
    try:
        chunk_data = json.loads(data_str)
        if chunk_data.get("choices") and chunk_data["choices"][0].get("delta"):
            return chunk_data["choices"][0]["delta"].get("content", "")
    except json.JSONDecodeError:
        pass
    return ""

async def main():
    """Run all tests"""
    print("🧪 OpenAI to Ollama Proxy Test Suite")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("List Models", test_list_models),
        ("Chat Completion", test_chat_completion),
        ("Streaming Chat", test_streaming_chat),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! The proxy is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
