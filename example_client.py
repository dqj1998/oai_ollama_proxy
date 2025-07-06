#!/usr/bin/env python3
"""
Example client for the OpenAI to Ollama Proxy
Demonstrates how to use the proxy with OpenAI-compatible libraries
"""

import os
import asyncio
from openai import AsyncOpenAI

# Configure the client to use our proxy
client = AsyncOpenAI(
    base_url=os.getenv("PROXY_URL", "http://localhost:8889/v1"),
    api_key="not-needed-but-required"  # Proxy doesn't require auth, but OpenAI client does
)

async def simple_chat():
    """Simple chat completion example"""
    print("🤖 Simple Chat Example")
    print("-" * 30)
    
    try:
        response = await client.chat.completions.create(
            model="llama2",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Explain what a proxy server is in one sentence."}
            ],
            max_tokens=100,
            temperature=0.7
        )
        
        print(f"Assistant: {response.choices[0].message.content}")
        print(f"Tokens used: {response.usage.total_tokens}")
        
    except Exception as e:
        print(f"Error: {e}")

async def streaming_chat():
    """Streaming chat completion example"""
    print("\n🌊 Streaming Chat Example")
    print("-" * 30)
    
    try:
        stream = await client.chat.completions.create(
            model="llama2",
            messages=[
                {"role": "user", "content": "Write a haiku about programming."}
            ],
            max_tokens=50,
            stream=True
        )
        
        print("Assistant: ", end="", flush=True)
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()  # New line at the end
        
    except Exception as e:
        print(f"Error: {e}")

async def list_models():
    """List available models"""
    print("\n📋 Available Models")
    print("-" * 30)
    
    try:
        models = await client.models.list()
        for model in models.data:
            print(f"- {model.id}")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Run all examples"""
    print("🚀 OpenAI to Ollama Proxy Client Examples")
    print("=" * 50)
    
    await list_models()
    await simple_chat()
    await streaming_chat()
    
    print("\n✨ Examples completed!")

if __name__ == "__main__":
    asyncio.run(main())
