# Test Ollama Connection
include("dev_load.jl")

println("🤖 Testing Ollama Connection...")

# 1. List Models
println("\n📋 Listing Models:")
models = OllamaClient.list_models()
if isempty(models)
    println("❌ No models found or Ollama not running.")
    println("   Please run 'ollama serve' and 'ollama pull llama3.2:3b'")
else
    println("✅ Found models: ", join(models, ", "))
    
    # 2. Test Generation
    # Filter for chat models (exclude embedding models)
    chat_models = filter(m -> !contains(m, "embed"), models)
    model_name = isempty(chat_models) ? models[1] : chat_models[end] # Use last one (likely llama/qwen)
    println("\n💭 Testing Generation with $model_name:")
    
    prompt = "What are the key properties of a bone tissue scaffold? List 3."
    println("   Prompt: \"$prompt\"")
    println("   ...")
    
    response = OllamaClient.generate(OllamaModel(model_name), prompt)
    println("\n✅ Response:\n$response")
end
