extern "C" {
#include "binding.h"
}

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

class LlamaCppSimple {
  public:
  LlamaCppSimple(const std::string& path, int context=2048, int gpuLayers=20, int threads=4, int seed=777) :
    modelPath(path), contextTokenLen(context), randSeed(seed)
  {
    llama_backend_init(gptParams.numa);
    loadModel(gpuLayers, threads);
    initContext();
 }

  llama_context* getContext() {
    return currentContext;
  }

  int generateText(const std::string& prompt, int maxNewTokens) {
    //initContext();
 
    llama_batch batch;
    fprintf(stderr, "top of generateText\n");
    
    int promptTokenCount = initAndPredictFirstToken(prompt, maxNewTokens, batch);
    int totalTokens = promptTokenCount + maxNewTokens;

    if (totalTokens > contextTokenLen) {
        fprintf(stderr , "%s: error: total tokens exceeds context length\n" , __func__);
        throw std::runtime_error("error: total tokens exceeds context length.");
    }

    llama_token selectedToken = 0;
    int currentTokenIndex = batch.n_tokens;
 
    llama_token endOfSequence = llama_token_eos(model);
    bool predictedEnd = false;

    fprintf(stderr, "loop begin\n");

    do {
      fprintf(stderr, "2\n");
      selectedToken = bestFromLastDecode(batch);
      fprintf(stderr, "3\n");
  
      predictedEnd = (selectedToken == endOfSequence);
      if (!predictedEnd) {
        fprintf(stderr, "4\n");
 
        outputSingleTokenAsString(selectedToken);
 
        llama_batch_clear(batch);
        llama_batch_add(batch, selectedToken, currentTokenIndex, { 0 }, true);
  
        decodeToNextTokenScores(batch);
        fprintf(stderr, "5\n");
 
      }
      currentTokenIndex++;
    } while (!predictedEnd && currentTokenIndex < totalTokens);
 
    llama_batch_free(batch);

    return currentTokenIndex;
  }

  ~LlamaCppSimple() {
    llama_free(currentContext);
    llama_free_model(model);

    llama_backend_free();
  }
 
  private:

  void loadModel(int gpuLayers, int threads) {
    gptParams.model = modelPath;
    gptParams.n_threads = threads;
    modelParams = llama_model_default_params();

    modelParams.n_gpu_layers = gpuLayers;

    model = llama_load_model_from_file(gptParams.model.c_str(), modelParams);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        throw std::runtime_error("Unable to load model.");
    }
  }

  void initContext() {
    fprintf(stderr, "initializing context..\n");
    
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed  = randSeed;
    ctx_params.n_ctx = contextTokenLen;
    ctx_params.n_threads = gptParams.n_threads;
    ctx_params.n_threads_batch = gptParams.n_threads_batch == -1 ? gptParams.n_threads : gptParams.n_threads_batch;

    currentContext = llama_new_context_with_model(model, ctx_params);

    if (currentContext == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        throw std::runtime_error("Failed to create the llama_context");
    }
  }

  inline void tokenize(const std::string& inputString, int totalTokens, std::vector<llama_token>& tokens_list) {
    tokens_list = llama_tokenize(currentContext, inputString, true);

    const int n_ctx    = llama_n_ctx(currentContext);
    const int n_kv_req = tokens_list.size() + (totalTokens - tokens_list.size());

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        throw std::runtime_error("Error: input overran context length.");
    }
  }

  inline void outputSingleTokenAsString(llama_token& token) {
    const char* str = llama_token_to_piece(currentContext, token).c_str();
    tokenCallback((void*)currentContext, (char*)str);
  }

  inline void outputTokensAsString(const std::vector<llama_token>& tokens) {
    for (auto id : tokens) {
      const char* str = llama_token_to_piece(currentContext, id).c_str();
      tokenCallback((void*)currentContext, (char*)str);
    }
  }

  inline int initAndPredictFirstToken(const std::string& prompt, int maxNewTokens, llama_batch& batch) {
    std::vector<llama_token> promptTokens;
    tokenize(prompt, contextTokenLen, promptTokens);
    //tokenize(prompt, totalTokens, promptTokens);

    fprintf(stderr, "a\n");
    outputTokensAsString(promptTokens);

    fprintf(stderr, "b\n");
    batch = llama_batch_init(512, 0, 1);

    fprintf(stderr, "c\n");

    for (size_t i = 0; i < promptTokens.size(); i++) {
        llama_batch_add(batch, promptTokens[i], i, { 0 }, false);
    }
    fprintf(stderr, "d\n");
    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    fprintf(stderr, "e\n");
    if (llama_decode(currentContext, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        throw std::runtime_error("llama_decode() failed");
    }
    fprintf(stderr, "f\n");
    return promptTokens.size();
  }

  inline llama_token bestFromLastDecode(llama_batch& batch) {
    int numTokensInVocabulary = llama_n_vocab(model);
    auto* tokenLikelihoodScores  = llama_get_logits_ith(currentContext, batch.n_tokens - 1);

    std::vector<llama_token_data> candidates;
    candidates.reserve(numTokensInVocabulary);

    for (llama_token token_id = 0; token_id < numTokensInVocabulary; token_id++) {
        candidates.emplace_back(llama_token_data{ token_id, tokenLikelihoodScores[token_id], 0.0f });
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    const llama_token highestScoringToken = llama_sample_token_greedy(currentContext, &candidates_p);

    return highestScoringToken;
  }

  inline void decodeToNextTokenScores(llama_batch& batch) {
    // evaluate the current batch with the transformer model
    if (llama_decode(currentContext, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        throw std::runtime_error("Error 1313: input exceeded context length.");
    }
  }

 
  llama_model* model;
  llama_model_params modelParams;
  gpt_params gptParams;
  std::string modelPath;
  llama_context* currentContext;
  int contextTokenLen, randSeed;
};

// Wrapper function definitions

extern "C" {

LlamaCppSimple* llama_create(const char* model_path, int context, int gpu_layers, int threads, int seed) {
    try {
        return new LlamaCppSimple(model_path, context, gpu_layers, threads, seed);
    } catch (const std::exception& e) {
        // Handle exceptions if necessary
        return nullptr;
    }
}

void llama_destroy(LlamaCppSimple* instance) {
    delete instance;
}

void* llama_get_context(LlamaCppSimple* instance) {
  return (void*)(instance->getContext());
}

int llama_generate_text(LlamaCppSimple* instance, const char* prompt, int total_tokens) {
    if (instance == nullptr) {
        return -1; // Indicate error
    }
    try {
        return instance->generateText(prompt, total_tokens);
    } catch (const std::exception& e) {
        // Handle exceptions if necessary
        return -1; // Indicate error
    }
}

} // extern "C"

// Remove the example usage from the main function
