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
  LlamaCppSimple(const std::string& path, int context=2048, int gpuLayers=20, int threads=4, int seed=777, int batch_size=512) :
    modelPath(path), contextTokenLen(context), randSeed(seed), batchSize(batch_size)
  {
    llama_backend_init(gptParams.numa);
    loadModel(gpuLayers, threads);
    batch = llama_batch_init(batchSize, 0, 1);
    currentTokenIndex = 0;
  }

  llama_context* getContext() {
    return currentContext;
  }

  int generateText(const std::string& prompt, int maxNewTokens) {
    initContext();
    currentTokenIndex = 0;

    llama_batch_clear(batch);

    int promptTokenCount = processPrompt(prompt, maxNewTokens);
    currentTokenIndex = promptTokenCount;

    int totalTokens = promptTokenCount + maxNewTokens;

    if (totalTokens > contextTokenLen) {
        fprintf(stderr , "%s: error: total potential tokens exceeds context length\n" , __func__);
        throw std::runtime_error("error: total potential tokens exceeds context length.");
    }

    llama_token selectedToken = 0;

    llama_token endOfSequence = llama_token_eos(model);
    bool predictedEnd = false;

    do {
      selectedToken = bestFromLastDecode();
  
      predictedEnd = (selectedToken == endOfSequence);
      if (!predictedEnd) {
        llama_batch_clear(batch);
 
        bool should_continue = outputSingleTokenAsString(selectedToken);
        fprintf(stderr, " ### %d ### ", should_continue);

        if (!should_continue) return currentTokenIndex;

        llama_batch_add(batch, selectedToken, currentTokenIndex++, { 0 }, true);
 
        decodeToNextTokenScores();
      } else {
      }
      
    } while (!predictedEnd && currentTokenIndex < totalTokens);

    return currentTokenIndex;
  }

  ~LlamaCppSimple() {
    llama_free(currentContext);
    llama_free_model(model);

    llama_batch_free(batch);

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

    if (currentContext != 0) {
      llama_free(currentContext);
    }
    currentContext = llama_new_context_with_model(model, ctx_params);

    if (currentContext == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        throw std::runtime_error("Failed to create the llama_context");
    }
  }

  inline void tokenize(const std::string& inputString, int totalTokens, std::vector<llama_token>& tokens_list, bool is_start) {
    tokens_list = llama_tokenize(model, inputString, is_start, true);
    llama_token endOfSequence = llama_token_eos(model);
    
    //fprintf(stderr, "EOS is %d\ntokenize result:\n", endOfSequence);
    for (int i = 0; i < tokens_list.size(); i++) {
      const char* str = llama_token_to_piece(currentContext, tokens_list[i]).c_str();
      //fprintf(stderr, " {%s}[%d] ", str, tokens_list[i]);

      if (tokens_list[i] == endOfSequence) {
        fprintf(stderr, " *Found EOS* ");
      }
    }
    //fprintf(stderr, "\n");

    const int n_ctx    = llama_n_ctx(currentContext);
    const int n_kv_req = tokens_list.size() + (totalTokens - tokens_list.size());

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        throw std::runtime_error("Error: input overran context length.");
    }
  }

  inline bool outputSingleTokenAsString(llama_token& token) {
    const char* str = llama_token_to_piece(currentContext, token).c_str();
    return tokenCallback((void*)10000, (char*)str);
  }

  inline bool outputTokensAsString(const std::vector<llama_token>& tokens) {
    for (auto id : tokens) {
      const char* str = llama_token_to_piece(currentContext, id).c_str();
      if (!tokenCallback((void*)10000, (char*)str)) {
        return false;
      }
    }
    return true;
  }

  inline int processPrompt(const std::string& prompt, int maxNewTokens) {
    // TODO: verify that we don't overrun context length 

    std::vector<llama_token> promptTokens;
    tokenize(prompt, contextTokenLen, promptTokens, true);

    fprintf(stderr, "c\n");
    fprintf(stderr, "Prompt tokens len: %d\n", promptTokens.size());

    fprintf(stderr, "Batch size: %d\n", batchSize);

    int processedTokens = 0;

    while (processedTokens < promptTokens.size() ) {
      int start = processedTokens;
      
      llama_batch_clear(batch);

      while (processedTokens < start + batchSize && 
          processedTokens < promptTokens.size() ) { 
          llama_batch_add(batch, promptTokens[processedTokens], processedTokens, { 0 }, false);
          //llama_batch_add(batch, promptTokens[processedTokens], currentTokenIndex, { 0 }, false); 
          processedTokens++;
          //currentTokenIndex++;
      }
      fprintf(stderr, "processed tokens: %d", processedTokens);

      if (processedTokens == promptTokens.size()) {
        // llama_decode will output logits only for the last token of the prompt
        batch.logits[batch.n_tokens - 1] = true;
      }
      if (llama_decode(currentContext, batch) != 0) {
          LOG_TEE("%s: llama_decode() failed\n", __func__);
          throw std::runtime_error("llama_decode() failed");
      }
    }

    return promptTokens.size();
    //return currentTokenIndex;
  }

  inline llama_token bestFromLastDecode() {

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

  inline void decodeToNextTokenScores() {
    // evaluate the current batch with the transformer model
    if (llama_decode(currentContext, batch)) {
        fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
        throw std::runtime_error("Error 1313: input exceeded context length.");
    }
  }
 
  llama_model* model;
  llama_model_params modelParams;
  gpt_params gptParams;
  int currentTokenIndex;
  std::string modelPath;
  llama_context* currentContext = 0;
  llama_batch batch;
  int contextTokenLen, randSeed, batchSize;
};

// Wrapper function definitions

extern "C" {

LlamaCppSimple* llama_create(const char* model_path, int context, int gpu_layers, int threads, int seed, int batch) {
    try {
        return new LlamaCppSimple(model_path, context, gpu_layers, threads, seed, batch);
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
