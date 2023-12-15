#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

typedef void (*callback)(const char*);

void printStr(const char*);

class LlamaCppSimple {
  public:
  LlamaCppSimple(const std::string& path, int context=2048, int gpuLayers=20, int threads=4, int seed=777) :
    modelPath(path), contextTokenLen(context), randSeed(seed)
  {
    llama_backend_init(gptParams.numa);
    loadModel(gpuLayers, threads);
 }

  int generateText(const std::string& prompt, int totalTokens, callback cb) {
    initContext();
 
    llama_batch batch;
    
    initAndPredictFirstToken(prompt, totalTokens, batch, cb);
 
    llama_token selectedToken = 0;
    int currentTokenIndex = batch.n_tokens;
 
    llama_token endOfSequence = llama_token_eos(model);
    bool predictedEnd = false;

    do {
      selectedToken = bestFromLastDecode(batch);
 
      predictedEnd = (selectedToken == endOfSequence);
      if (!predictedEnd) {
        outputSingleTokenAsString(selectedToken, cb);
 
        llama_batch_clear(batch);
        llama_batch_add(batch, selectedToken, currentTokenIndex, { 0 }, true);
  
        decodeToNextTokenScores(batch);
      }
      currentTokenIndex++;
    } while (!predictedEnd && currentTokenIndex < totalTokens);
 
    llama_batch_free(batch);

    return currentTokenIndex;
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

  inline void outputSingleTokenAsString(llama_token& token, callback tokenCallback) {
    const char* str = llama_token_to_piece(currentContext, token).c_str();
    tokenCallback(str);
  }

  inline void outputTokensAsString(const std::vector<llama_token>& tokens, callback tokenCallback) {
    for (auto id : tokens) {
      const char* str = llama_token_to_piece(currentContext, id).c_str();
      tokenCallback(str);    
      //printf("%s", llama_token_to_piece(currentContext, id).c_str());
    }
  }

  inline void initAndPredictFirstToken(const std::string& prompt, int totalTokens, llama_batch& batch, callback cb) {
    std::vector<llama_token> promptTokens;
    tokenize(prompt, totalTokens, promptTokens);
    outputTokensAsString(promptTokens, cb);

    batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < promptTokens.size(); i++) {
        llama_batch_add(batch, promptTokens[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(currentContext, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        throw std::runtime_error("llama_decode() failed");
    }
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

  ~LlamaCppSimple() {
    llama_free(currentContext);
    llama_free_model(model);

    llama_backend_free();
  }
  
  llama_model* model;
  llama_model_params modelParams;
  gpt_params gptParams;
  std::string modelPath;
  llama_context* currentContext;
  int contextTokenLen, randSeed;
};

void printStr(const char* str) {
  printf("%s", str);
  fflush(stdout);
}

int main(int /* argc */, char ** argv) {
    auto modelFile = argv[1];
    auto llamaCpp = new LlamaCppSimple(modelFile, 2048, 1);

    const auto start = ggml_time_us();

    auto numGeneratedTokens = llamaCpp->generateText("The top 100 fruits are:", 32, printStr);

    const auto end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, numGeneratedTokens, (end - start) / 1000000.0f, numGeneratedTokens / ((end - start) / 1000000.0f));

    const auto start2 = ggml_time_us();
    auto numGeneratedTokens2 = llamaCpp->generateText("The full moon in the crisp night, ", 32, printStr);
    const auto end2 = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, numGeneratedTokens2, (end2 - start2) / 1000000.0f, numGeneratedTokens2 / ((end2 - start2) / 1000000.0f));

    return 0;
}
