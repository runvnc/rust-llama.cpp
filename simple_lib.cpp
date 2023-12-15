#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

class LlamaCppSimple {

  LlamaCppSimple(const std::string& path, int context=2048, int offloadLayers=20, int seed=777) :
    modelPath(path), contextTokenLen(context), gpuLayers(offloadLayers), randSeed(seed)
  {
    llama_backend_init(gptParams.numa);
    load_model(modelPath, gpuLayers);
    initContext();
  }

  public:

  std::string generateText(const std::string& prompt) { //, callback) {
    auto batch = initAndPredictFirstToken(prompt);

    llama_token selectedToken = 0;
    int numGeneratedTokens = 0;
    llama_token endOfSequence = llama_token_eos(model);
    bool predictedEnd = false;

    do {
      selectedToken = bestFromLastDecode(&batch);
      predictedEnd = (selectedToken == endOfSequence);
      if (!predictedEnd) {
        outputString([predictedToken])

        llama_batch_clear(batch);
        llama_batch_add(batch, selectedToken, numGeneratedTokens, { 0 }, true);

        decodeToNextTokenScores(batch);
      }
      numGeneratedTokens++;
    } while (!predictedEnd && numGeneratedTokens < totalTokens)

    llama_batch_free(batch);

    return numGeneratedTokens;
  }

  private:

  void loadModel(const std::string& path) {
    params.model = modelPath;
    modelParams = llama_model_default_params();

    modelParams.n_gpu_layers = gpuLayers;

    model = llama_load_model_from_file(params.model.c_str(), modelParams);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        throw std::runtime_error("Unable to load model.");
    }
  }

  initContext() {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.seed  = randSeed;
    ctx_params.n_ctx = contextTokenLen;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

    currentContext = llama_new_context_with_model(model, ctx_params);

    if (currentContext == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        throw std::runtime_error("Failed to create the llama_context");
    }
  }

  void tokenize(const std::string& inputString) {
    std::vector<llama_token> tokens_list;
    tokens_list = llama_tokenize(currentContext, inputString, true);

    const int n_ctx    = llama_n_ctx(currentContext);
    const int n_kv_req = tokens_list.size() + (n_len - tokens_list.size());

    LOG_TEE("\n%s: n_len = %d, n_ctx = %d, n_kv_req = %d\n", __func__, n_len, n_ctx, n_kv_req);

    // make sure the KV cache is big enough to hold all the prompt and generated tokens
    if (n_kv_req > n_ctx) {
        throw std::runtime_error("Error: input overran context length.");
    }
  }

  void outputString(const std::vector<llama_token>& tokens) {
    for (auto id : tokens) {
        fprintf(stderr, "%s", llama_token_to_piece(currentContext, id).c_str());
    }
    fflush(stdout);
  }

  llama_batch initAndPredictFirstToken(const std::string& prompt) {
    const int totalTokens = 32;
    auto promptTokens = tokenize(prompt);
    outputString(promptTokens);

    llama_batch batch = llama_batch_init(512, 0, 1);

    for (size_t i = 0; i < promptTokens.size(); i++) {
        llama_batch_add(batch, promptTokens[i], i, { 0 }, false);
    }

    // llama_decode will output logits only for the last token of the prompt
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch) != 0) {
        LOG_TEE("%s: llama_decode() failed\n", __func__);
        throw std::runtime_error("llama_decode() failed")
    }
    return batch
  }

  inline llama_token bestFromLastDecode(llama_batch* batch) {
    int numTokensInVocabulary = llama_n_vocab(model);
    auto* tokenLikelihoodScores  = llama_get_logits_ith(currentContext, batch->n_tokens - 1);

    std::vector<llama_token_data> candidates;
    candidates.reserve(n_vocab);

    for (llama_token token_id = 0; token_id < numTokensInVocabulary; token_id++) {
        candidates.emplace_back(llama_token_data{ token_id, tokenLikelihoodScores[token_id], 0.0f });
    }

    llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

    const llama_token highestScoringToken = llama_sample_token_greedy(currentContext, &candidates_p);

    return highestScoringToken;
  }

  inline void decodeToNextTokenScores(llama_batch* batch) {
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

  llama_model_params modelParams;
  llama_model* model;
  gpt_params gptParams;
  llama_context* currentContext;
  int contextTokenLen, randSeed;
}


int main(int argc, char ** argv) {
    auto modelFile = argv[1];
    auto llamaCpp = new LLamaCppSimple(modelFile);

    const auto start = ggml_time_us();

    auto numGeneratedTokens = llamaCpp.generateText("It was the best of times, ");

    const auto end = ggml_time_us();

    LOG_TEE("%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, numGeneratedTokens, (end - start) / 1000000.0f, numGeneratedTokens / ((end - start) / 1000000.0f));

    return 0;
}
