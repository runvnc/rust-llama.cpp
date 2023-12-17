#ifndef BINDING_H
#define BINDING_H

#ifdef __cplusplus
extern "C" {
#endif



extern unsigned int tokenCallback(void *, char *);


//typedef void (*callback)(const char*);

// Forward declaration of the C++ class
#ifdef __cplusplus
class LlamaCppSimple;
#else
typedef struct LlamaCppSimple LlamaCppSimple;
#endif

// C-compatible function declarations
LlamaCppSimple* llama_create(const char* model_path, int context, int gpu_layers, int threads, int seed);
void llama_destroy(LlamaCppSimple* instance);
void* llama_get_context(LlamaCppSimple* instance);
int llama_generate_text(LlamaCppSimple* instance, const char* prompt, int total_tokens);

#ifdef __cplusplus
}
#endif


#endif // BINDING_H
