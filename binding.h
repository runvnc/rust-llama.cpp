#ifndef BINDING_H
#define BINDING_H

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration of the C++ class
class LlamaCppSimple;

// C-compatible function declarations

 #ifdef __cplusplus
 extern "C" {
 #endif

 typedef void (*callback)(const char*);

 // Wrapper functions for the LlamaCppSimple class
 LlamaCppSimple* llama_create(const char* model_path, int context, int gpu_layers, int threads, int seed);
 void llama_destroy(LlamaCppSimple* instance);
 int llama_generate_text(LlamaCppSimple* instance, const char* prompt, int total_tokens, callback cb);

 #ifdef __cplusplus
 }
 #endif


#endif // BINDING_H
