/* automatically generated by rust-bindgen 0.66.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct LlamaCppSimple {
    _unused: [u8; 0],
}
extern "C" {
    pub fn llama_create(
        model_path: *const ::std::os::raw::c_char,
        context: ::std::os::raw::c_int,
        gpu_layers: ::std::os::raw::c_int,
        threads: ::std::os::raw::c_int,
        seed: ::std::os::raw::c_int,
        batch_size: ::std::os::raw::c_int,
    ) -> *mut LlamaCppSimple;
}
extern "C" {
    pub fn llama_destroy(instance: *mut LlamaCppSimple);
}
extern "C" {
    pub fn llama_get_context(instance: *mut LlamaCppSimple) -> *mut ::std::os::raw::c_void;
}
extern "C" {
    pub fn llama_generate_text(
        instance: *mut LlamaCppSimple,
        prompt: *const ::std::os::raw::c_char,
        total_tokens: ::std::os::raw::c_int,
    ) -> ::std::os::raw::c_int;
}
