



use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

extern "C" {
    fn llama_create(model_path: *const c_char, context: i32, gpu_layers: i32, threads: i32, seed: i32) -> *mut c_void;
    fn llama_destroy(instance: *mut c_void);
    fn llama_generate_text(instance: *mut c_void, prompt: *const c_char, total_tokens: i32, cb: extern "C" fn(*mut c_void, *const c_char)) -> i32;
}

pub struct LlamaCppSimple {
    inner: *mut c_void,
}

impl LlamaCppSimple {
    pub fn new(model_path: &str, context: i32, gpu_layers: i32, threads: i32, seed: i32) -> Self {
        let c_model_path = CString::new(model_path).expect("CString::new failed");
        let inner = unsafe { llama_create(c_model_path.as_ptr(), context, gpu_layers, threads, seed) };
        if inner.is_null() {
            panic!("Failed to create LlamaCppSimple instance");
        }
        LlamaCppSimple { inner }
    }

    pub fn generate_text<F>(&self, prompt: &str, total_tokens: i32, mut user_callback: F) -> i32
    where
        F: FnMut(String),
    {
        let c_prompt = CString::new(prompt).expect("CString::new failed");
        extern "C" fn callback<F>(user_data: *mut c_void, token: *const c_char)
        where
            F: FnMut(String),
        {
            let user_callback: &mut F = unsafe { &mut *(user_data as *mut F) };
            let token_str = unsafe { CStr::from_ptr(token).to_string_lossy().into_owned() };
            user_callback(token_str);
        }

        let callback_data: *mut c_void = &mut user_callback as *mut _ as *mut c_void;
        unsafe { llama_generate_text(self.inner, c_prompt.as_ptr(), total_tokens, callback::<F>) }
    }
}

impl Drop for LlamaCppSimple {
    fn drop(&mut self) {
        unsafe {
            llama_destroy(self.inner);
        }
    }
}
