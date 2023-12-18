use lazy_static::lazy_static;
use libc::{c_char, c_void};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::sync::Mutex;

mod bindings {
    include!("../bindings.rs");
}

lazy_static! {
    static ref CALLBACKS: Mutex<HashMap<usize, Box<dyn FnMut(String) -> bool + Send + 'static>>> =
        Mutex::new(HashMap::new());
}

#[derive(Debug)]
pub struct LlamaCppSimple {
    inner: *mut bindings::LlamaCppSimple,
}

pub struct LlamaOptions {
    pub model_path: String,
    pub context: i32,
    pub gpu_layers: i32,
    pub threads: i32,
    pub seed: i32,
}

unsafe impl Send for LlamaCppSimple {}
unsafe impl Sync for LlamaCppSimple {}


impl Default for LlamaOptions {
    fn default() -> Self {
        LlamaOptions {
            model_path: "path/to/model".to_string(),
            context: 4096,
            gpu_layers: 20,
            threads: 4,
            seed: 777,
        }
    }
}

fn set_callback(
    state: *mut c_void,
    callback: Option<Box<dyn FnMut(String) -> bool + Send + 'static>>,
) {
    let mut callbacks = CALLBACKS.lock().unwrap();

    if let Some(callback) = callback {
        callbacks.insert(state as usize, callback);
    } else {
        callbacks.remove(&(state as usize));
    }
}

impl Default for LlamaCppSimple {
    fn default() -> Self {
        LlamaCppSimple::new(LlamaOptions::default())
            .expect("Failed to create LlamaCppSimple with default parameters")
    }
}

impl LlamaCppSimple {
    pub fn new(options: LlamaOptions) -> Option<Self> {
        let c_model_path = CString::new(options.model_path).unwrap();
        let inner = unsafe {
            bindings::llama_create(
                c_model_path.as_ptr(),
                options.context,
                options.gpu_layers,
                options.threads,
                options.seed,
            )
        };
        if inner.is_null() {
            None
        } else {
            Some(Self { inner })
        }
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        total_tokens: i32,
        callback: Box<dyn FnMut(String) -> bool + Send + 'static>,
    ) -> i32 {
        let c_prompt = CString::new(prompt).expect("CString::new failed");

        unsafe { set_callback(bindings::llama_get_context(self.inner), Some(callback)); }
        //unsafe { set_callback(self.inner as *mut c_void, Some(callback)); }

        unsafe { bindings::llama_generate_text(self.inner, c_prompt.as_ptr(), total_tokens) }
    }
}

impl Drop for LlamaCppSimple {
    fn drop(&mut self) {
        unsafe {
            bindings::llama_destroy(self.inner);
        }
    }
}

#[no_mangle]
extern "C" fn tokenCallback(state: *mut c_void, token: *const c_char) -> bool {
    let mut callbacks = CALLBACKS.lock().unwrap();
    if let Some(callback) = callbacks.get_mut(&(state as usize)) {
        let c_str: &CStr = unsafe { CStr::from_ptr(token) };
        let str_slice: &str = c_str.to_str().unwrap();
        let string: String = str_slice.to_owned();
        return callback(string);
    } else {
        println!("Could not find callback");

    }

    true
}

