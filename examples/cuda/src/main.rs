use llama_cpp_rs::{LlamaOptions, LlamaCppSimple};


fn main() {
    let options = LlamaOptions::default();

    let llama = LlamaCppSimple::new(LlamaOptions {
        model_path: "models/orca-2-7b.Q4_0.gguf".to_string(),
        context: 512,
        ..Default::default()
    })
    .unwrap();
    let prompt = "abcdefg";
    let total_tokens = 15;

    let generated_text = Vec::new();
    let mut cloned_text = generated_text.clone();
    println!("Start.");    
    let result = llama.generate_text(prompt, total_tokens, Box::new(move |token| {
        print!("{}", token);
        cloned_text.push(token);
        true
    }));
    println!("Done.");

    assert_eq!(result, total_tokens);

    let result2 = llama.generate_text("12345", total_tokens, Box::new(move |token| {
        print!("{}", token);
        true
    }));
 

    //assert!(!generated_text.is_empty());
}


