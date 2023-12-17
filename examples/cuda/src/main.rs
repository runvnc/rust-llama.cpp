use llama_cpp_rs::{LlamaOptions, LlamaCppSimple};
use strfmt::strfmt;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::io::{self, Write};


fn main() {
    let options = LlamaOptions::default();

    let llama = LlamaCppSimple::new(LlamaOptions {
        model_path: "models/orca-2-7b.Q4_0.gguf".to_string(),
        context: 2048,
        ..Default::default()
    })
    .unwrap();

    let system_message = "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior.";
    let user_message = "How can you determine if a restaurant is popular among locals or mainly attracts tourists, and why might this information be useful?";
    let mut vars = HashMap::new();
    vars.insert("system_message".to_string(), system_message.to_string());
    vars.insert("user_message".to_string(), user_message.to_string());

    let promptfmt = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant";
    let prompt = strfmt(&promptfmt,&vars).unwrap();

    let total_tokens = 256;

    let answer = Arc::new(Mutex::new(String::new()));
    let answer_clone = Arc::clone(&answer);

    println!("Start.");

    let result = llama.generate_text(prompt.as_str(), total_tokens, Box::new(move |token_str| {
        print!("{}", token_str);
        io::stdout().flush().unwrap();

        let mut answer = answer_clone.lock().unwrap();
        answer.push_str(&token_str);
        true
    }));

    println!("Done.");

    let final_answer = answer.lock().unwrap();

    println!("Answer was: {}", final_answer);
    println!();
    println!();
    println!();


    let prompt2_ = (&*final_answer).to_string() + "<|im_end|>\n<|im_start|>user\nGive me a list of the key points of your first answer.<|im_end|>\n<|im_start|>assistant";
    let prompt2 = "<|im_start|>assistant\nThere are different ways to determine if a restaurant is popular among locals or mainly attracts tourists, and one possible reason why this information might be useful is to avoid overpriced or low-quality food. Here are some possible steps to find this information:\n\n- Look for online reviews from locals or travel blogs that mention the restaurant and its food, service, and atmosphere. These can give you an idea of how the restaurant is perceived by the local community and what they like or dislike about it.\n- Check the restaurant's website or social media pages for any information about their target audience, location, or specialties. These can indicate whether the<|im_end|>\n<|im_start|>user\nGive me a list of the key points of your first answer.<|im_end|>\n<|im_start|>assistant";

    println!("Prompt2_: {}", prompt2_);
    println!();
    println!();
    println!("Prompt2: {}", prompt2);

    println!();
    println!();
    println!();

    let result2 = llama.generate_text(prompt2_.as_str(), 256, Box::new(move |token| {
        print!("{}", token);
        io::stdout().flush().unwrap();
        true
    }));

    //assert!(!generated_text.is_empty());
}


