use crabml::error::Result;
use crabml::tensor::Tensor;

use crate::llama2::Llama2Runner;
use crate::Llama2Sampler;

struct Llama2Chat<T: Tensor> {
    runner: Llama2Runner<T>,
    sampler: Llama2Sampler,
}

impl<T: Tensor> Llama2Chat<T> {
    pub fn new(runner: Llama2Runner<T>, sampler: Llama2Sampler) -> Result<Self> {
        let mut chat = Self { runner, sampler };
        // insert BOS token
        chat.runner.prefill("", &mut chat.sampler, true, false)?;
        Ok(chat)
    }

    pub fn chat(
        &mut self,
        prompt: &str,
        prepend_end_token: bool,
    ) -> Result<impl Iterator<Item = Result<String>> + '_> {
        let prompt = wrap_prompt(prompt, prepend_end_token);
        let (_, last_token, token) =
            self.runner
                .prefill(&prompt, &mut self.sampler, false, false)?;
        // self.runner.generate(pos, prev_token, token, steps, sampler);
        Ok(vec![].into_iter())
    }
}

fn wrap_prompt(prompt: &str, prepend_end_token: bool) -> String {
    let end_token = if prepend_end_token { "</s>" } else { "" };
    format!("{}<s> [INST] {} [/INST]", end_token, prompt)
}
