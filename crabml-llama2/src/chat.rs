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

    pub fn chat(&mut self, prompt: &str) -> Result<impl Iterator<Item = Result<String>> + '_> {
        let prepend_end_token = self.runner.kv_cache_len() > 1;
        let prompt = wrap_prompt(prompt, prepend_end_token);
        let (pos, last_token, token) =
            self.runner
                .prefill(&prompt, &mut self.sampler, false, false)?;
        let iter = self
            .runner
            .generate(pos, last_token, token, None, &mut self.sampler);
        Ok(iter)
    }
}

fn wrap_prompt(prompt: &str, prepend_end_token: bool) -> String {
    let end_token = if prepend_end_token {
        "<end_of_turn>"
    } else {
        ""
    };
    format!(
        "{}<start_of_turn>user\n{}<end_of_turn><start_of_turn>model\n",
        end_token, prompt
    )
}

#[cfg(test)]
mod tests {
    use crabml::backends::cpu::CpuTensorDevice;
    use crabml::error::Result;
    use crabml::gguf::GGUFFileLoader;
    use crabml::tensor::TensorMetrics;

    use crate::chat::Llama2Chat;
    use crate::llama2::Llama2Runner;
    use crate::CpuLlama2Model;
    use crate::Llama2Sampler;

    #[test]
    fn test_generate_q8_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/gemma-2b.Q8_0.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::new();
        let lm = CpuLlama2Model::load(&gf, device.clone())?;

        let sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0, device.exp_cache());
        let runner = Llama2Runner::new(&lm, TensorMetrics::default(), 200, false)?;

        let mut chat = Llama2Chat::new(runner, sampler)?;
        let output = chat.chat("what's 1 + 1?")?;
        let output_vec = output.take(2).collect::<Result<Vec<String>>>()?;
        assert_eq!(output_vec, vec!["2".to_string()]);
        Ok(())
    }
}
