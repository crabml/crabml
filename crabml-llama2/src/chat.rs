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
    ) -> Result<ChatIterator<impl Iterator<Item = Result<String>> + '_>> {
        let prompt = wrap_prompt(prompt);
        let (pos, last_token, token) =
            self.runner
                .prefill(&prompt, &mut self.sampler, false, false)?;
        let iter = self
            .runner
            .generate(pos, last_token, token, None, &mut self.sampler);
        let chat_iter = ChatIterator::new(iter, "<end_of_turn>");
        Ok(chat_iter)
    }
}

/// each dialog has a start mark and an end mark. The chat iterator will
/// take the generation result from the model and concatenate them until
/// got the end mark, like "<end_of_turn>".
/// on some cases the model may not generate the end mark, so we need to
/// tell the iterator is finished by end mark or not.
struct ChatIterator<I: Iterator<Item = Result<String>>> {
    inner: I,
    buf: String,
    end_mark: String,
    has_end_mark: bool,
}

impl<I: Iterator<Item = Result<String>>> ChatIterator<I> {
    pub fn new(inner: I, end_mark: &str) -> Self {
        Self {
            inner,
            buf: String::new(),
            end_mark: end_mark.to_string(),
            has_end_mark: false,
        }
    }

    fn has_end_mark(&self) -> bool {
        self.has_end_mark
    }
}

impl<T: Iterator<Item = Result<String>>> Iterator for ChatIterator<T> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.has_end_mark {
            return None;
        }

        let token = match self.inner.next() {
            None => return None,
            Some(Err(err)) => return Some(Err(err)),
            Some(Ok(token)) => token,
        };

        self.buf.push_str(&token);
        if self.buf.ends_with(&self.end_mark) {
            self.has_end_mark = true;
            return None;
        }
        Some(Ok(token))
    }
}

fn wrap_prompt(prompt: &str) -> String {
    format!(
        "<start_of_turn>user\n{}<end_of_turn><start_of_turn>model\n",
        prompt
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
        let gl = GGUFFileLoader::new("../testdata/gemma-2b-it-q8_0.gguf")?;
        let gf = gl.open()?;

        let device = CpuTensorDevice::new();
        let lm = CpuLlama2Model::load(&gf, device.clone())?;

        let sampler = Llama2Sampler::new(lm.conf.vocab_size, 0.0, 0.0, device.exp_cache());
        let runner = Llama2Runner::new(&lm, TensorMetrics::default(), 200, false)?;

        let mut chat = Llama2Chat::new(runner, sampler)?;
        let output = chat.chat("how to understand spacetime curvature?")?;
        for token in output {
            print!("{}", token?);
        }
        Ok(())
    }
}
