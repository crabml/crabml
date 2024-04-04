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

    pub fn chat<'a>(&'a mut self, prompt: &str) -> Llama2Dialogue<'a, T> {
        Llama2Dialogue {
            inner: self,
            prompt: prompt.to_string(),
            stats: Default::default(),
        }
    }
}

struct Llama2Dialogue<'a, T: Tensor> {
    inner: &'a mut Llama2Chat<T>,
    prompt: String,
    stats: Llama2DialogueStats,
}

impl<'a, T: Tensor> Llama2Dialogue<'a, T> {
    pub fn iter(&mut self) -> Result<Llama2DialogueIterator> {
        let prompt = wrap_prompt(&self.prompt);
        let (pos, last_token, token) =
            self.inner
                .runner
                .prefill(&prompt, &mut self.inner.sampler, false, false)?;
        let iter =
            self.inner
                .runner
                .generate(pos, last_token, token, None, &mut self.inner.sampler);
        let chat_iter =
            Llama2DialogueIterator::new(Box::new(iter), "<end_of_turn>", &mut self.stats);
        Ok(chat_iter)
    }

    pub fn finish(&mut self) -> Result<()> {
        if !self.stats.has_end_mark {
            self.inner
                .runner
                .prefill("<end_of_turn>", &mut self.inner.sampler, false, false)?;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct Llama2DialogueStats {
    has_end_mark: bool,
}

/// each dialog has a start mark and an end mark. The chat iterator will
/// take the generation result from the model and concatenate them until
/// got the end mark, like "<end_of_turn>".
/// on some cases the model may not generate the end mark, so we need to
/// tell the iterator is finished by end mark or not.
struct Llama2DialogueIterator<'a> {
    inner: Box<dyn Iterator<Item = Result<String>> + 'a>,
    buf: String,
    end_mark: String,
    stats: &'a mut Llama2DialogueStats,
}

impl<'a> Llama2DialogueIterator<'a> {
    pub fn new(
        inner: Box<dyn Iterator<Item = Result<String>> + 'a>,
        end_mark: &str,
        stats: &'a mut Llama2DialogueStats,
    ) -> Self {
        Self {
            inner,
            buf: String::new(),
            end_mark: end_mark.to_string(),
            stats,
        }
    }
}

impl<'a> Iterator for Llama2DialogueIterator<'a> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stats.has_end_mark {
            return None;
        }

        let token = match self.inner.next() {
            None => return None,
            Some(Err(err)) => return Some(Err(err)),
            Some(Ok(token)) => token,
        };

        self.buf.push_str(&token);
        if self.buf.ends_with(&self.end_mark) {
            self.stats.has_end_mark = true;
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
        let mut dialogue = chat.chat("what's 1+1?");
        let output = dialogue.iter()?;
        for token in output {
            print!("{}", token?);
        }
        Ok(())
    }
}
