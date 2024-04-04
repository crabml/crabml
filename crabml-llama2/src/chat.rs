use crabml::error::Result;
use crabml::tensor::Tensor;

use crate::llama2::Llama2Runner;

struct Llama2Chat<'a, T: Tensor> {
    inner: &'a mut Llama2Runner<T>,
    prompt: String,
    bos: bool,
    stats: Llama2ChatStats,
}

impl<'a, T: Tensor> Llama2Chat<'a, T> {
    pub fn new(runner: &'a mut Llama2Runner<T>, prompt: &str, bos: bool) -> Self {
        Self {
            inner: runner,
            prompt: prompt.to_string(),
            stats: Default::default(),
            bos,
        }
    }

    pub fn reply(&mut self) -> Result<Llama2ChatReplyIterator> {
        let prompt = wrap_prompt(&self.prompt);
        let (pos, last_token, token) = self.inner.prefill(&prompt, self.bos, false)?;
        let iter = self.inner.generate(pos, last_token, token, None);
        let chat_iter =
            Llama2ChatReplyIterator::new(Box::new(iter), "<end_of_turn>", &mut self.stats);
        Ok(chat_iter)
    }

    pub fn finish(&mut self) -> Result<()> {
        if !self.stats.has_end_mark {
            self.inner.prefill("<end_of_turn>", false, false)?;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct Llama2ChatStats {
    has_end_mark: bool,
}

/// each dialog has a start mark and an end mark. The chat iterator will
/// take the generation result from the model and concatenate them until
/// got the end mark, like "<end_of_turn>".
/// on some cases the model may not generate the end mark, so we need to
/// tell the iterator is finished by end mark or not.
struct Llama2ChatReplyIterator<'a> {
    inner: Box<dyn Iterator<Item = Result<String>> + 'a>,
    buf: String,
    end_mark: String,
    stats: &'a mut Llama2ChatStats,
}

impl<'a> Llama2ChatReplyIterator<'a> {
    pub fn new(
        inner: Box<dyn Iterator<Item = Result<String>> + 'a>,
        end_mark: &str,
        stats: &'a mut Llama2ChatStats,
    ) -> Self {
        Self {
            inner,
            buf: String::new(),
            end_mark: end_mark.to_string(),
            stats,
        }
    }
}

impl<'a> Iterator for Llama2ChatReplyIterator<'a> {
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
    use std::rc::Rc;

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

        let sampler = Rc::new(Llama2Sampler::new(
            lm.conf.vocab_size,
            0.0,
            0.0,
            device.exp_cache(),
        ));
        let mut runner = Llama2Runner::new(&lm, sampler, TensorMetrics::default(), 200, false)?;

        let mut chat = Llama2Chat::new(&mut runner, "what's 1+1?", true);
        let output = chat.reply()?;
        for token in output {
            print!("{}", token?);
        }
        Ok(())
    }
}
