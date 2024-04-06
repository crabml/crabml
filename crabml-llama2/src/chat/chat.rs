use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use crabml::tensor::Tensor;

use super::mark_matcher::MarkMatcher;
use crate::llama2::Llama2Runner;

pub struct Llama2Chat<'a, T: Tensor> {
    inner: &'a mut Llama2Runner<T>,
    prompt: String,
    stats: Llama2ChatReplyIteratorStats,
    chat_template: ChatTemplateKind,
}

impl<'a, T: Tensor> Llama2Chat<'a, T> {
    pub fn new(runner: &'a mut Llama2Runner<T>, prompt: &str) -> Result<Self> {
        let model_name = &runner.conf().model_name;
        let chat_template = ChatTemplateKind::heuristic_guess(model_name, "")?;
        Ok(Self {
            inner: runner,
            prompt: prompt.to_string(),
            stats: Default::default(),
            chat_template,
        })
    }

    pub fn reply(&mut self) -> Result<Llama2ChatReplyIterator> {
        let templated_prompt = wrap_chat_template(self.chat_template, &self.prompt, None, true);

        let bos = self.inner.kv_cache_len() == 0;
        let (pos, last_token, token) = self.inner.prefill(&templated_prompt, bos, false)?;
        let iter = self.inner.generate(pos, last_token, token, None);
        let chat_iter = Llama2ChatReplyIterator::new(
            Box::new(iter),
            vec!["<end_of_turn>".to_string()],
            &mut self.stats,
        );
        Ok(chat_iter)
    }

    /// the reply might ended with <eos>, but not <end_of_turn>, so we need to append the <end_of_turn>
    pub fn finish(&mut self) -> Result<()> {
        if !self.stats.has_stop_mark {
            self.inner.prefill("<end_of_turn>", false, false)?;
        }

        Ok(())
    }
}

#[derive(Debug, Default)]
struct Llama2ChatReplyIteratorStats {
    has_stop_mark: bool,
}

/// each dialog has a start mark and an end mark. The chat iterator will
/// take the generation result from the model and concatenate them until
/// got the end mark, like "<end_of_turn>".
/// on some cases the model may not generate the end mark, so we need to
/// tell the iterator is finished by end mark or not.
pub struct Llama2ChatReplyIterator<'a> {
    inner: Box<dyn Iterator<Item = Result<String>> + 'a>,
    stop_mark_matcher: MarkMatcher,
    stop_marks: Vec<String>,
    stats: &'a mut Llama2ChatReplyIteratorStats,
}

impl<'a> Llama2ChatReplyIterator<'a> {
    fn new(
        inner: Box<dyn Iterator<Item = Result<String>> + 'a>,
        stop_marks: Vec<String>,
        stats: &'a mut Llama2ChatReplyIteratorStats,
    ) -> Self {
        Self {
            inner,
            stats,
            stop_marks: stop_marks.clone(),
            stop_mark_matcher: MarkMatcher::new(stop_marks),
        }
    }
}

impl<'a> Iterator for Llama2ChatReplyIterator<'a> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.stats.has_stop_mark {
            return None;
        }

        let token = match self.inner.next() {
            None => return None,
            Some(Err(err)) => return Some(Err(err)),
            Some(Ok(token)) => token,
        };

        let token = match self.stop_mark_matcher.push(token) {
            None => return Some(Ok("".to_string())),
            Some(token) => token,
        };

        if self.stop_marks.contains(&token) {
            self.stats.has_stop_mark = true;
            return None;
        }

        Some(Ok(token))
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ChatTemplateKind {
    Llama2,
    Gemma,
}

impl ChatTemplateKind {
    fn heuristic_guess(model_name: &str, _chat_tmpl_meta: &str) -> Result<Self> {
        if model_name.contains("gemma") {
            Ok(ChatTemplateKind::Gemma)
        } else if model_name.contains("llama2") {
            Ok(ChatTemplateKind::Llama2)
        } else {
            Err(Error::new(
                ErrorKind::ChatTemplateNotFound,
                format!("chat template for {} is not supported yet", model_name),
            ))
        }
    }
}

fn wrap_chat_template(
    tmpl: ChatTemplateKind,
    prompt: &str,
    system_prompt: Option<&str>,
    append_assistant_prefix: bool,
) -> String {
    match tmpl {
        ChatTemplateKind::Llama2 => {
            let system_prompt = system_prompt
                .map(|s| format!("<<SYS>>{}<</SYS>>", s))
                .unwrap_or("".to_string());
            let assistant_prefix = append_assistant_prefix.then(|| "[[INST]]").unwrap_or("");
            format!(
                "[INST] {} {} [/INST]{}",
                system_prompt, prompt, assistant_prefix
            )
        }
        ChatTemplateKind::Gemma => {
            let system_prompt = system_prompt.map(|s| s).unwrap_or("");
            let assistant_prefix = append_assistant_prefix
                .then(|| "<start_of_turn>model\n")
                .unwrap_or("");
            format!(
                "<start_of_turn>user\n{} {}<end_of_turn>{}",
                system_prompt, prompt, assistant_prefix
            )
        }
    }
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

        let mut chat = Llama2Chat::new(&mut runner, "what's 1+1?")?;
        let output = chat.reply()?;
        for token in output {
            print!("{}", token?);
        }
        Ok(())
    }
}
