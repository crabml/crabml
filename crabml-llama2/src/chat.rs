use crabml::error::Result;
use crabml::tensor::Tensor;

use crate::llama2::Llama2Runner;
use crate::model::ModelArchitecture;

pub struct Llama2Chat<'a, T: Tensor> {
    inner: &'a mut Llama2Runner<T>,
    prompt: String,
    system_prompt: Option<String>,
    stats: Llama2ChatReplyIteratorStats,
    chat_template: ChatTemplate,
}

impl<'a, T: Tensor> Llama2Chat<'a, T> {
    pub fn new(
        runner: &'a mut Llama2Runner<T>,
        prompt: impl Into<String>,
        system_prompt: Option<String>,
    ) -> Result<Self> {
        let model_name = &runner.conf().model_name;
        let model_arch = runner.conf().architecture;
        let chat_template = ChatTemplate::heuristic_guess(model_name, model_arch, "")?;
        Ok(Self {
            inner: runner,
            prompt: prompt.into(),
            system_prompt,
            stats: Default::default(),
            chat_template,
        })
    }

    pub fn reply(&mut self) -> Result<Llama2ChatReplyIterator> {
        let templated_prompt =
            self.chat_template
                .apply(&self.prompt, self.system_prompt.as_deref(), true);

        let bos = self.inner.kv_cache_len() == 0;
        let (pos, last_token, token) = self.inner.prefill(&templated_prompt, bos, false)?;
        let iter = self.inner.generate(pos, last_token, token, None);
        let chat_iter = Llama2ChatReplyIterator::new(
            Box::new(iter),
            vec![self.chat_template.stop_mark().to_string()],
            &mut self.stats,
        );
        Ok(chat_iter)
    }

    /// the reply might ended with <eos>, but not <end_of_turn>, so we need to append the <end_of_turn>
    pub fn finish(&mut self) -> Result<()> {
        if !self.stats.has_stop_mark {
            self.inner
                .prefill(self.chat_template.stop_mark(), false, false)?;
        }

        Ok(())
    }
}

/// Llama2ChatReplyIteratorStats is used to return some useful infomation
/// after/during the execution of iterator.
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

/// the model response might ends with stop mark like "<end_of_turn>", please note
/// these stop mark are NOT a single token. we'd use a simple state machine to track
/// wheter we've entered these stop marks, and merge these tokens in the stop mark
/// to a single String.
pub struct MarkMatcher {
    state: MarkMatchState,
    buf: String,
    marks: Vec<String>,
}

pub enum MarkMatchState {
    Inactive,
    Active,
}

impl MarkMatcher {
    pub fn new(marks: Vec<String>) -> Self {
        Self {
            state: MarkMatchState::Inactive,
            buf: String::new(),
            marks,
        }
    }

    pub fn push(&mut self, token: String) -> Option<String> {
        match self.state {
            MarkMatchState::Inactive => {
                // exact match, do not change state
                if self.marks.contains(&token) {
                    return Some(token);
                }

                // got any partial match, change state to active, and push the token
                // to the buffer, and wait for the rest of the mark.
                if self.marks.iter().any(|m| m.starts_with(&token)) {
                    self.state = MarkMatchState::Active;
                    self.buf = token;
                    return None;
                }

                // no match, return the token directly
                Some(token)
            }
            MarkMatchState::Active => {
                self.buf.push_str(&token);

                // exact match, change state to inactive, and return the buffer
                if self.marks.contains(&self.buf) {
                    self.state = MarkMatchState::Inactive;
                    return Some(self.buf.clone());
                }

                // not match anymore, return the buffer directly
                if !self.marks.iter().any(|m| m.starts_with(&self.buf)) {
                    self.state = MarkMatchState::Inactive;
                    return Some(self.buf.clone());
                }

                // partial match, wait for the rest of the mark
                None
            }
        }
    }
}

/// buildin the commonly used chat templates inside the code.
/// TODO: support customized template, it might need some template engine.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ChatTemplate {
    Llama2,
    Gemma,
}

impl ChatTemplate {
    /// GGUF may contains a metadata called tokenizer.chat_template (maybe in a jinja format),
    /// we'd not take the chat_template directly but use a heuristic to guess the common ones.
    fn heuristic_guess(
        model_name: &str,
        model_arch: ModelArchitecture,
        _chat_tmpl_meta: &str,
    ) -> Result<Self> {
        if model_name.contains("gemma") || model_arch == ModelArchitecture::Gemma {
            Ok(ChatTemplate::Gemma)
        } else if model_name.contains("llama2") {
            Ok(ChatTemplate::Llama2)
        } else {
            // take llama2 as fallback.
            Ok(ChatTemplate::Llama2)
        }
    }

    fn stop_mark(&self) -> &str {
        match self {
            ChatTemplate::Llama2 => "[/INST]",
            ChatTemplate::Gemma => "<end_of_turn>",
        }
    }

    fn apply(
        &self,
        prompt: &str,
        system_prompt: Option<&str>,
        append_assistant_prefix: bool,
    ) -> String {
        match *self {
            ChatTemplate::Llama2 => {
                let system_prompt = system_prompt
                    .map(|s| format!("<<SYS>>{}<</SYS>>", s))
                    .unwrap_or("".to_string());
                let assistant_prefix = if append_assistant_prefix {
                    "[[INST]]"
                } else {
                    ""
                };
                format!(
                    "[INST] {} {} [/INST]{}",
                    system_prompt, prompt, assistant_prefix
                )
            }
            ChatTemplate::Gemma => {
                let system_prompt = system_prompt.unwrap_or("");
                let assistant_prefix = match append_assistant_prefix {
                    true => "<start_of_turn>model\n",
                    false => "",
                };
                format!(
                    "<start_of_turn>user\n{} {}<end_of_turn>{}",
                    system_prompt, prompt, assistant_prefix
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crabml::error::Result;
    use crabml::gguf::GGUFFileLoader;

    use crate::chat::Llama2Chat;
    use crate::llama2::Llama2Runner;
    use crate::model::CpuLlama2ModelLoader;

    #[test]
    #[ignore]
    fn test_generate_q8_0() -> Result<()> {
        let gl = GGUFFileLoader::new("../testdata/gemma-2b-it-q8_0.gguf", false)?;
        let gf = gl.open()?;

        let lm = CpuLlama2ModelLoader::new().load(&gf)?;

        let mut runner = Llama2Runner::new(&lm, 200, false)?;
        let mut chat = Llama2Chat::new(&mut runner, "what's 1+1?", None)?;
        let output = chat.reply()?;
        for token in output {
            print!("{}", token?);
        }
        Ok(())
    }
}
