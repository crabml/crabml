use std::rc::Rc;

use super::tokenizer_gpt2::Gpt2Tokenizer;
use super::tokenizer_llama::LlamaTokenizer;
use crate::error::Result;

pub type TokenID = usize;

pub struct Tokenizer {
    tokens: Rc<Vec<String>>,
    eos_token: TokenID,
    inner: TokenizerInner,
}

pub enum TokenizerInner {
    Llama(LlamaTokenizer),
    GPT2(Gpt2Tokenizer),
}

impl Tokenizer {
    pub fn new(
        tokens: Vec<String>,
        token_scores: Vec<f32>,
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let tokens = Rc::new(tokens);
        let tokenizer = LlamaTokenizer::new(tokens.clone(), token_scores, bos_token, eos_token);

        Self {
            tokens,
            eos_token,
            inner: TokenizerInner::Llama(tokenizer),
        }
    }

    pub fn vocab(&self) -> &[String] {
        &self.tokens
    }

    pub fn eos_token(&self) -> TokenID {
        self.eos_token
    }

    pub fn token(&self, token_id: TokenID) -> String {
        self.tokens[token_id].clone()
    }

    pub fn decode(&self, token: usize) -> Result<String> {
        match &self.inner {
            TokenizerInner::Llama(inner) => inner.decode(token),
            TokenizerInner::GPT2(inner) => Ok(inner.decode(&[token])),
        }
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<TokenID>> {
        match &self.inner {
            TokenizerInner::Llama(inner) => Ok(inner.encode(text, bos, eos, true)),
            TokenizerInner::GPT2(inner) => Ok(inner.encode(text, bos, eos, true)),
        }
    }
}

enum TokenizerKind {
    Llama,
    GPT2,
}

/// on the cases that a utf-8 character is split into multiple tokens, we need to buffer the tokens
/// until we have a valid utf-8 string, then return it.
pub struct Utf8Buf {
    buf: Vec<u8>,
}

impl Utf8Buf {
    pub fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
        }
    }

    pub fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes)
    }

    pub fn push_with_check(&mut self, bytes: &[u8]) -> bool {
        self.buf.extend_from_slice(bytes);
        std::str::from_utf8(&self.buf).is_ok()
    }

    pub fn take(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).to_string();
        self.buf.clear();
        s
    }
}
