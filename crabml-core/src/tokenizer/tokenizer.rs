use std::cell::RefCell;
use std::rc::Rc;

use super::tokenizer_gpt2::Gpt2Tokenizer;
use super::tokenizer_llama::LlamaTokenizer;
use crate::error::Result;

pub type TokenID = usize;

pub struct Tokenizer {
    tokens: Rc<Vec<String>>,
    eos_token: TokenID,
    inner: TokenizerInner,
    utf8_buf: RefCell<Utf8Buf>,
}

pub enum TokenizerInner {
    Llama(LlamaTokenizer),
    GPT2(Gpt2Tokenizer),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TokenizerKind {
    Llama,
    GPT2,
}

impl Tokenizer {
    /// if TokenizerKind is Llama, we need to provide scores, if GPT2, we need to provide merges.
    pub fn new_llama(
        tokens: Vec<String>,
        scores: Vec<f32>,
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let tokens = Rc::new(tokens);
        let decode_buf = RefCell::new(Utf8Buf::new());
        let inner = TokenizerInner::Llama(LlamaTokenizer::new(
            tokens.clone(),
            scores,
            bos_token,
            eos_token,
        ));

        Self {
            tokens,
            eos_token,
            utf8_buf: decode_buf,
            inner,
        }
    }

    pub fn new_gpt2(
        tokens: Vec<String>,
        merges: Vec<String>,
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let tokens = Rc::new(tokens);
        let decode_buf = RefCell::new(Utf8Buf::new());
        let inner = TokenizerInner::GPT2(Gpt2Tokenizer::new(
            tokens.clone(),
            &merges,
            bos_token,
            eos_token,
        ));
        Self {
            tokens,
            eos_token,
            utf8_buf: decode_buf,
            inner,
        }
    }

    pub fn kind(&self) -> TokenizerKind {
        match &self.inner {
            TokenizerInner::Llama(_) => TokenizerKind::Llama,
            TokenizerInner::GPT2(_) => TokenizerKind::GPT2,
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

    /// TODO: make it consume an Interator<Item=Result<TokenID>>
    pub fn decode(&self, token: TokenID) -> Result<String> {
        let bytes = match &self.inner {
            TokenizerInner::Llama(inner) => inner.decode(token),
            TokenizerInner::GPT2(inner) => inner.decode(token),
        };
        Ok(self.utf8_buf.borrow_mut().step(&bytes))
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

    pub fn is_valid(&self) -> bool {
        std::str::from_utf8(&self.buf).is_ok()
    }

    pub fn step(&mut self, bytes: &[u8]) -> String {
        let is_utf8 = std::str::from_utf8(bytes).is_ok();
        if is_utf8 {
            self.buf.extend(bytes);
            let s = String::from_utf8_lossy(&self.buf).to_string();
            self.buf.clear();
            return s;
        }

        self.buf.extend(bytes);
        if self.is_valid() || self.buf.len() >= 4 {
            let s = String::from_utf8_lossy(&self.buf).to_string();
            self.buf.clear();
            return s;
        }

        "".to_string()
    }
}
