use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use super::tokenizer_llama::LlamaTokenEncoder;
use crate::error::Result;

pub type TokenID = usize;

pub struct BpeTokenizer {
    tokens: Rc<Vec<String>>,
    eos_token: TokenID,
    decode_buf: RefCell<Utf8Buf>,
    encoder: LlamaTokenEncoder,
}

impl BpeTokenizer {
    pub fn new(
        tokens: Vec<String>,
        token_scores: Vec<f32>,
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let tokens = Rc::new(tokens);
        let encoder = LlamaTokenEncoder::new(tokens.clone(), token_scores, bos_token, eos_token);

        Self {
            tokens,
            decode_buf: RefCell::new(Utf8Buf::new()),
            eos_token,
            encoder,
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
        // get the token string from the tokens table
        let piece: &[u8] = self.tokens[token].as_bytes();

        // some tokens designate raw bytes, and look like e.g. '<0x01>', we need parse this and
        // convert and return the actual byte.
        // this is a bit of a hack, the byte itself might not be a valid utf8 character, we need append
        // it to the decode_buf until we have a valid utf8 string, then return that. before that, we
        // return an empty string.
        let is_byte = piece.starts_with(b"<0x") && piece[piece.len() - 1] == b'>';
        if is_byte {
            let s = String::from_utf8_lossy(&piece[1..piece.len() - 1]);
            let byte = u8::from_str_radix(s.trim_start_matches("0x"), 16).unwrap();
            if self.decode_buf.borrow_mut().push_with_check(&[byte]) {
                Ok(self.decode_buf.borrow_mut().take())
            } else {
                Ok("".to_string())
            }
        } else {
            // it's considered a normal token, if the decode_buf is not empty, we need to concatenate
            // the charactors of the current token to the decode_buf, and then return the decode_buf
            // in a utf8 string.
            self.decode_buf.borrow_mut().push(piece);
            let mut s = self.decode_buf.borrow_mut().take();
            s = s.replace('▁', " ");
            Ok(s)
        }
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<TokenID>> {
        Ok(self.encoder.encode(text, bos, eos))
    }
}

/// on the cases that a utf-8 character is split into multiple tokens, we need to buffer the tokens
/// until we have a valid utf-8 string, then return it.
struct Utf8Buf {
    buf: Vec<u8>,
}

impl Utf8Buf {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
        }
    }

    fn push(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes)
    }

    fn push_with_check(&mut self, bytes: &[u8]) -> bool {
        self.buf.extend_from_slice(bytes);
        std::str::from_utf8(&self.buf).is_ok()
    }

    fn take(&mut self) -> String {
        let s = String::from_utf8_lossy(&self.buf).to_string();
        self.buf.clear();
        s
    }
}

enum TokenizerKind {
    Llama,
    GPT2,
}
