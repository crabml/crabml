use std::cell::RefCell;
use std::collections::HashMap;

use crate::error::Result;

type Token = String;
type TokenID = usize;

pub struct BpeTokenizer {
    tokens: Vec<Token>,
    token_scores: Vec<f32>,
    token_ids: HashMap<String, TokenID>,
    bos_token: TokenID,
    eos_token: TokenID,
    decode_buf: RefCell<Utf8Buf>,
    token_buf_len: usize,
}

impl BpeTokenizer {
    pub fn new(
        tokens: Vec<String>,
        token_scores: Vec<f32>,
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let token_ids = tokens
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        Self {
            tokens,
            token_ids,
            token_scores,
            token_buf_len: 128,
            decode_buf: RefCell::new(Utf8Buf::new()),
            bos_token,
            eos_token,
        }
    }

    pub fn vocab(&self) -> &[String] {
        &self.tokens
    }

    pub fn eos_token(&self) -> TokenID {
        self.eos_token
    }

    pub fn token(&self, token_id: TokenID) -> Token {
        self.tokens[token_id].clone()
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> Result<String> {
        // get the token string from the tokens table
        let mut piece: &[u8] = self.tokens[token].as_bytes();

        // following BOS (1) token, sentencepiece decoder strips any leading whitespace
        if prev_token == 1 && piece[0] == b' ' {
            piece = &piece[1..];
        }

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
        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        let mut token_buf = String::with_capacity(self.token_buf_len * 2 + 1 + 2);
        let mut tokens: Vec<TokenID> = vec![];

        let text = text.replace(' ', "▁");

        if bos {
            tokens.push(self.bos_token);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if !text.starts_with('\u{0}') {
            if let Some(dummy_prefix) = self.token_ids.get("▁") {
                tokens.push(*dummy_prefix);
            }
        }

        let chars = text.chars();
        for ch in chars {
            token_buf.clear();
            token_buf.push(ch);
            if let Some(tok) = self.token_ids.get(&token_buf) {
                // we found this codepoint in vocab, add it as a token
                tokens.push(*tok);
            } else {
                // byte_fallback encoding: just encode each byte as a token
                // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
                // so the individual bytes only start at index 3
                for byte in token_buf.bytes() {
                    tokens.push(byte as usize + 3);
                }
            }
        }

        // merge the best consecutive pair each iteration, according the scores in vocab_scores
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx: Option<usize> = None;
            let mut best_token: Option<usize> = None;
            let mut i = 0;

            while i < (tokens.len() - 1) {
                token_buf.clear();
                token_buf.push_str(&self.tokens[tokens[i]]);
                token_buf.push_str(&self.tokens[tokens[i + 1]]);
                if let Some(tok) = self.token_ids.get(&token_buf) {
                    let new_score = self.token_scores[*tok];
                    if new_score > best_score {
                        best_score = new_score;
                        best_idx = Some(i);
                        best_token = Some(*tok);
                    }
                }
                i += 1;
            }

            if let Some(idx) = best_idx {
                tokens[idx] = best_token.unwrap();
                tokens.remove(idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(self.eos_token);
        }

        Ok(tokens)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GGUFFileLoader;

    #[test]
    fn test_gguf_tokenizer() -> Result<()> {
        let gf_loader = GGUFFileLoader::new("../testdata/tinyllamas-stories-15m-f32.gguf", false)?;
        let gf = gf_loader.open()?;

        let tokens = gf
            .metadata()
            .get_string_array("tokenizer.ggml.tokens")
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let token_scores = gf
            .metadata()
            .get_f32_array("tokenizer.ggml.scores")
            .unwrap()
            .to_vec();
        let tk = BpeTokenizer::new(tokens, token_scores, 1, 2);

        let tests = vec![
            (10842, "▁Captain"),
            (6813, "▁America"),
            (29901, ":"),
            (29871, "▁"),
            (260, "▁t"),
            (10373, "ictures"),
            (287, "ed"),
            (259, "▁▁"),
            (1218, "ating"),
        ];
        for (token_id, token) in tests {
            let got = tk.token(token_id);
            assert_eq!(token, got);
        }

        let tests = vec![
            (
                "Captain America: ",
                "<s> - ▁Captain - ▁America - : - ▁ - </s>",
            ),
            ("hello, world", "<s> - ▁hello - , - ▁world - </s>"),
            ("tiktok", "<s> - ▁t - ik - tok - </s>"),
        ];

        for tt in tests {
            let tokens = tk.encode(tt.0, true, true)?;
            let tokens_in_string = tokens
                .iter()
                .map(|t| tk.vocab()[*t].clone())
                .collect::<Vec<String>>()
                .join(" - ");
            assert_eq!(tokens_in_string, tt.1, "failed to encode {}", tt.0);
        }
        Ok(())
    }
}
