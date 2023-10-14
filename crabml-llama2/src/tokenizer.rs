use crabml::error::Result;
use std::collections::HashMap;

pub struct Llama2Tokenizer {
    vocab: Vec<String>,
    vocab_scores: Vec<f32>,
    token_buf_len: usize,
    byte_pieces: [u8; 256],
    vocab_index: HashMap<String, usize>,
    bos_token: usize,
    eos_token: usize,
}

impl Llama2Tokenizer {
    pub fn new(
        vocab: Vec<String>,
        vocab_scores: Vec<f32>,
        token_buf_len: usize,
        bos_token: usize,
        eos_token: usize,
    ) -> Self {
        let vocab_index = vocab
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect();

        let mut byte_pieces = [0u8; 256];
        for (i, p) in byte_pieces.iter_mut().enumerate() {
            *p = i as u8
        }

        Self {
            vocab,
            vocab_index,
            vocab_scores,
            token_buf_len,
            byte_pieces,
            bos_token,
            eos_token,
        }
    }

    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    pub fn decode(&self, prev_token: usize, token: usize) -> Result<String> {
        let mut piece: &[u8] = self.vocab[token].as_bytes();
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token == 1 && piece[0] == b' ' {
            piece = &piece[1..];
        }
        // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
        // parse this and convert and return the actual byte
        if piece.starts_with(b"<0x") && piece[piece.len() - 1] == b'>' {
            let s = String::from_utf8_lossy(&piece[1..piece.len() - 1]);
            let s = s.trim_start_matches("0x");
            if let Ok(byte) = u8::from_str_radix(s, 16) {
                piece = &self.byte_pieces[(byte as usize)..(byte as usize) + 1]
            }
        }

        let mut s = String::from_utf8(piece.to_vec()).unwrap();
        s = s.replace('▁', " ");
        Ok(s)
    }

    #[allow(dead_code)]
    pub fn decode_string(&self, tokens: &[usize]) -> Result<String> {
        let mut result = String::new();
        let mut prev_token = 0;
        for token in tokens {
            let piece = self.decode(prev_token, *token)?;
            result.push_str(&piece);
            prev_token = *token;
        }
        Ok(result)
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Result<Vec<usize>> {
        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        let mut token_buf = String::with_capacity(self.token_buf_len * 2 + 1 + 2);
        let mut tokens: Vec<usize> = vec![];

        if bos {
            tokens.push(self.bos_token);
        }

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if !text.starts_with('\u{0}') {
            if let Some(dummy_prefix) = self.vocab_index.get(" ") {
                tokens.push(*dummy_prefix);
            }
        }

        let chars = text.chars();
        for ch in chars {
            token_buf.clear();
            token_buf.push(ch);
            if let Some(tok) = self.vocab_index.get(&token_buf) {
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
                token_buf.push_str(&self.vocab[tokens[i]]);
                token_buf.push_str(&self.vocab[tokens[i + 1]]);
                if let Some(tok) = self.vocab_index.get(&token_buf) {
                    let new_score = self.vocab_scores[*tok];
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
