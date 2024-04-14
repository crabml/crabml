use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::error::Result;

type TokenID = usize;

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
        let token_ids = Rc::new(
            tokens
                .iter()
                .enumerate()
                .map(|(i, v)| (v.clone(), i))
                .collect(),
        );
        let tokens = Rc::new(tokens);
        let encoder = LlamaTokenEncoder {
            tokens: tokens.clone(),
            token_ids: token_ids,
            token_scores: Rc::new(
                token_scores
                    .into_iter()
                    .enumerate()
                    .collect::<HashMap<_, _>>(),
            ),
            token_buf_len: 128,
            bos_token,
            eos_token,
        };

        Self {
            tokens: tokens,
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

enum TokenDecoderKind {
    Llama,
    GPT2,
}

pub enum TokenDecoder {
    Llama(LlamaTokenEncoder),
}

struct LlamaTokenEncoder {
    tokens: Rc<Vec<String>>,
    token_ids: Rc<HashMap<String, TokenID>>,
    token_scores: Rc<HashMap<TokenID, f32>>,
    token_buf_len: usize,
    bos_token: TokenID,
    eos_token: TokenID,
}

impl LlamaTokenEncoder {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<TokenID> {
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
        if text.len() > 0 {
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
                    let new_score = *self.token_scores.get(tok).unwrap();
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

        tokens
    }
}

struct Gpt2TokenEncoder {
    tokens: Rc<Vec<String>>,
    token_ids: Rc<HashMap<String, TokenID>>,
    merges: HashMap<(TokenID, TokenID), usize>,
    bos_token: TokenID,
    eos_token: TokenID,
}

impl Gpt2TokenEncoder {
    fn new(
        tokens: Rc<Vec<String>>,
        merges: &[String],
        bos_token: TokenID,
        eos_token: TokenID,
    ) -> Self {
        let token_ids: Rc<HashMap<String, TokenID>> = Rc::new(
            tokens
                .iter()
                .enumerate()
                .map(|(i, v)| (v.clone(), i))
                .collect(),
        );
        let merges = merges
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let parts = s.split(' ').collect::<Vec<_>>();
                let first = token_ids.get(parts[0]).unwrap();
                let second = token_ids.get(parts[1]).unwrap();
                ((*first, *second), i)
            })
            .collect();
        Self {
            tokens,
            token_ids,
            merges,
            bos_token,
            eos_token,
        }
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<TokenID> {
        // create a temporary buffer that will store merge candidates of always two consecutive tokens
        // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
        let mut token_buf = String::with_capacity(128 * 2 + 1 + 2);
        let mut tokens: Vec<TokenID> = vec![];

        // let text = text.replace(' ', "▁");

        if bos {
            tokens.push(self.bos_token);
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

        // merge the best consecutive pair each iteration, according the merges
        loop {
            let mut lowest_rank = usize::MAX;
            let mut merging_pair: Option<(TokenID, TokenID)> = None;
            let mut merging_idx = usize::MAX;
            for (idx, pair) in tokens.windows(2).enumerate() {
                if let Some(rank) = self.merges.get(&(pair[0], pair[1])) {
                    if *rank < lowest_rank {
                        lowest_rank = *rank;
                        merging_pair = Some((pair[0], pair[1]));
                        merging_idx = idx;
                    }
                    println!("idx: {}", idx);
                }
            }
            if let Some((tok1, tok2)) = merging_pair {
                let token1 = self.tokens[tok1].clone();
                let token2 = self.tokens[tok2].clone();
                tokens[merging_idx] = self
                    .token_ids
                    .get(&format!("{}{}", token1, token2))
                    .unwrap()
                    .clone();
                tokens.remove(merging_idx + 1);
            } else {
                break;
            }
        }

        if eos {
            tokens.push(self.eos_token);
        }

        tokens
    }
}

fn is_cjk_char(character: &char) -> bool {
    let u32_char = *character as u32;
    (0x4E00..=0x9FFF).contains(&u32_char)
        | (0x3400..=0x4DBF).contains(&u32_char)
        | (0x20000..=0x2A6DF).contains(&u32_char)
        | (0x2A700..=0x2B73F).contains(&u32_char)
        | (0x2B740..=0x2B81F).contains(&u32_char)
        | (0x2B820..=0x2CEAF).contains(&u32_char)
        | (0xF900..=0xFAFF).contains(&u32_char)
        | (0x2F800..=0x2FA1F).contains(&u32_char)
}

pub const WHITESPACE_CHARS: [u32; 20] = [
    //        Standard whitespace characters (unicode category Zs)
    0x0020, 0x00A0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008,
    0x2009, 0x200A, 0x202F, 0x205F, 0x3000,
    //        Additional characters considered whitespace for BERT (tab, newline, carriage return)
    0x0009, 0x000D, 0x00A,
];

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
        for (i, tok) in tokens.iter().enumerate().take(100) {
            println!("{} {}", i, tok);
        }
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
            (
                "i don't eat beaf.",
                "<s> - ▁i - ▁don - ' - t - ▁eat - ▁be - af - . - </s>",
            ),
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

    #[test]
    fn test_gpt2_tokenizer() -> Result<()> {
        let gf_loader =
            GGUFFileLoader::new("/Users/yazhou/llm/qwen1_5-0_5b-chat-q8_0.gguf", false)?;
        let gf = gf_loader.open()?;

        let tokens = Rc::new(
            gf.metadata()
                .get_string_array("tokenizer.ggml.tokens")
                .unwrap()
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>(),
        );
        let merges = gf
            .metadata()
            .get_string_array("tokenizer.ggml.merges")
            .unwrap()
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>();
        let tk = Gpt2TokenEncoder::new(tokens.clone(), &merges, 1, 2);

        let tests = vec![
            (
                "Captain America: ",
                "<s> - ▁Captain - ▁America - : - ▁ - </s>",
            ),
            ("hello, world", "<s> - ▁hello - , - ▁world - </s>"),
            ("tiktok", "<s> - ▁t - ik - tok - </s>"),
            (
                "i don't eat beaf.",
                "<s> - ▁i - ▁don - ' - t - ▁eat - ▁be - af - . - </s>",
            ),
        ];

        for tt in tests {
            let outputs = tk.encode(tt.0, false, false);
            let tokens_in_string = outputs
                .iter()
                .map(|t| tokens[*t].clone())
                .collect::<Vec<String>>()
                .join(" - ");
            assert_eq!(tokens_in_string, tt.1, "failed to encode {}", tt.0);
        }

        Ok(())
    }
}
