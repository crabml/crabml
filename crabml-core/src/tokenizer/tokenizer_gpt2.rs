use std::collections::HashMap;
use std::rc::Rc;

use super::tokenizer::TokenID;

struct Gpt2Tokenizer {
    tokens: Rc<Vec<String>>,
    token_ids: Rc<HashMap<String, TokenID>>,
    bpe_ranks: HashMap<(TokenID, TokenID), usize>,
    byte_encode_map: HashMap<u8, char>,
    bos_token: TokenID,
    eos_token: TokenID,
}

impl Gpt2Tokenizer {
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
        let byte_encode_map = build_byte_encode_map();
        Self {
            tokens,
            token_ids,
            bpe_ranks: merges,
            byte_encode_map,
            bos_token,
            eos_token,
        }
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool) -> Vec<TokenID> {
        let tokens = text
            .bytes()
            .map(|b| {
                let ch = self.byte_encode_map.get(&b).unwrap().to_string();
                let token_id = self.token_ids.get(&ch).unwrap();
                *token_id
            })
            .collect::<Vec<_>>();
        let mut tokens = self.bpe_merge(tokens);

        if bos {
            tokens.insert(0, self.bos_token);
        }
        if eos {
            tokens.push(self.eos_token);
        }
        tokens
    }

    fn bpe_merge(&self, mut tokens: Vec<TokenID>) -> Vec<TokenID> {
        // merge the best consecutive pair each iteration, according the merges
        loop {
            let mut lowest_rank = usize::MAX;
            let mut merging_pair: Option<(TokenID, TokenID)> = None;
            let mut merging_idx = usize::MAX;
            for (idx, pair) in tokens.windows(2).enumerate() {
                if let Some(rank) = self.bpe_ranks.get(&(pair[0], pair[1])) {
                    if *rank < lowest_rank {
                        lowest_rank = *rank;
                        merging_pair = Some((pair[0], pair[1]));
                        merging_idx = idx;
                    }
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
                return tokens;
            }
        }
    }
}

/// the merge map are all unicodes, we need convert the raw bytes into an encoded
/// unicode character.
fn build_byte_encode_map() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let ranges = [
        ('!' as u8, '~' as u8),
        ('¡' as u8, '¬' as u8),
        ('®' as u8, 'ÿ' as u8),
    ];
    for (start, end) in ranges.iter() {
        for i in *start..=*end {
            map.insert(i, i as char);
        }
    }
    let mut extra_unicode = 0x100;
    for i in 0..=255 {
        if !map.contains_key(&(i as u8)) {
            map.insert(i as u8, std::char::from_u32(extra_unicode).unwrap());
            extra_unicode += 1;
        }
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::gguf::GGUFFileLoader;

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
        let tk = Gpt2Tokenizer::new(tokens.clone(), &merges, 1, 2);

        let tests = vec![
            ("Captain America: ", "Captain - ĠAmerica - : - Ġ"),
            ("hello, world", "hello - , - Ġworld"),
            ("tiktok", "t - ik - tok"),
            ("i don't eat beaf.", "i - Ġdon - 't - Ġeat - Ġbe - af - ."),
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
