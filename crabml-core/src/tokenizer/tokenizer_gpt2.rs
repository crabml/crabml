use std::collections::HashMap;
use std::rc::Rc;

use regex::Regex;

use super::TokenID;

pub struct Gpt2Tokenizer {
    tokens: Rc<Vec<String>>,
    token_ids: Rc<HashMap<String, TokenID>>,
    bpe_ranks: HashMap<(TokenID, TokenID), usize>,
    byte_encodes: HashMap<u8, char>,
    byte_decodes: HashMap<char, u8>,
    bos_token: TokenID,
    eos_token: TokenID,
}

impl Gpt2Tokenizer {
    pub fn new(
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
        let byte_encodes = build_byte_encode_map();
        let byte_decodes = byte_encodes.iter().map(|(b, u)| (*u, *b)).collect();
        // i still dunno what this regex is doing
        let _pattern =
            Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+")
                .unwrap();
        Self {
            tokens,
            token_ids,
            bpe_ranks: merges,
            byte_encodes,
            byte_decodes,
            bos_token,
            eos_token,
        }
    }

    pub fn decode(&self, token_id: TokenID) -> Vec<u8> {
        let token = &self.tokens[token_id];
        if token.len() > 1 {
            let s: Vec<u8> = token
                .chars()
                .flat_map(|c| {
                    self.byte_decodes
                        .get(&c)
                        .map(|b| vec![*b])
                        .unwrap_or(c.to_string().bytes().collect::<Vec<_>>())
                })
                .collect();
            s
        } else if token.len() == 1 {
            let ch = token.chars().next().unwrap();
            match self.byte_decodes.get(&ch) {
                Some(b) => vec![*b],
                None => vec![ch as u8],
            }
        } else {
            vec![]
        }
    }

    #[allow(unused)]
    pub fn decode_tokens(&self, token_ids: &[TokenID]) -> String {
        let bytes = token_ids
            .iter()
            .flat_map(|t| self.decode(*t))
            .collect::<Vec<_>>();
        String::from_utf8_lossy(&bytes).to_string()
    }

    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    pub fn encode(&self, text: &str, bos: bool, eos: bool, add_prefix_space: bool) -> Vec<TokenID> {
        let text = if add_prefix_space {
            format!(" {}", text)
        } else {
            text.to_string()
        };

        let special_tokens = vec![
            // for qwen2
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            // for llama3
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>",
        ];
        let parts = split_text_by_keyword(&text, &special_tokens);

        let mut tokens = parts
            .iter()
            .flat_map(|s| {
                if special_tokens.contains(&s.as_str()) {
                    return vec![*self.token_ids.get(s).unwrap()];
                }
                let mut toks = vec![];
                for b in s.bytes() {
                    let ch = self.byte_encodes.get(&b).unwrap().to_string();
                    let token_id = self.token_ids.get(&ch).unwrap();
                    toks.push(*token_id);
                }
                toks = self.bpe_merge(toks);
                toks
            })
            .collect::<Vec<_>>();

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
                tokens[merging_idx] = *self
                    .token_ids
                    .get(&format!("{}{}", token1, token2))
                    .unwrap();
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
    let ranges = [('!', '~'), ('¡', '¬'), ('®', 'ÿ')];
    for (start, end) in ranges.iter() {
        for i in *start..=*end {
            map.insert(i as u8, i);
        }
    }
    let mut extra_unicode = 256;
    for i in 0..=255 {
        if !map.contains_key(&(i as u8)) {
            map.insert(i as u8, std::char::from_u32(extra_unicode).unwrap());
            extra_unicode += 1;
        }
    }
    map
}

fn split_text_by_keyword(text: &str, keywords: &[&str]) -> Vec<String> {
    // Escape the keywords and join them into a regular expression pattern
    let escaped_keywords: Vec<String> = keywords.iter().map(|k| regex::escape(k)).collect();
    let pattern = format!("({})", escaped_keywords.join("|"));
    let re = Regex::new(&pattern).unwrap();

    let mut result = Vec::new();
    let mut last = 0;

    for mat in re.find_iter(text) {
        if mat.start() > last {
            result.push(text[last..mat.start()].to_string());
        }
        result.push(mat.as_str().to_string());
        last = mat.end();
    }
    if last < text.len() {
        result.push(text[last..].to_string());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use crate::gguf::GGUFFileLoader;

    #[ignore]
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

        let token_ids = tk.encode("我不吃牛肉", false, false, false);
        assert_eq!(tk.tokens[token_ids[0]], "æĪĳä¸į");
        assert_eq!(tk.decode_tokens(&[token_ids[0]]), "我不");

        let tests = vec![
            ("Captain America: ", "Captain America: "),
            (
                "<|im_start|> blah <|im_end|> ",
                "<|im_start|> blah <|im_end|> ",
            ),
            ("tiktok", "tiktok"),
            ("i don't eat beaf.", "i don't eat beaf."),
            ("我不吃牛肉", "我不吃牛肉"),
        ];

        for tt in tests {
            let outputs = tk.encode(tt.0, false, false, false);
            let tokens_in_string = tk.decode_tokens(&outputs);
            assert_eq!(tokens_in_string, tt.1, "failed to encode {}", tt.0);
        }

        Ok(())
    }

    #[test]
    fn test_split_words() {
        let output: Vec<String> = split_text_by_keyword(
            "<|im_start|>i don't eat beaf<|im_end|> <|im_start|> i don't eat beaf",
            &["<|im_start|>", "<|im_end|>"],
        );
        assert_eq!(output, vec![
            "<|im_start|>",
            "i don't eat beaf",
            "<|im_end|>",
            " ",
            "<|im_start|>",
            " i don't eat beaf"
        ]);
    }

    #[test]
    fn test_unicode_table() {
        let encode_map = build_byte_encode_map();
        let b = 230_u8;
        assert_eq!(encode_map.get(&b).cloned(), Some('æ'));
    }
}
