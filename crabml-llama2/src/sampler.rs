use std::cell::RefCell;
use std::rc::Rc;

use crabml::cpu::buf::buf_f32::exp_f32_cached;
use crabml::error::Error;
use crabml::error::ErrorKind;
use crabml::error::Result;
use half::f16;
use rand::Rng;

pub struct Llama2Sampler {
    prob_index: RefCell<Vec<(f32, usize)>>,
    temperature: f32,
    topp: f32,
    exp_cache: Rc<Vec<f16>>,
}

pub type Llama2SamplerRef = Rc<Llama2Sampler>;

impl Llama2Sampler {
    pub fn new(
        vocab_size: usize,
        temperature: f32,
        topp: f32,
        exp_cache: Rc<Vec<f16>>,
    ) -> Llama2SamplerRef {
        Rc::new(Self {
            prob_index: RefCell::new(vec![(0.0, 0); vocab_size]),
            temperature,
            topp,
            exp_cache,
        })
    }

    pub fn sample(&self, logits: &mut [f32]) -> Result<usize> {
        if self.temperature == 0.0 {
            return Self::sample_argmax(logits);
        }

        // apply the temperature to the logits. the lower the temperature,
        // the more deterministic the sampling.
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, self.exp_cache.as_ref());

        // flip a (float) coin (this is our source of entropy for sampling)
        let mut rng = rand::thread_rng();
        let coin: f32 = rng.gen_range(0.0..1.0);

        // we sample from this distribution to get the next token
        if self.topp <= 0_f32 || self.topp >= 1.0_f32 {
            // simply sample from the predicted probability distribution
            Self::sample_multi(logits, coin);
        }

        Self::sample_topp(logits, self.topp, &self.prob_index, coin)
    }

    pub fn sample_multi(probs: &[f32], coin: f32) -> usize {
        // sample index from probabilities (they must sum to 1!)
        // coin is a random number in [0, 1), usually from random_f32()
        let mut cdf = 0_f32;
        for (i, p) in probs.iter().enumerate() {
            cdf += p;
            if cdf > coin {
                return i;
            }
        }
        probs.len() - 1 // in case of rounding errors
    }

    pub fn sample_topp(
        probs: &[f32],
        topp: f32,
        prob_index: &RefCell<Vec<(f32, usize)>>,
        coin: f32,
    ) -> Result<usize> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability topp. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        // coin is a random number in [0, 1), usually from random_f32()
        let mut prob_index = prob_index.borrow_mut();

        let cutoff = (1.0_f32 - topp) / (probs.len() - 1) as f32;
        let mut n0 = 0;
        for (i, prob) in probs.iter().enumerate() {
            if *prob >= cutoff {
                prob_index[n0] = (probs[i], i);
                n0 += 1;
            }
        }
        prob_index[..n0].sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // truncate the list where cumulative probability exceeds topp
        let mut cumulative_prob = 0_f32;
        let mut last_idx = n0 - 1; // in case of rounding errors consider all elements
        for (i, prob) in prob_index[0..n0].iter().enumerate() {
            cumulative_prob += prob.0;
            if cumulative_prob > topp {
                last_idx = i;
                break; // we've exceeded topp by including last_idx
            }
        }

        // sample from the truncated list
        let r = coin * cumulative_prob;
        let mut cdf = 0_f32;
        for prob in prob_index[0..=last_idx].iter() {
            cdf += prob.0;
            if cdf > r {
                return Ok(prob.1);
            }
        }
        Ok(prob_index[last_idx].1) // in case of rounding errors
    }

    pub fn sample_argmax(probs: &[f32]) -> Result<usize> {
        probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .ok_or_else(|| Error {
                kind: ErrorKind::Unexpected,
                message: "failed to sample from logits".to_string(),
                cause: None,
            })
    }
}

pub fn softmax(a: &mut [f32], exp_cache: &[f16]) {
    let max = a.iter().fold(f32::NAN, |a, b| a.max(*b));
    let mut sum = 0.0;
    for a in a.iter_mut() {
        *a = exp_f32_cached(*a - max, exp_cache);
        sum += *a;
    }
    for a in a.iter_mut() {
        *a /= sum;
    }
}
