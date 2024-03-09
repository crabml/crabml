use std::borrow::Cow;

#[derive(Debug, Clone)]
pub struct QuantBufQ2_K<'a> {
    pub blocks: Cow<'a, [BlockQ2_K]>,
}

#[derive(Debug, Clone)]
pub struct BlockQ2_K {}
