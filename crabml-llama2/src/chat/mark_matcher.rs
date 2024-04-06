pub enum MarkMatchState {
    Inactive,
    Active,
}

pub struct MarkMatcher {
    state: MarkMatchState,
    buf: String,
    marks: Vec<String>,
}

impl MarkMatcher {
    pub fn new(marks: Vec<String>) -> Self {
        Self {
            state: MarkMatchState::Inactive,
            buf: String::new(),
            marks,
        }
    }

    pub fn push(&mut self, token: String) -> Option<String> {
        match self.state {
            MarkMatchState::Inactive => {
                // exact match, do not change state
                if self.marks.contains(&token) {
                    return Some(token);
                }

                // got any partial match, change state to active, and push the token
                // to the buffer, and wait for the rest of the mark.
                if self.marks.iter().any(|m| m.starts_with(&token)) {
                    self.state = MarkMatchState::Active;
                    self.buf = token;
                    return None;
                }

                // no match, return the token directly
                Some(token)
            }
            MarkMatchState::Active => {
                self.buf.push_str(&token);

                // exact match, change state to inactive, and return the buffer
                if self.marks.contains(&self.buf) {
                    self.state = MarkMatchState::Inactive;
                    return Some(self.buf.clone());
                }

                // not match anymore, return the buffer directly
                if !self.marks.iter().any(|m| m.starts_with(&self.buf)) {
                    self.state = MarkMatchState::Inactive;
                    return Some(self.buf.clone());
                }

                // partial match, wait for the rest of the mark
                None
            }
        }
    }
}
