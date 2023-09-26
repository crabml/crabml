use std::collections::HashMap;
use std::os::unix::process::parent_id;

use crate::tensor::Tensor;
use crate::error::Result;

#[derive(Debug, Clone)]
pub enum TensorVar<'a> {
    Value(Tensor<'a>),
    Var { id: usize, shape: Vec<usize> },
}

impl<'a> TensorVar<'a> {
    pub fn shape(&self) -> &[usize] {
        match self {
            TensorVar::Value(v) => v.shape(),
            TensorVar::Var { ref shape, .. } => shape,
        }
    }
}

pub struct TensorCompute<'a> {
    preallocated_tensors: Vec<Tensor<'a>>,
    ops: Vec<TensorOp<'a>>,
}

pub enum TensorOp<'a> {
    Nop,

    LoadInput {
        dst: TensorVar<'a>,
    },

    Return {
        ret: TensorVar<'a>,
    },

    Define {
        ret: TensorVar<'a>,
    },

    Copy {
        dst: TensorVar<'a>,
        src: TensorVar<'a>,
    },

    Mul {
        dst: TensorVar<'a>,
        t1: TensorVar<'a>,
        t2: TensorVar<'a>,
    },

    Add {
        dst: TensorVar<'a>,
        t1: TensorVar<'a>, // t1.shape == t2.shape
        t2: TensorVar<'a>,
    },

    RmsNorm {
        dst: TensorVar<'a>,
        mat: TensorVar<'a>, // (m, n) or (n, )
        eps: f32,
    },

    SiLU {
        dst: TensorVar<'a>,
        src: TensorVar<'a>,
    },

    MatMul {
        dst: TensorVar<'a>, // (a, c)
        t1: TensorVar<'a>, // (a, b)
        t2: TensorVar<'a>, // (b, c)
    },
}

#[derive(Debug, Clone)]
pub struct TensorVarDef {
    shape: Vec<usize>,
    id: usize,
    parent_ids: Vec<usize>,
    name: Option<String>,
    physical_id: Option<usize>,
    buf_size: usize,
}

pub struct TensorComputeBuilder<'a> {
    ops: Vec<TensorOp<'a>>,
    defined_vars: Vec<TensorVarDef>,
}

impl<'a> TensorComputeBuilder<'a> {
    fn new() -> Self {
        Self {
            ops: vec![],
            defined_vars: vec![],
        }
    }

    fn new_tensor(
        &mut self,
        shape: Vec<usize>,
        parents: &[&TensorVar<'a>],
    ) -> TensorVar<'a> {
        let mut parent_ids = vec![];
        for parent in parents {
            if let TensorVar::Var { id, .. } = parent {
                parent_ids.push(*id);
            }
        }

        let next_var_id = self.defined_vars.len();
        self.defined_vars.push(TensorVarDef {
            id: next_var_id,
            shape: shape.clone(),
            parent_ids,
            name: None,
            physical_id: None,
            buf_size: shape.iter().product(),
        });
        let var = TensorVar::Var {
            id: next_var_id,
            shape: shape.clone(),
        };
        var
    }

    pub fn bind_name(&mut self, t: impl Into<TensorVar<'a>>, name: String) -> TensorVar<'a> {
        let t = t.into();
        match t {
            TensorVar::Value(t) => TensorVar::Value(t.with_name(name)),
            TensorVar::Var { id, shape } => {
                self.defined_vars[id].name = Some(name.clone());
                TensorVar::Var { id, shape }
            }
        }
    }

    pub fn matmul(
        &mut self,
        t1: impl Into<TensorVar<'a>>,
        t2: impl Into<TensorVar<'a>>,
    ) -> TensorVar<'a> {
        let t1 = t1.into();
        let t2 = t2.into();
        let dst = self.new_tensor(vec![t1.shape()[0], t2.shape()[1]], &[&t1, &t2]);
        self.ops.push(TensorOp::MatMul {
            dst: dst.clone(),
            t1,
            t2,
        });
        dst
    }

    pub fn rms_norm(&mut self, mat: impl Into<TensorVar<'a>>, eps: f32) -> TensorVar<'a> {
        let mat = mat.into();
        let dst = self.new_tensor(mat.shape().to_vec(), &[&mat]);
        self.ops.push(TensorOp::RmsNorm {
            dst: dst.clone(),
            mat,
            eps,
        });
        dst
    }

    // Assign the physical id to all the tensor vars in the op codes. One preallocated tensor might be 
    // reused multiple times.
    fn assign_physical_var_ids(&mut self) -> Vec<(usize, usize)> {
        let mut physical_ids_map: HashMap<usize, usize> = HashMap::new();
        let mut allocated_tensors: Vec<usize> = vec![];
        let mut recycled_tensors: Vec<usize> = vec![];
        for current_var in self.defined_vars.iter() {
            // try allocate a new tensor, reuse the recycled tensor when possible
            let reusable_tensor = recycled_tensors
                .iter()
                .find(|id| self.defined_vars[**id].buf_size == current_var.buf_size)
                .cloned();
            let physical_id = if let Some(reuseable_tensor_id) = reusable_tensor {
                // if the tensor is reusable, we need to remove it from the recycled_tensors list, and 
                // just reuse the physical id of it.
                recycled_tensors.retain(|id| *id != reuseable_tensor_id);
                physical_ids_map[&reuseable_tensor_id]
            } else {
                // if the tensor is not reusable, we need to allocate a new physical id for it.
                allocated_tensors.push(current_var.id);
                allocated_tensors.len() - 1
            };
            physical_ids_map.insert(current_var.id, physical_id);

            // whenever a new tensor is defined, we have a chance to check whether its parent is still 
            // referenced or not.
            for parent_id in current_var.parent_ids.iter() {
                if current_var.id == self.defined_vars.len() - 1 {
                    break;
                }
                let is_referenced = self.defined_vars[current_var.id+1..].iter().any(|v| v.parent_ids.contains(parent_id));
                if !is_referenced {
                    recycled_tensors.push(*parent_id);
                }
            }
        }
        (0..physical_ids_map.len()).map(|i| (i, physical_ids_map[&i])).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_compute_builder() -> Result<()> {
        let mut b = TensorComputeBuilder::new();

        let t1 = b.new_tensor(vec![2, 3], &vec![]);
        let t2 = b.new_tensor(vec![3, 2], &vec![]);
        let t3 = b.matmul(t1, t2);
        let t4 = b.rms_norm(t3, 1e-5);
        let m = b.assign_physical_var_ids();
        assert_eq!(format!("{:?}", m), "[(0, 0), (1, 1), (2, 2), (3, 3)]");

        let mut b = TensorComputeBuilder::new();
        let t0 = b.new_tensor(vec![2, 2], &vec![]);
        let t1 = b.new_tensor(vec![2, 2], &vec![]);
        let t2 = b.matmul(t0, t1);
        let t3 = b.rms_norm(t2, 1e-5); // t0 and t1 can be reused
        let t4 = b.rms_norm(t3, 1e-5); // t1, t2 and t3 can be reused
        let m = b.assign_physical_var_ids();
        assert_eq!(format!("{:?}", m), "(0, 0), (1, 1), (2, 2), (3, 0), (4, 1)");

        Ok(())
    }
}