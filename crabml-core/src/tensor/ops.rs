use std::collections::HashMap;

use crate::error::Result;
use crate::tensor::Tensor;

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

pub struct TensorOps<'a> {
    preallocated_sizes: Vec<usize>,
    ops: Vec<TensorLogicalOp<'a>>,
}

impl<'a> TensorOps<'a> {
}

pub trait TensorOpsRunner<'a> {
    fn new(&self, compute: TensorOps<'a>) -> Result<Box<Self>>;

    fn run(&mut self, input: Tensor<'a>) -> Result<()>;
}

pub struct TensorOpsRunnerCPU<'a> {
    preallocated_tensors: Vec<Option<Tensor<'a>>>,
}

pub enum TensorOp<'a, 'b> where 'a: 'b {
    Nop,

    LoadInput {
        dst: &'b mut Tensor<'a>,
    }
}

// TODO: use TensorVarID instead of TensorVar here
// get_recycleable_tensor_ids(op_pos)
pub enum TensorLogicalOp<'a> {
    Nop,

    LoadInput {
        dst: TensorVar<'a>,
    },

    Return {
        ret: TensorVar<'a>,
    },

    Define {
        var: TensorVar<'a>,
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
        t1: TensorVar<'a>,  // (a, b)
        t2: TensorVar<'a>,  // (b, c)
    },
}

#[derive(Debug, Clone)]
pub struct TensorVarDef {
    shape: Vec<usize>,
    id: usize,
    parent_ids: Vec<usize>,
    name: Option<String>,
    buf_size: usize,
}

pub struct TensorOpsBuilder<'a> {
    ops: Vec<TensorLogicalOp<'a>>,
    defined_vars: Vec<TensorVarDef>,
}

impl<'a> TensorOpsBuilder<'a> {
    fn new() -> Self {
        Self {
            ops: vec![],
            defined_vars: vec![],
        }
    }

    fn new_tensor(&mut self, shape: Vec<usize>, parents: &[&TensorVar<'a>]) -> TensorVar<'a> {
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
        self.ops.push(TensorLogicalOp::MatMul {
            dst: dst.clone(),
            t1,
            t2,
        });
        dst
    }

    pub fn rms_norm(&mut self, mat: impl Into<TensorVar<'a>>, eps: f32) -> TensorVar<'a> {
        let mat = mat.into();
        let dst = self.new_tensor(mat.shape().to_vec(), &[&mat]);
        self.ops.push(TensorLogicalOp::RmsNorm {
            dst: dst.clone(),
            mat,
            eps,
        });
        dst
    }

    fn finish(mut self, ret: TensorVar<'a>) -> TensorOps<'a> {
        self.ops.push(
            TensorLogicalOp::Return { ret }
        );

        let physical_var_ids = self.assign_physical_tensor_var_ids();
        let preallocated_count = physical_var_ids.iter().max().unwrap() + 1;
        let preallocated_sizes = (0..preallocated_count).map(|id| {
            let tensor_def = &self.defined_vars[id];
            let n = tensor_def.buf_size;
            n
        }).collect::<Vec<_>>();

        TensorOps {
            preallocated_sizes,
            ops: self.ops,
        }
    }

    // Assign the physical id to all the tensor vars in the op codes. One preallocated tensor might be
    // reused multiple times.
    fn assign_physical_tensor_var_ids(&self) -> Vec<usize> {
        let mut physical_ids_map: HashMap<usize, usize> = HashMap::new();
        let mut allocated_tensors: Vec<usize> = vec![];
        let mut recycled_tensors: Vec<usize> = vec![];

        for current_var in self.defined_vars.iter() {
            // try allocate a new tensor, reuse the recycled tensor when possible
            let reusable_tensor = recycled_tensors
                .iter()
                .find(|id| self.defined_vars[**id].buf_size == current_var.buf_size)
                .cloned();

            // if the tensor is reusable, we need to remove it from the recycled_tensors list, and
            // just reuse the physical id of it.
            // else if the tensor is not reusable, we need to allocate a new physical id for it.
            let physical_id = if let Some(reuseable_tensor_id) = reusable_tensor {
                recycled_tensors.retain(|id| *id != reuseable_tensor_id);
                physical_ids_map[&reuseable_tensor_id]
            } else {
                allocated_tensors.push(current_var.id);
                allocated_tensors.len() - 1
            };
            physical_ids_map.insert(current_var.id, physical_id);

            // whenever a new tensor is defined, we have a chance to check whether its parents are still
            // being referenced or not.
            for parent_id in current_var.parent_ids.iter() {
                if current_var.id == self.defined_vars.len() - 1 {
                    break;
                }
                let is_referenced = self.defined_vars[current_var.id + 1..]
                    .iter()
                    .any(|v| v.parent_ids.contains(parent_id));
                if !is_referenced {
                    recycled_tensors.push(*parent_id);
                }
            }
        }
        (0..physical_ids_map.len())
            .map(|i| physical_ids_map[&i])
            .collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_compute_builder() -> Result<()> {
        let mut b = TensorOpsBuilder::new();

        let t1 = b.new_tensor(vec![2, 3], &vec![]);
        let t2 = b.new_tensor(vec![3, 2], &vec![]);
        let t3 = b.matmul(t1, t2);
        let t4 = b.rms_norm(t3, 1e-5);
        let m = b.assign_physical_tensor_var_ids();
        assert_eq!(format!("{:?}", m), "[0, 1, 2, 3]");

        let mut b = TensorOpsBuilder::new();
        let t0 = b.new_tensor(vec![2, 2], &vec![]);
        let t1 = b.new_tensor(vec![2, 2], &vec![]);
        let t2 = b.matmul(t0, t1);
        let t3 = b.rms_norm(t2, 1e-5); // t0 and t1 can be reused
        let t4 = b.rms_norm(t3, 1e-5); // t1, t2 and t3 can be reused
        let m = b.assign_physical_tensor_var_ids();
        assert_eq!(format!("{:?}", m), "[0, 1, 2, 0, 1]");

        Ok(())
    }
}