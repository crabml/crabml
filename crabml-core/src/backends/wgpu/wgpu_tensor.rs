use std::rc::Rc;

use wgpu;

struct WgpuTensor {
    buf: Rc<wgpu::Buffer>,
}