// SPDX-License-Identifier: GPL-3.0-only

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum PixelSampling {
    #[default]
    R2,
    Sobol,
    Stbn,
}

impl PixelSampling {
    pub const ALL: [Self; 3] = [Self::R2, Self::Sobol, Self::Stbn];

    pub fn label(self) -> &'static str {
        match self {
            Self::R2 => "R2 (original)",
            Self::Sobol => "Sobol",
            Self::Stbn => "STBN (blue noise)",
        }
    }

    pub fn uniform(self) -> f32 { self as u8 as f32 }
}

pub(crate) fn blue_noise_view(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let size = wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 32 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Local spatiotemporal blue-noise mask"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    const MASK: &[u8; 32 * 16 * 16 * 4] = include_bytes!("../assets/stbn-16x16x32.bin");
    queue.write_texture(
        texture.as_image_copy(),
        MASK,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size.width * 4), rows_per_image: Some(size.height) },
        size,
    );
    texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}
