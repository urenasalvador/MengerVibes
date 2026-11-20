use std::sync::{Arc, mpsc, atomic::{AtomicBool, Ordering}};

use crate::denoise::{self, Backend};

struct FilteredFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

struct Job {
    result: mpsc::Receiver<Result<FilteredFrame, String>>,
    cancel: Arc<AtomicBool>,
}

#[derive(Default)]
pub(super) struct ViewportDenoise {
    mode: Option<Backend>,
    job: Option<Job>,
    attempted: bool,
    pub(super) guides: Option<[wgpu::Texture; 2]>,
    pub(super) pipeline: Option<wgpu::RenderPipeline>,
    pub(super) display: Option<wgpu::BindGroup>,
    pub(super) error: Option<String>,
}

impl ViewportDenoise {
    pub fn busy(&self) -> Option<Backend> {
        self.job.as_ref().filter(|job| !job.cancel.load(Ordering::Relaxed)).and(self.mode)
    }

    pub fn configure(&mut self, mode: Option<Backend>, reset: bool) {
        if reset || self.mode != mode {
            if let Some(job) = &self.job { job.cancel.store(true, Ordering::Relaxed); }
            self.display = None;
            self.error = None;
            self.attempted = false;
        }
        if mode.is_none() { self.guides = None; }
        self.mode = mode;
    }

    pub fn prepare(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, size: wgpu::Extent3d) -> bool {
        if self.mode.is_none() { return false; }
        if !self.guides.as_ref().is_some_and(|guides| guides[0].size() == size) {
            self.guides = Some(denoise::guide_textures(device, size));
        }
        self.pipeline.get_or_insert_with(|| crate::export::sample_pipeline(device, layout, "fs_progress_denoise", true));
        true
    }

    pub fn start(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, accumulation: &wgpu::Texture,
                 [width, height]: [u32; 2], wake: impl FnOnce() + Send + 'static) {
        let (Some(mode), Some([albedo, normal])) = (self.mode, self.guides.as_ref()) else { return; };
        if self.attempted || self.job.is_some() { return; }
        self.attempted = true;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let row = (width * 8).div_ceil(alignment) * alignment;
        let plane = u64::from(row) * u64::from(height);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport denoising readback"), size: plane * 3,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        for (index, texture) in [accumulation, albedo, normal].into_iter().enumerate() {
            encoder.copy_texture_to_buffer(
                texture.as_image_copy(),
                wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout {
                    offset: index as u64 * plane, bytes_per_row: Some(row), rows_per_image: Some(height),
                } },
                wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            );
        }
        let submission = queue.submit(Some(encoder.finish()));
        let device = device.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let worker_cancel = cancel.clone();
        let (send, result) = mpsc::channel();
        self.job = Some(Job { result, cancel });

        std::thread::spawn(move || {
            let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (done, ready) = mpsc::channel();
                let slice = buffer.slice(..);
                slice.map_async(wgpu::MapMode::Read, move |mapped| { let _ = done.send(mapped); });
                let _ = device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
                ready.recv().map_err(|error| error.to_string())?.map_err(|error| error.to_string())?;
                if worker_cancel.load(Ordering::Relaxed) { return Err("Viewport changed".into()); }
                let mut frame = denoise::Frame::new(width, height)?;
                {
                    let mapped = slice.get_mapped_range();
                    frame.write_tile(&mapped, plane as usize, row as usize, [0, 0], [width, height]);
                }
                buffer.unmap();
                let pixels = frame.finish_viewport(mode, &worker_cancel)?;
                Ok(FilteredFrame { width, height, pixels })
            })).unwrap_or_else(|_| Err("Viewport denoising failed unexpectedly".into()));
            let _ = send.send(output);
            wake();
        });
    }

    pub fn poll(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, composite: &wgpu::RenderPipeline,
                camera: &wgpu::Buffer, size: wgpu::Extent3d) {
        let Some(job) = &self.job else { return; };
        let result = match job.result.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return,
            Err(mpsc::TryRecvError::Disconnected) => Err("Viewport denoising worker disconnected".into()),
        };
        let stale = job.cancel.load(Ordering::Relaxed);
        self.job = None;
        if stale { return; }
        match result {
            Ok(frame) => {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Denoised viewport"), size, mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
                });
                queue.write_texture(texture.as_image_copy(), &frame.pixels, wgpu::TexelCopyBufferLayout {
                    offset: 0, bytes_per_row: Some(frame.width * 8), rows_per_image: Some(frame.height),
                }, wgpu::Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 });
                self.display = Some(crate::Renderer::bind_composite(device, composite, camera, &texture));
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub fn resize(&mut self) {
        self.configure(self.mode, true);
        self.guides = None;
    }
}
