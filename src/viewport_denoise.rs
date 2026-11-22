use std::{sync::{Arc, mpsc, atomic::{AtomicBool, Ordering}}, time::Instant};

use crate::denoise::{self, Backend};

pub(super) const DEFAULT_INTERVAL: u32 = 4;

struct FilteredFrame {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    timing: ReadySnapshot,
}

struct ReadySnapshot {
    samples: u32,
    readback_ms: f64,
    preparation_ms: f64,
    oidn_ms: f64,
    filtered_at: Instant,
}

pub(super) struct PresentedSnapshot {
    pub samples: u32,
    pub traced_samples: u32,
    pub readback_ms: f64,
    pub preparation_ms: f64,
    pub oidn_ms: f64,
    pub presentation_ms: f64,
}

struct Job {
    result: mpsc::Receiver<Result<FilteredFrame, String>>,
    cancel: Arc<AtomicBool>,
    samples: u32,
}

fn snapshot_due(last_started: u32, completed: u32, target: u32, interval: u32) -> bool {
    completed > last_started && (last_started == 0 || completed >= target || completed % interval == 0)
}

#[derive(Default)]
pub(super) struct ViewportDenoise {
    mode: Option<Backend>,
    job: Option<Job>,
    last_started: u32,
    display_texture: Option<wgpu::Texture>,
    ready: Option<ReadySnapshot>,
    pub(super) presented: Option<PresentedSnapshot>,
    pub(super) guides: Option<[wgpu::Texture; 2]>,
    pub(super) pipeline: Option<wgpu::RenderPipeline>,
    pub(super) display: Option<wgpu::BindGroup>,
    pub(super) error: Option<String>,
}

impl ViewportDenoise {
    pub fn active_snapshot(&self) -> Option<(Backend, u32)> {
        let job = self.job.as_ref().filter(|job| !job.cancel.load(Ordering::Relaxed))?;
        Some((self.mode?, job.samples))
    }

    pub fn awaiting_first_preview(&self) -> bool {
        self.presented.is_none() && (self.ready.is_some() || self.active_snapshot().is_some())
    }

    pub fn snapshot_pending(&self, completed: u32, target: u32, interval: u32) -> bool {
        self.mode.is_some() && self.error.is_none() && snapshot_due(self.last_started, completed, target, interval)
    }

    pub fn ready_to_present(&self) -> bool {
        self.ready.is_some()
    }

    pub fn record_presentation(&mut self, traced_samples: u32) {
        let Some(snapshot) = self.ready.take() else { return; };
        self.presented = Some(PresentedSnapshot {
            samples: snapshot.samples,
            traced_samples,
            readback_ms: snapshot.readback_ms,
            preparation_ms: snapshot.preparation_ms,
            oidn_ms: snapshot.oidn_ms,
            presentation_ms: snapshot.filtered_at.elapsed().as_secs_f64() * 1000.0,
        });
    }

    pub fn configure(&mut self, mode: Option<Backend>, reset: bool) {
        if reset || self.mode != mode {
            if let Some(job) = &self.job { job.cancel.store(true, Ordering::Relaxed); }
            self.display = None;
            self.display_texture = None;
            self.ready = None;
            self.presented = None;
            self.error = None;
            self.last_started = 0;
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
                 [width, height]: [u32; 2], samples: u32, target: u32, interval: u32, wake: impl FnOnce() + Send + 'static) -> bool {
        let (Some(mode), Some([albedo, normal])) = (self.mode, self.guides.as_ref()) else { return false; };
        if self.job.is_some() || !self.snapshot_pending(samples, target, interval) { return false; }
        self.last_started = samples;
        let started_at = Instant::now();
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
        self.job = Some(Job { result, cancel, samples });

        std::thread::spawn(move || {
            let output = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let (done, ready) = mpsc::channel();
                let slice = buffer.slice(..);
                slice.map_async(wgpu::MapMode::Read, move |mapped| { let _ = done.send(mapped.map(|()| Instant::now())); });
                let _ = device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
                let mapped_at = ready.recv().map_err(|error| error.to_string())?.map_err(|error| error.to_string())?;
                if worker_cancel.load(Ordering::Relaxed) { return Err("Viewport changed".into()); }
                let mut frame = denoise::Frame::new(width, height)?;
                {
                    let mapped = slice.get_mapped_range();
                    frame.write_tile(&mapped, plane as usize, row as usize, [0, 0], [width, height]);
                }
                buffer.unmap();
                let prepared_at = Instant::now();
                let (pixels, filtered_at) = frame.finish_viewport(mode, &worker_cancel)?;
                Ok(FilteredFrame {
                    width, height, pixels,
                    timing: ReadySnapshot {
                        samples,
                        readback_ms: (mapped_at - started_at).as_secs_f64() * 1000.0,
                        preparation_ms: (prepared_at - mapped_at).as_secs_f64() * 1000.0,
                        oidn_ms: (filtered_at - prepared_at).as_secs_f64() * 1000.0,
                        filtered_at,
                    },
                })
            })).unwrap_or_else(|_| Err("Viewport denoising failed unexpectedly".into()));
            let _ = send.send(output);
            wake();
        });
        true
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
                let texture = self.display_texture.get_or_insert_with(|| {
                    let texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("Denoised viewport"), size, mip_level_count: 1, sample_count: 1,
                        dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, view_formats: &[],
                    });
                    self.display = Some(crate::Renderer::bind_composite(device, composite, camera, &texture));
                    texture
                });
                queue.write_texture(texture.as_image_copy(), &frame.pixels, wgpu::TexelCopyBufferLayout {
                    offset: 0, bytes_per_row: Some(frame.width * 8), rows_per_image: Some(frame.height),
                }, wgpu::Extent3d { width: frame.width, height: frame.height, depth_or_array_layers: 1 });
                self.ready = Some(frame.timing);
            }
            Err(error) => self.error = Some(error),
        }
    }

    pub fn resize(&mut self) {
        self.configure(self.mode, true);
        self.guides = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_snapshots_wait_only_when_the_next_checkpoint_is_due() {
        let mut denoiser = ViewportDenoise::default();
        denoiser.configure(Some(Backend::Gpu), false);
        assert!(!denoiser.awaiting_first_preview());

        let (_sender, result) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        denoiser.job = Some(Job { result, cancel: cancel.clone(), samples: 4 });
        denoiser.last_started = 4;
        assert!(denoiser.awaiting_first_preview());

        denoiser.presented = Some(PresentedSnapshot {
            samples: 4, traced_samples: 4, readback_ms: 0.0, preparation_ms: 0.0, oidn_ms: 0.0, presentation_ms: 0.0,
        });
        assert!(!denoiser.awaiting_first_preview());
        assert!(!denoiser.snapshot_pending(7, 24, 4));
        assert!(denoiser.snapshot_pending(8, 24, 4));

        denoiser.configure(Some(Backend::Gpu), true);
        assert!(cancel.load(Ordering::Relaxed));
        assert!(!denoiser.awaiting_first_preview());
    }

    #[test]
    fn snapshots_follow_the_fixed_interval_and_include_first_and_last_samples() {
        assert!(!snapshot_due(0, 0, 24, 4));
        assert!(snapshot_due(0, 1, 24, 4));
        assert!(!snapshot_due(1, 3, 24, 4));
        assert!(snapshot_due(1, 4, 24, 4));
        assert!(snapshot_due(4, 8, 24, 4));
        assert!(!snapshot_due(4, 9, 24, 4));
        assert!(snapshot_due(4, 12, 24, 4));
        assert!(snapshot_due(4, 6, 6, 4));
        assert!(!snapshot_due(6, 6, 6, 4));
        assert!(snapshot_due(1, 2, 24, 1));
        assert!(!snapshot_due(1, 8, 24, 5));
        assert!(snapshot_due(1, 10, 24, 5));
    }
}
