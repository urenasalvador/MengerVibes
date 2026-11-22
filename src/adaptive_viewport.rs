use std::{sync::mpsc, time::{Duration, Instant}};

pub(crate) const DEFAULT_BUDGET_MS: u32 = 24;
pub(crate) const POLL_INTERVAL: Duration = Duration::from_millis(2);

pub(crate) struct AdaptiveViewport {
    pub enabled: bool,
    pub budget_ms: u32,
    divisor: u32,
    ms_per_ray: Option<f64>,
    affordable_passes: u8,
    promotion_cooldown: u8,
    pub last: Option<PassStats>,
}

impl Default for AdaptiveViewport {
    fn default() -> Self {
        Self { enabled: true, budget_ms: DEFAULT_BUDGET_MS, divisor: 4,
            ms_per_ray: None, affordable_passes: 0, promotion_cooldown: 0, last: None }
    }
}

pub(crate) struct PassStats {
    pub size: [u32; 2],
    divisor: u32,
    pub rays: u64,
    pub ms: f64,
    pub gpu: bool,
    pub valid_fraction: Option<f64>,
    pub metrics_ms: Option<f64>,
    pub completion_ms: f64,
}

impl AdaptiveViewport {
    pub fn restart(&mut self) {
        self.divisor = 4;
        self.ms_per_ray = None;
        self.affordable_passes = 0;
        self.promotion_cooldown = 0;
        self.last = None;
    }

    pub fn divisor(&self) -> u32 { self.divisor }

    pub fn observe(&mut self, stats: PassStats) {
        let gpu_cost = stats.ms + stats.metrics_ms.unwrap_or(0.0);
        let ms_per_ray = gpu_cost / stats.rays.max(1) as f64;
        let smoothed = self.ms_per_ray.map_or(ms_per_ray, |previous| previous * 0.7 + ms_per_ray * 0.3);
        self.ms_per_ray = Some(smoothed);
        let budget = f64::from(self.budget_ms);
        if self.divisor < 4 && (gpu_cost > budget * 1.2 || stats.completion_ms > budget * 1.5) {
            self.divisor *= 2;
            self.affordable_passes = 0;
            self.promotion_cooldown = 8;
        } else if self.promotion_cooldown > 0 {
            self.promotion_cooldown -= 1;
        } else if self.divisor > 1 && self.divisor == stats.divisor
            && smoothed * stats.rays as f64 * 4.0 < budget
            && stats.valid_fraction.is_none_or(|fraction| fraction >= 0.90)
        {
            self.affordable_passes += 1;
            if self.affordable_passes >= 2 {
                self.divisor /= 2;
                self.affordable_passes = 0;
            }
        } else {
            self.affordable_passes = 0;
        }
        self.last = Some(stats);
    }
}

pub(crate) struct PreviewWork {
    pub size: [u32; 2],
    pub divisor: u32,
    pub samples: u32,
}

pub(crate) struct InFlight {
    completion: mpsc::Receiver<(Duration, bool)>,
    preview: Option<PreviewWork>,
}

pub(crate) struct GpuTiming {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    period: f32,
    validity_pipeline: wgpu::ComputePipeline,
    validity: wgpu::Buffer,
}

impl GpuTiming {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Viewport trace timestamps"), ty: wgpu::QueryType::Timestamp, count: 4,
        });
        let buffer = |size, usage| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Viewport measurements"), size, usage, mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Primary ray validity"), source: wgpu::ShaderSource::Wgsl(include_str!("viewport_metrics.wgsl").into()),
        });
        let validity_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Primary ray validity"), layout: None, module: &shader, entry_point: Some("count_valid"),
            compilation_options: Default::default(), cache: None,
        });
        Self { queries, resolve: buffer(32, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            read: buffer(40, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            period: queue.get_timestamp_period(), validity_pipeline,
            validity: buffer(4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC) }
    }

    pub fn writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.queries, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder, device: &wgpu::Device, preview: &wgpu::Texture, camera: &wgpu::Buffer) {
        let view = preview.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Viewport ray metrics"), layout: &self.validity_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.validity.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sample primary ray validity"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.queries, beginning_of_pass_write_index: Some(2), end_of_pass_write_index: Some(3),
                }),
            });
            pass.set_pipeline(&self.validity_pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.resolve_query_set(&self.queries, 0..4, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.read, 0, 32);
        encoder.copy_buffer_to_buffer(&self.validity, 0, &self.read, 32, 4);
    }

    fn read(&self) -> (Option<f64>, f64, Option<f64>) {
        let bytes = self.read.slice(..).get_mapped_range();
        let duration = |offset: usize| {
            let begin = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            let end = u64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
            (end >= begin && self.period > 0.0).then_some(end.wrapping_sub(begin) as f64 * f64::from(self.period) / 1e6)
        };
        let trace_ms = duration(0).filter(|&ms| ms > 0.0);
        let metrics_ms = duration(16);
        let valid = u32::from_le_bytes(bytes[32..36].try_into().unwrap());
        drop(bytes);
        self.read.unmap();
        (trace_ms, f64::from(valid.min(4096)) / 4096.0, metrics_ms)
    }
}

impl InFlight {
    pub fn submit(queue: &wgpu::Queue, timing: Option<&GpuTiming>, preview: Option<PreviewWork>) -> Self {
        let (send, completion) = mpsc::channel();
        let started = Instant::now();
        if let Some(timing) = timing.filter(|_| preview.is_some()) {
            timing.read.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                let _ = send.send((started.elapsed(), result.is_ok()));
            });
        } else {
            queue.on_submitted_work_done(move || { let _ = send.send((started.elapsed(), false)); });
        }
        Self { completion, preview }
    }

    pub fn poll(&self, timing: Option<&GpuTiming>) -> Option<Option<PassStats>> {
        let (elapsed, mapped) = match self.completion.try_recv() {
            Ok(result) => result,
            Err(mpsc::TryRecvError::Empty) => return None,
            Err(mpsc::TryRecvError::Disconnected) => return Some(None),
        };
        Some(self.preview.as_ref().map(|work| {
            let measurements = timing.filter(|_| mapped).map(GpuTiming::read);
            let gpu_ms = measurements.and_then(|(elapsed, _, _)| elapsed);
            let completion_ms = elapsed.as_secs_f64() * 1000.0;
            PassStats { size: work.size, divisor: work.divisor, rays: u64::from(work.size[0]) * u64::from(work.size[1]) * u64::from(work.samples),
                ms: gpu_ms.unwrap_or(completion_ms).max(0.001), gpu: gpu_ms.is_some(), completion_ms,
                valid_fraction: measurements.map(|(_, fraction, _)| fraction),
                metrics_ms: measurements.and_then(|(_, _, ms)| ms) }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotes_with_headroom_and_backs_off_on_expensive_frames() {
        let mut viewport = AdaptiveViewport::default();
        let pass = |divisor, ms| PassStats { divisor, size: [640 / divisor, 360 / divisor],
            rays: 640 * 360 / u64::from(divisor * divisor), ms, gpu: true, valid_fraction: Some(1.0),
            metrics_ms: None, completion_ms: ms };
        assert_eq!(viewport.divisor(), 4);
        viewport.observe(pass(4, 1.0));
        assert_eq!(viewport.divisor(), 4);
        viewport.observe(pass(4, 1.0));
        assert_eq!(viewport.divisor(), 2);
        viewport.observe(pass(2, 4.0));
        viewport.observe(pass(2, 4.0));
        assert_eq!(viewport.divisor(), 1);
        viewport.observe(pass(1, 27.0));
        assert_eq!(viewport.divisor(), 1);
        viewport.observe(pass(1, 40.0));
        assert_eq!(viewport.divisor(), 2);
        for _ in 0..8 {
            viewport.observe(pass(2, 4.0));
            assert_eq!(viewport.divisor(), 2);
        }
        viewport.restart();
        assert_eq!(viewport.divisor(), 4);
        assert!(viewport.last.is_none());
        viewport.observe(pass(4, 5.8));
        viewport.observe(pass(4, 5.8));
        assert_eq!(viewport.divisor(), 2);
        for _ in 0..4 { viewport.observe(pass(2, 16.7)); }
        assert_eq!(viewport.divisor(), 2);
    }
}
