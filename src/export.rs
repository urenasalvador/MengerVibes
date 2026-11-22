use std::{fs::{self, OpenOptions}, path::PathBuf, sync::{Arc, OnceLock, atomic::{AtomicBool, Ordering}, mpsc::Sender}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};
use crate::{denoise, noise_sampling};

const TILE_SIZE: u32 = 128;
const MAX_SAMPLE_BATCH: u32 = 8;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

pub enum Update {
    Progress(f32),
    Denoising(denoise::Backend),
    Finished(Result<PathBuf, String>),
}

pub struct Request {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub noise: noise_sampling::Settings,
    pub time_limit: Duration,
    pub denoise: Option<denoise::Backend>,
    pub uniforms: [f32; super::CAMERA_UNIFORM_FLOATS],
    pub environment: Option<super::hdri::Environment>,
}

// The GPU cannot be interrupted mid-submission. Small submissions bound the
// amount of work already queued, while this detects sustained slowdowns.
#[derive(Default)]
struct RenderWatchdog { slow_batches: u32 }
impl RenderWatchdog {
    fn observe(&mut self, elapsed: Duration) -> Result<(), String> {
        if elapsed >= Duration::from_millis(1000) {
            return Err("Render stopped: GPU work took too long. Lower Detail quality in Surface and try again.".into());
        }
        if elapsed >= Duration::from_millis(400) {
            self.slow_batches += 1;
        } else {
            self.slow_batches = 0;
        }
        if self.slow_batches >= 2 {
            return Err("Render stopped: GPU is responding slowly. Lower Detail quality in Surface and try again.".into());
        }
        Ok(())
    }
}

pub fn spawn(device: wgpu::Device, queue: wgpu::Queue, pipelines: Arc<OnceLock<Pipelines>>, request: Request, cancel: Arc<AtomicBool>, updates: Sender<Update>, window: Arc<winit::window::Window>) {
    std::thread::spawn(move || {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
            let pipelines = pipelines.get_or_init(|| Pipelines::new(&device, &queue));
            render(&device, &queue, pipelines, &request, &cancel, |fraction| {
                let update = match request.denoise {
                    Some(backend) if fraction == 1.0 => Update::Denoising(backend),
                    _ => Update::Progress(fraction),
                };
                let _ = updates.send(update);
                window.request_redraw();
            })
        })).unwrap_or_else(|_| Err("Render failed unexpectedly".into()));
        let _ = updates.send(Update::Finished(outcome));
        window.request_redraw();
    });
}

pub struct Pipelines {
    samples: wgpu::RenderPipeline,
    denoise_samples: OnceLock<wgpu::RenderPipeline>,
    composite: wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
    blue_noise: wgpu::TextureView,
    noise: OnceLock<noise_sampling::Pipelines>,
}

impl Pipelines {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let camera_layout = super::hdri::layout(device, true);
        let samples = sample_pipeline(device, &camera_layout, "fs_export", false);
        let shader_display = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Export display"), source: wgpu::ShaderSource::Wgsl(super::COMPOSITE_SHADER.into()),
        });
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Export output"), layout: None,
            vertex: wgpu::VertexState { module: &shader_display, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader_display, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let blue_noise = super::pixel_sampling::blue_noise_view(device, queue);
        Self { samples, denoise_samples: OnceLock::new(), composite, camera_layout, blue_noise, noise: OnceLock::new() }
    }

}

pub(super) fn sample_pipeline(device: &wgpu::Device, camera_layout: &wgpu::BindGroupLayout, entry: &'static str, denoise: bool) -> wgpu::RenderPipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Export sample layout"), bind_group_layouts: &[camera_layout], push_constant_ranges: &[],
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export fractal"), source: wgpu::ShaderSource::Wgsl(super::FRACTAL_SHADER.into()),
    });
    let target = Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba16Float,
        blend: Some(wgpu::BlendState { color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Constant, dst_factor: wgpu::BlendFactor::OneMinusConstant, operation: wgpu::BlendOperation::Add }, alpha: wgpu::BlendComponent::REPLACE }),
        write_mask: wgpu::ColorWrites::ALL,
    });
    let targets = vec![target; if denoise { 3 } else { 1 }];
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Export samples"), layout: Some(&layout),
        vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
        fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some(entry), compilation_options: Default::default(), targets: &targets }),
        primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
    })
}

enum Captured {
    Png(image::RgbaImage),
    Hdr(denoise::Frame),
}

fn render_pixels(
    device: &wgpu::Device, queue: &wgpu::Queue, pipelines: &Pipelines,
    request: &Request, cancel: &AtomicBool, tile: u32, progress: impl FnMut(f32),
) -> Result<image::RgbaImage, String> {
    let runtime = request.denoise.map(denoise::Runtime::load).transpose()?;
    match capture_pixels(device, queue, pipelines, request, cancel, tile, progress)? {
        Captured::Png(image) => Ok(image),
        Captured::Hdr(frame) => frame.finish(runtime.as_ref().expect("HDR capture uses OIDN"), request.uniforms[51], cancel),
    }
}

fn capture_pixels(
    device: &wgpu::Device, queue: &wgpu::Queue, pipelines: &Pipelines,
    request: &Request, cancel: &AtomicBool, tile: u32, mut progress: impl FnMut(f32),
) -> Result<Captured, String> {
    if request.width < 64 || request.height < 64 || request.width > 8192 || request.height > 8192 || request.samples == 0 {
        return Err("Resolution must be 64-8192 pixels and samples must be positive".into());
    }
    if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }

    let alignment = device.limits().min_uniform_buffer_offset_alignment;
    let stride = (super::CAMERA_UNIFORM_BYTES as u32).div_ceil(alignment) * alignment;
    let mut batch_bytes = vec![0; (stride * MAX_SAMPLE_BATCH) as usize];
    let uniform = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export settings"), size: batch_bytes.len() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
    });
    let environment = request.environment.clone().unwrap_or_else(|| super::hdri::upload(device, queue, 1, 1, &[0.0; 3]));
    let camera_bind = super::hdri::bind(device, &pipelines.camera_layout, &uniform, &environment, &pipelines.blue_noise);
    let tile_width = tile.min(request.width);
    let tile_height = tile.min(request.height);
    let (accum, composite_bind) = super::Renderer::create_accumulation(
        device, &pipelines.composite, &uniform, tile_width, tile_height,
    );
    let accum_view = accum.create_view(&Default::default());
    let denoise = request.denoise.is_some();
    let guides = denoise.then(|| denoise::guide_textures(device, accum.size()));
    let guide_views = guides.as_ref().map(|textures| textures.each_ref().map(|texture| texture.create_view(&Default::default())));
    let mut attachments = vec![&accum_view];
    if let Some(views) = &guide_views { attachments.extend(views); }
    let noise = request.noise.plan(request.samples).map(|plan| {
        let noise_pipelines = pipelines.noise.get_or_init(|| noise_sampling::Pipelines::new(device, true));
        let buffers = noise_sampling::Buffers::new(device, noise_pipelines, &accum);
        let samples = noise_pipelines.sample_pipeline(device, &pipelines.camera_layout, denoise);
        (plan, noise_pipelines, buffers, samples)
    });
    let pipeline = match &noise {
        Some((_, _, _, samples)) => samples,
        None if denoise => pipelines.denoise_samples.get_or_init(|| sample_pipeline(device, &pipelines.camera_layout, "fs_export_denoise", true)),
        None => &pipelines.samples,
    };
    let mut sample_attachments = attachments.clone();
    if let Some((_, _, buffers, _)) = &noise { sample_attachments.insert(1, &buffers.half_view); }
    let out = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export tile"),
        size: wgpu::Extent3d { width: tile_width, height: tile_height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    });
    let out_view = out.create_view(&Default::default());
    let pixel_size = if denoise { 8 } else { 4 };
    let row = (tile_width * pixel_size).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let plane_stride = u64::from(row) * u64::from(tile_height);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export readback"), size: plane_stride * attachments.len() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });

    let mut hdr_frame = denoise.then(|| denoise::Frame::new(request.width, request.height)).transpose()?;
    let mut final_pixels = if denoise { Vec::new() } else { vec![0u8; request.width as usize * request.height as usize * 4] };
    let tiles_x = request.width.div_ceil(tile);
    let tiles_y = request.height.div_ceil(tile);
    let tile_count = tiles_x * tiles_y;
    let total = u64::from(tile_count) * u64::from(request.samples);
    let deadline = (!request.time_limit.is_zero()).then(|| Instant::now() + request.time_limit);
    let mut completed = 0u64;
    let mut last_progress = Instant::now();
    let mut watchdog = RenderWatchdog::default();
    let mut values = request.uniforms;
    values[3] = request.width as f32 / request.height as f32;
    values[10] = request.height as f32;
    values[11] = 0.0;
    values[super::NOISE_SAMPLING_UNIFORM] = tile_width as f32;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x = tx * tile;
            let y = ty * tile;
            let width = tile.min(request.width - x);
            let height = tile.min(request.height - y);
            let remaining_pixels = request.width * (request.height - y) - x * height;
            let tile_deadline = deadline.map(|end| {
                let now = Instant::now();
                now + end.saturating_duration_since(now).mul_f64(f64::from(width * height) / f64::from(remaining_pixels))
            });
            values[36..40].copy_from_slice(&[x as f32, y as f32, request.width as f32, request.height as f32]);
            let copy_tile = |encoder: &mut wgpu::CommandEncoder| {
                if guides.is_none() {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export conversion"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &out_view, resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                        })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                    });
                    pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    pass.set_pipeline(&pipelines.composite);
                    pass.set_bind_group(0, &composite_bind, &[]);
                    pass.draw(0..3, 0..1);
                }
                let textures: &[&wgpu::Texture] = match &guides {
                    Some([albedo, normal]) => &[&accum, albedo, normal],
                    None => &[&out],
                };
                for (plane, texture) in textures.iter().enumerate() {
                    encoder.copy_texture_to_buffer(
                        texture.as_image_copy(),
                        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: plane as u64 * plane_stride, bytes_per_row: Some(row), rows_per_image: Some(height) } },
                        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    );
                }
            };
            let mut readback = None;
            let mut sample = 0;
            // Probe each tile with one sample before batching. Cost can vary sharply
            // across the image, even when tiles have the same dimensions.
            let mut batch_limit = 1;
            while sample < request.samples {
                if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
                if sample > 0 && tile_deadline.is_some_and(|end| Instant::now() >= end) { break; }
                let next_check = noise.as_ref().map_or(request.samples, |(plan, _, _, _)| plan.next_check(sample));
                let batch = batch_limit.min(request.samples - sample).min(next_check - sample);
                let final_batch = sample + batch == request.samples;
                let mut encoder = device.create_command_encoder(&Default::default());
                if sample == 0 {
                    if let Some((_, _, buffers, _)) = &noise { buffers.reset(&mut encoder); }
                }
                for slot in 0..batch {
                    let index = sample + slot;
                    values[22] = index as f32;
                    let offset = (stride * slot) as usize;
                    for (bytes, value) in batch_bytes[offset..][..super::CAMERA_UNIFORM_BYTES as usize].chunks_exact_mut(4).zip(values) {
                        bytes.copy_from_slice(&value.to_ne_bytes());
                    }
                    let color_attachments: Vec<_> = sample_attachments.iter().map(|view| Some(wgpu::RenderPassColorAttachment {
                            view, resolve_target: None,
                            ops: wgpu::Operations { load: if index == 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
                    })).collect();
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export sample"), color_attachments: &color_attachments, depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                    });
                    pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &camera_bind, &[stride * slot]);
                    if let Some((_, _, buffers, _)) = &noise { pass.set_bind_group(1, &buffers.mask_bind, &[]); }
                    let weight = 1.0 / (index + 1) as f64;
                    pass.set_blend_constant(wgpu::Color { r: weight, g: weight, b: weight, a: 1.0 });
                    pass.draw(0..3, 0..1);
                }
                let convergence = noise.as_ref().filter(|_| sample + batch == next_check && next_check < request.samples)
                    .map(|(plan, pipelines, buffers, _)| buffers.check(device, queue, &mut encoder, pipelines, [width, height], *plan, sample + batch));
                if final_batch { copy_tile(&mut encoder); }
                queue.write_buffer(&uniform, 0, &batch_bytes[..(stride * batch) as usize]);
                let started = Instant::now();
                let submission = queue.submit(Some(encoder.finish()));
                let convergence = convergence.map(noise_sampling::Readback::new);
                if final_batch { readback = Some(map_tile(&buffer)); }
                let _ = device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
                let elapsed = started.elapsed();
                watchdog.observe(elapsed)?;
                batch_limit = if elapsed < Duration::from_millis(20) {
                    (batch_limit * 2).min(MAX_SAMPLE_BATCH)
                } else if elapsed > Duration::from_millis(60) {
                    (batch_limit / 2).max(1)
                } else {
                    batch_limit
                };
                if let Some(end) = tile_deadline {
                    let affordable = end.saturating_duration_since(Instant::now()).as_secs_f64()
                        / (elapsed.as_secs_f64().max(0.000_001) / f64::from(batch));
                    batch_limit = batch_limit.min((affordable as u32).max(1));
                }
                sample += batch;
                if completed + u64::from(sample) < total && last_progress.elapsed() >= PROGRESS_INTERVAL {
                    progress((completed + u64::from(sample)) as f32 / total as f32);
                    last_progress = Instant::now();
                }
                if let Some(check) = convergence {
                    if check.wait()? == 0 { break; }
                }
            }
            let readback = readback.unwrap_or_else(|| {
                let mut encoder = device.create_command_encoder(&Default::default());
                copy_tile(&mut encoder);
                let submission = queue.submit(Some(encoder.finish()));
                let readback = map_tile(&buffer);
                let _ = device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
                readback
            });
            readback.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
            let mapped = buffer.slice(..).get_mapped_range();
            if let Some(frame) = &mut hdr_frame {
                frame.write_tile(&mapped, plane_stride as usize, row as usize, [x, y], [width, height]);
            } else {
                for local_y in 0..height as usize {
                    let dest = ((y as usize + local_y) * request.width as usize + x as usize) * 4;
                    let source = local_y * row as usize;
                    let bytes = width as usize * 4;
                    final_pixels[dest..dest + bytes].copy_from_slice(&mapped[source..source + bytes]);
                }
            }
            drop(mapped);
            buffer.unmap();
            completed += u64::from(request.samples);
        }
    }
    if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
    progress(1.0);
    if let Some(frame) = hdr_frame { return Ok(Captured::Hdr(frame)); }
    image::RgbaImage::from_raw(request.width, request.height, final_pixels).map(Captured::Png).ok_or_else(|| "Invalid output dimensions".into())
}

fn map_tile(buffer: &wgpu::Buffer) -> std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>> {
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });
    receiver
}

fn render(device: &wgpu::Device, queue: &wgpu::Queue, pipelines: &Pipelines, request: &Request, cancel: &AtomicBool, progress: impl FnMut(f32)) -> Result<PathBuf, String> {
    let image = render_pixels(device, queue, pipelines, request, cancel, TILE_SIZE, progress)?;
    if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
    fs::create_dir_all("renders").map_err(|e| e.to_string())?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|e| e.to_string())?.as_millis();
    let (path, mut file) = (0..1000).find_map(|n| {
        let path = PathBuf::from(format!("renders/render_{stamp}_{n}.png"));
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(file) => Some(Ok((path, file))),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => None,
            Err(e) => Some(Err(e.to_string())),
        }
    }).ok_or_else(|| "Could not allocate an output name".to_string())??;
    image::DynamicImage::ImageRgba8(image).write_to(&mut file, image::ImageFormat::Png).map_err(|e| e.to_string())?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "GPU render comparison of HDRI sampling convergence"]
    fn compare_hdri_importance_sampling() {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).expect("GPU");
        println!("GPU: {}", adapter.get_info().name);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let pipelines = Pipelines::new(&device, &queue);
        let mut app = crate::App::default();
        app.lighting.path_traced = true;
        app.lighting.enabled = false;
        app.lighting.environment = crate::HDRI_ENVIRONMENT;
        app.lighting.environment_color = [1.0; 3];
        app.lighting.environment_rotation = 20.0;
        app.lighting.bounces = 3;
        let (width, height) = (64, 32);
        let mut rgb = vec![0.04; (width * height * 3) as usize];
        for y in 7..10 {
            for x in 53..56 {
                let start = ((y * width + x) * 3) as usize;
                rgb[start..start + 3].copy_from_slice(&[36.0, 28.0, 20.0]);
            }
        }
        let fallback = crate::hdri::upload(&device, &queue, 1, 1, &[0.0; 3]);
        let hdri = crate::hdri::upload(&device, &queue, width, height, &rgb);
        app.hdri = Some(crate::hdri::Panorama { path: "synthetic.hdr".into(), texture: hdri.clone() });
        let bsdf = hdri.without_importance(&fallback);
        let cancel = AtomicBool::new(false);
        let mut request = Request {
            width: 64, height: 64, samples: 1,
            noise: noise_sampling::Settings::new(0.01), time_limit: Duration::ZERO, denoise: None,
            uniforms: app.camera_uniforms([64.0, 64.0], None, [0.0; 4], true),
            environment: None,
        };
        let mut measure = |samples, environment: &crate::hdri::Environment, tile_size| {
            request.samples = samples;
            request.environment = Some(environment.clone());
            let start = Instant::now();
            let pixels = render_pixels(&device, &queue, &pipelines, &request, &cancel, tile_size, |_| {}).unwrap();
            (pixels, start.elapsed().as_secs_f64() * 1e3)
        };
        measure(1, &hdri, 64);
        let (reference, _) = measure(512, &hdri, 64);
        let error = |image: &image::RgbaImage| {
            let mut square = 0.0;
            let mut bias = 0.0;
            let mut count = 0;
            let background = reference.get_pixel(0, 0);
            for (actual, expected) in image.pixels().zip(reference.pixels()) {
                if expected == background { continue; }
                for channel in 0..3 {
                    let difference = f64::from(actual[channel]) - f64::from(expected[channel]);
                    square += difference * difference;
                    bias += difference;
                    count += 1;
                }
            }
            assert!(count > 0);
            ((square / f64::from(count)).sqrt(), bias / f64::from(count))
        };
        for samples in [8, 32] {
            let mut bsdf_times = Vec::new();
            let mut mis_times = Vec::new();
            let mut previous = None;
            let mut combined = None;
            for run in 0..10 {
                let (a, b) = if run % 2 == 0 {
                    (measure(samples, &bsdf, 64), measure(samples, &hdri, 64))
                } else {
                    let b = measure(samples, &hdri, 64);
                    (measure(samples, &bsdf, 64), b)
                };
                bsdf_times.push(a.1);
                mis_times.push(b.1);
                previous = Some(a.0);
                combined = Some(b.0);
            }
            bsdf_times.sort_by(f64::total_cmp);
            mis_times.sort_by(f64::total_cmp);
            let bsdf_ms = (bsdf_times[4] + bsdf_times[5]) * 0.5;
            let mis_ms = (mis_times[4] + mis_times[5]) * 0.5;
            let (previous, combined) = (previous.unwrap(), combined.unwrap());
            let (bsdf_error, bsdf_bias) = error(&previous);
            let (mis_error, _) = error(&combined);
            println!("{samples} spp: BSDF RMSE {bsdf_error:.2}, bias {bsdf_bias:.2}, {bsdf_ms:.1} ms; MIS RMSE {mis_error:.2}, {mis_ms:.1} ms");
            assert!(mis_error < bsdf_error);
        }
        let (previous, _) = measure(512, &bsdf, 64);
        let (bsdf_error, bsdf_bias) = error(&previous);
        println!("512 spp: BSDF RMSE {bsdf_error:.2}, bias {bsdf_bias:.2} relative to MIS");
        assert!(bsdf_bias.abs() < 3.0);
        let (whole, _) = measure(8, &hdri, 64);
        let (tiled, _) = measure(8, &hdri, 32);
        assert!(whole.as_raw().iter().zip(tiled.as_raw()).all(|(a, b)| a.abs_diff(*b) <= 2));
    }

    #[test]
    fn watchdog_stops_sustained_or_severe_delays() {
        let mut watch = RenderWatchdog::default();
        assert!(watch.observe(Duration::from_millis(450)).is_ok());
        assert!(watch.observe(Duration::from_millis(30)).is_ok());
        assert!(watch.observe(Duration::from_millis(450)).is_ok());
        assert!(watch.observe(Duration::from_millis(450)).is_err());
        assert!(RenderWatchdog::default().observe(Duration::from_millis(1000)).is_err());
    }

    #[test]
    fn export_tiles_match_single_tile_and_cancel() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else { return; };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();
        let pipelines = Pipelines::new(&device, &queue);
        let mut uniforms = [0.0f32; crate::CAMERA_UNIFORM_FLOATS];
        uniforms[80..83].copy_from_slice(&crate::DEFAULT_BACKGROUND);
        uniforms[2] = 3.2; uniforms[8] = 8.0; uniforms[17] = 0.002; uniforms[18] = 1.0; uniforms[20] = 2.0;
        uniforms[24..28].copy_from_slice(&[0.18, 0.48, 0.62, 0.5]);
        uniforms[28..36].copy_from_slice(&[1.0, 1.0, 1.0, 0.0, 0.6, 0.8, 1.0, 0.0]);
        uniforms[40..48].copy_from_slice(&[1.8, 1.5, 0.035, 0.4, 0.16, 0.20, 0.25, 1.0]);
        let mut request = Request { width: 130, height: 130, samples: 4, noise: noise_sampling::Settings::new(0.01),
            time_limit: Duration::ZERO, denoise: None, uniforms, environment: None };
        let cancel = AtomicBool::new(false);
        let tiled = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        let whole = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_eq!(tiled.dimensions(), (130, 130));
        let colors: std::collections::HashSet<_> = tiled.pixels().map(|p| p.0).collect();
        assert!(colors.len() > 8, "render did not contain a varied image");
        let max_difference = tiled.as_raw().iter().zip(whole.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(max_difference <= 2, "tile mismatch: {max_difference}");
        request.samples = 1;
        let first = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        request.samples = 32;
        request.time_limit = Duration::from_nanos(1);
        let limited = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        assert_eq!(limited, first, "a small time limit must still cover every tile");
        request.time_limit = Duration::ZERO;
        request.samples = 8;
        request.noise = noise_sampling::Settings { enabled: true, threshold: 0.1, min_samples: 2 };
        let adaptive = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        assert_eq!(adaptive.dimensions(), (130, 130));
        assert!(adaptive.pixels().all(|pixel| pixel[3] == 255));
        request.denoise = Some(denoise::Backend::Cpu);
        let Captured::Hdr(adaptive_hdr) = capture_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap()
            else { panic!("Expected adaptive denoising guides"); };
        assert!(adaptive_hdr.guides()[0].iter().any(|rgb| rgb[0] > 0.0));
        request.denoise = None;
        request.noise.enabled = false;
        request.samples = 4;
        let mut compared = vec![tiled.clone()];
        for mode in [crate::pixel_sampling::PixelSampling::Sobol, crate::pixel_sampling::PixelSampling::Stbn] {
            request.uniforms[crate::PIXEL_SAMPLING_UNIFORM] = mode.uniform();
            let tiles = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
            let full = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
            let max_difference = tiles.as_raw().iter().zip(full.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
            assert!(max_difference <= 2, "{} tile mismatch: {max_difference}", mode.label());
            assert!(!compared.contains(&full), "{} must change the sampling", mode.label());
            compared.push(full);
        }
        request.uniforms[crate::PIXEL_SAMPLING_UNIFORM] = 0.0;
        request.denoise = Some(denoise::Backend::Cpu);
        let Captured::Hdr(hdr) = capture_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap()
            else { panic!("Expected linear render passes"); };
        let [albedo, normal] = hdr.guides();
        assert!(albedo.iter().any(|rgb| rgb[0] > 0.0));
        assert!(normal.iter().any(|rgb| rgb.iter().any(|channel| *channel < -0.01)));
        let linear_export = hdr.into_png(request.uniforms[51]).unwrap();
        let conversion_error = linear_export.as_raw().iter().zip(tiled.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(conversion_error <= 2, "HDR display conversion mismatch: {conversion_error}");
        for backend in [denoise::Backend::Cpu, denoise::Backend::Gpu].into_iter().filter(|backend| backend.available()) {
            request.denoise = Some(backend);
            let filtered = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
            assert_eq!(filtered.dimensions(), tiled.dimensions());
            assert_ne!(filtered, tiled, "OIDN should filter the PNG render");
        }
        request.denoise = None;
        request.uniforms[51] = 1.25;
        let contrasted = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(whole, contrasted, "contrast should change the composited image");
        request.uniforms[51] = 1.0;
        request.uniforms[crate::BOKEH_UNIFORM_OFFSET] = 3.0;
        let polygon = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(whole, polygon, "aperture blades should change the bokeh");
        request.uniforms[crate::BOKEH_UNIFORM_OFFSET + 1] = 0.6;
        let rotated_polygon = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(polygon, rotated_polygon, "aperture rotation should change the bokeh");
        request.uniforms[crate::BOKEH_UNIFORM_OFFSET..crate::BOKEH_UNIFORM_OFFSET + 2].fill(0.0);
        request.uniforms[42] = 0.0;
        let pinhole = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(whole, pinhole, "aperture should change the rendered image");
        request.uniforms[51] = 0.7;
        let flat = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(pinhole, flat, "contrast should also work without DOF");
        request.uniforms[51] = 1.0;
        request.uniforms[43] = 0.0;
        let clear = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(pinhole, clear, "fog density should change the rendered image");
        let mut pearl = crate::App::default();
        let mut legacy_material = pearl.preset_values();
        legacy_material[11] = 1.0; // Pearl in legacy presets.
        pearl.restore_scene(&legacy_material[..crate::PRESET_SURFACE_FIELDS]);
        assert_eq!(pearl.material, crate::STANDARD_MATERIAL);
        assert_eq!(pearl.lighting.color, [0.55, 0.61, 0.69]);
        let pearl_uniforms = pearl.camera_uniforms([130.0, 130.0], None, [0.0; 4], true);
        request.uniforms[24..36].copy_from_slice(&pearl_uniforms[24..36]);
        let lit_pearl = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        let finish = crate::SURFACE_UNIFORM_OFFSET;
        request.uniforms[finish..finish + 4].copy_from_slice(&[2.3, 0.8, 0.0, 0.3]);
        let higher_ior = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(lit_pearl, higher_ior, "the dielectric specular settings should affect preset materials");
        request.uniforms[finish + 2] = 1.0;
        let coated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(higher_ior, coated, "the coat should change the material response");
        request.uniforms[finish..finish + 4].fill(0.0);
        request.uniforms[34] = 0.0;
        let unlit_pearl = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(lit_pearl, unlit_pearl, "the migrated material should respond to the directional light");
        request.uniforms[finish + 4] = 1.0;
        for path_traced in [0.0, 1.0] {
            request.uniforms[crate::PATH_UNIFORM_OFFSET] = path_traced;
            let emissive = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
            assert!(emissive.pixels().any(|pixel| pixel[0] > 240 && pixel[1] < 10),
                "emission should remain visible with the directional light off");
        }
        request.uniforms[finish + 4] = 0.0;
        request.uniforms[crate::PATH_UNIFORM_OFFSET] = 0.0;
        request.uniforms[34] = pearl_uniforms[34];
        request.uniforms[9] = 1.0;
        let mut parameters = crate::MandelboxParameters::default();
        request.uniforms[56..60].copy_from_slice(&parameters.uniforms());
        let mandelbox = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        parameters.scale = -1.7;
        request.uniforms[56..60].copy_from_slice(&parameters.uniforms());
        let mutated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(mandelbox, mutated, "Mandelbox scale should change the rendered image");
        request.uniforms[9] = 0.0;
        request.uniforms[2] = 0.0;
        let inside = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        request.uniforms[60..63].copy_from_slice(&[0.4, 0.6, 0.25]);
        let mutated_inside = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(inside, mutated_inside, "Mandelbulb mutations should change its interior render");
        request.uniforms[2] = 3.2;
        request.uniforms[9] = crate::Fractal::Menger.value();
        let mut parameters = crate::MengerParameters::default();
        request.uniforms[64..80].copy_from_slice(&parameters.uniforms());
        let menger = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        let previous_uniforms = request.uniforms;
        request.uniforms[12] = 1.0; // Original matcap, enabled.
        let corner_view = crate::App::default().camera_uniforms([130.0; 2], None, [0.0; 4], true);
        request.uniforms[..6].copy_from_slice(&corner_view[..6]);
        request.denoise = Some(denoise::Backend::Cpu);
        let Captured::Hdr(matcap) = capture_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap()
            else { panic!("Expected a matcap albedo pass"); };
        let [albedo, _] = matcap.guides();
        let (min, max) = albedo.iter().filter(|rgb| rgb[2] > 0.05).map(|rgb| rgb[0] / rgb[2])
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), ratio| (min.min(ratio), max.max(ratio)));
        assert!(max - min > 0.10, "matcap hue should vary across faces: {min}–{max}");
        request.denoise = None;
        request.uniforms[crate::PATH_UNIFORM_OFFSET] = 1.0;
        request.uniforms[34] = 1.0;
        let lit_matcap = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        request.uniforms[34] = 0.0;
        let unlit_matcap = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(lit_matcap, unlit_matcap, "the path tracer should light the matcap material");
        request.uniforms = previous_uniforms;
        request.uniforms[24..28].copy_from_slice(&[0.5, 0.5, 0.5, 1.0]);
        request.uniforms[28..36].copy_from_slice(&[1.0, 1.0, 1.0, 0.0, 0.6, 0.8, 1.0, 0.0]);
        request.uniforms[80..83].fill(0.02);
        let diffuse_without_ao = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        request.uniforms[19] = 1.0; // Directional shadows
        let shadowed = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(diffuse_without_ao, shadowed, "geometry should block the directional light with AO off");
        assert!(shadowed.pixels().zip(diffuse_without_ao.pixels()).all(|(shadow, lit)| shadow[0] <= lit[0]));
        assert!(shadowed.pixels().zip(diffuse_without_ao.pixels()).any(|(shadow, lit)| shadow[0] == lit[0] && lit[0] > 100));
        let lighting = crate::LIGHTING_UNIFORM_OFFSET;
        request.uniforms[lighting + 3] = 1.0 - 12.0_f32.to_radians().cos();
        let soft_shadows = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        let tiled_soft_shadows = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        assert!(soft_shadows.pixels().zip(shadowed.pixels()).any(|(soft, hard)| soft[0] > hard[0].saturating_add(10)));
        let soft_tile_difference = soft_shadows.as_raw().iter().zip(tiled_soft_shadows.as_raw())
            .map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(soft_tile_difference <= 2, "soft shadow tile mismatch: {soft_tile_difference}");
        request.uniforms[lighting + 3] = 0.0;
        request.uniforms[19] = 0.0;
        request.uniforms[13..16].copy_from_slice(&[1.0, 1.0, 0.75]);
        let ray_traced_ao = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        let tiled_ao = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        assert_ne!(diffuse_without_ao, ray_traced_ao, "ray-traced AO must shade the fractal");
        assert!(ray_traced_ao.pixels().all(|p| p[0] == p[1] && p[1] == p[2]), "white light on diffuse gray must stay neutral");
        let max_ao_difference = tiled_ao.as_raw().iter().zip(ray_traced_ao.as_raw())
            .map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(max_ao_difference <= 2, "ray-traced AO tile mismatch: {max_ao_difference}");
        request.uniforms[13..16].fill(0.0);
        request.uniforms[34] = 0.0; // Directional light off
        let unlit = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        request.uniforms[lighting..lighting + 3].copy_from_slice(&[0.35, 0.16, 0.06]);
        let environment = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(environment.pixels().zip(unlit.pixels()).any(|(lit, dark)| lit[0] > lit[1] && lit[1] > lit[2] && lit[0] > dark[0]));
        assert_eq!(environment.get_pixel(0, 0), unlit.get_pixel(0, 0));
        request.uniforms[13..16].copy_from_slice(&[1.0, 1.0, 0.75]);
        let occluded_environment = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(occluded_environment.pixels().zip(environment.pixels()).any(|(occluded, lit)| occluded[0] < lit[0]));

        let path = crate::PATH_UNIFORM_OFFSET;
        request.uniforms[23] = 1.0; // Refined march and multiple bounces
        request.uniforms[path..path + 5].copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 0.0]);
        let one_bounce = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(unlit, one_bounce, "the path tracer must receive light when a ray escapes to the environment");
        request.uniforms[13..16].fill(0.0);
        assert_eq!(one_bounce, render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap(),
            "the legacy AO controls must not affect paths");
        request.uniforms[path + 3] = 3.0;
        let bounced = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(bounced.pixels().zip(one_bounce.pixels()).any(|(multi, one)| multi[0] > one[0].saturating_add(5)),
            "additional bounces should carry light to some surfaces");
        let tiled_paths = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        let tile_difference = tiled_paths.as_raw().iter().zip(bounced.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(tile_difference <= 2, "path-traced tile mismatch: {tile_difference}");

        request.uniforms[path + 1] = 3.0; // Copper environment
        let copper = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(copper, bounced, "studio presets should change the environment radiance");
        request.uniforms[path + 1] = crate::HDRI_ENVIRONMENT as f32;
        request.uniforms[lighting..lighting + 3].fill(0.5);
        let hdri_path = std::env::temp_dir().join(format!("mengervibes-test-{}.hdr", std::process::id()));
        image::codecs::hdr::HdrEncoder::new(std::fs::File::create(&hdri_path).unwrap())
            .encode(&[image::Rgb([0.125, 2.0, 0.25]); 8], 4, 2).unwrap();
        let panorama = crate::hdri::Panorama::load(&device, &queue, &hdri_path);
        std::fs::remove_file(hdri_path).unwrap();
        request.environment = Some(panorama.unwrap().texture);
        let hdri = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(hdri.pixels().any(|pixel| pixel[1] > pixel[0].saturating_add(50) && pixel[1] > pixel[2].saturating_add(50)),
            "HDRI radiance must light the surface");
        request.uniforms[path] = 0.0; // Ambient + AO
        let ambient_hdri = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(ambient_hdri.pixels().zip(unlit.pixels()).any(|(lit, dark)|
            lit[1] > dark[1].saturating_add(30) && lit[1] > lit[0].saturating_add(30)));
        assert_eq!(ambient_hdri.get_pixel(0, 0), unlit.get_pixel(0, 0));
        request.uniforms[13..16].copy_from_slice(&[1.0, 1.0, 0.75]);
        let occluded_hdri = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(ambient_hdri.pixels().zip(occluded_hdri.pixels()).any(|(unoccluded, occluded)|
            unoccluded[1] > occluded[1].saturating_add(5)));
        let stripes = (0..32).flat_map(|i| if i % 8 < 4 { [1.0, 0.02, 0.02] } else { [0.02, 0.02, 1.0] }).collect::<Vec<_>>();
        request.environment = Some(crate::hdri::upload(&device, &queue, 8, 4, &stripes));
        let rotated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        request.uniforms[path + 2] = std::f32::consts::PI;
        let opposite = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert!(rotated.pixels().zip(opposite.pixels()).any(|(first, second)| first[0].abs_diff(second[0]) > 10));
        request.environment = None;
        request.uniforms = previous_uniforms;
        parameters.depth = 14;
        parameters.rotation = [18.0, -27.0, 138.0];
        parameters.tunnel_width = 1.2;
        request.uniforms[64..80].copy_from_slice(&parameters.uniforms());
        let mutated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        use crate::Fractal::{AmazingSurface, Kifs, Mandelbox, Mandelbulb, Menger, PseudoKleinian, Sierpinski};
        let mut folded = crate::App::default();
        folded.lighting.shadows = false;
        let render_formula = |app: &crate::App| {
            let request = Request {
                width: 64,
                height: 64,
                samples: 1,
                noise: noise_sampling::Settings::new(0.01),
                time_limit: Duration::ZERO,
                denoise: None,
                uniforms: app.camera_uniforms([64.0, 64.0], None, [0.0; 4], true),
                environment: None,
            };
            render_pixels(&device, &queue, &pipelines, &request, &cancel, 64, |_| {}).unwrap()
        };
        folded.fractal = Sierpinski;
        folded.sierpinski.depth = 0;
        let solid = render_formula(&folded);
        folded.sierpinski.depth = 5;
        let subdivided = render_formula(&folded);
        let background = solid.get_pixel(0, 0);
        assert!(subdivided.pixels().any(|pixel| pixel != background), "Sierpinski should contain visible geometry");
        assert!(solid.pixels().zip(subdivided.pixels()).any(|(before, after)| before != background && after == background),
            "subdivision should open holes in the solid tetrahedron");
        for (base, secondary) in [
            (Kifs, AmazingSurface), (AmazingSurface, Kifs), (Mandelbulb, Kifs),
            (Mandelbox, AmazingSurface), (Menger, Kifs), (Mandelbox, Menger),
            (PseudoKleinian, AmazingSurface), (AmazingSurface, PseudoKleinian),
            (Menger, Sierpinski),
        ] {
            folded.fractal = base;
            folded.base_weight = 1.0;
            folded.mutation.formula = Some(secondary);
            folded.mutation.weight = 0.0;
            let base_image = render_formula(&folded);
            let colors: std::collections::HashSet<_> = base_image.pixels().map(|pixel| pixel.0).collect();
            assert!(colors.len() > 8, "{} should contain visible geometry", base.name());
            folded.mutation.weight = 0.5;
            let blended_image = render_formula(&folded);
            assert_ne!(base_image, blended_image, "{} should blend with {}", base.name(), secondary.name());
            folded.base_weight = 0.0;
            folded.mutation.weight = 1.0;
            let full_weight = render_formula(&folded);
            folded.fractal = secondary;
            folded.mutation.formula = None;
            assert_eq!(full_weight, render_formula(&folded), "full weight should render {}", secondary.name());
        }
        folded.fractal = Mandelbox;
        folded.base_weight = 1.0;
        folded.mutation.formula = Some(Mandelbulb);
        folded.mutation.weight = 0.6;
        let blend = render_formula(&folded);
        folded.mutation.bulb_power = 5.0;
        assert_ne!(blend, render_formula(&folded), "the mutation should use its own formula parameters");

        folded.fractal = Kifs;
        folded.mutation.formula = None;
        let base = render_formula(&folded);
        folded.mutation.formula = Some(AmazingSurface);
        folded.mutation.weight = 1.0;
        folded.extra_mutations.push(crate::MutationStage {
            formula: Some(PseudoKleinian), weight: 1.0, ..Default::default()
        });
        let equal_weights = render_formula(&folded);
        std::mem::swap(&mut folded.mutation, &mut folded.extra_mutations[0]);
        assert_eq!(equal_weights, render_formula(&folded), "relative weights ignore mutation order");
        folded.extra_mutations[0].weight = 0.0;
        assert_ne!(equal_weights, render_formula(&folded), "a weight of 1 must not hide another active formula");
        folded.base_weight = 0.0;
        let without_base = render_formula(&folded);
        folded.mutation.weight = 0.0;
        assert_eq!(base, render_formula(&folded), "all weights at zero should show the base");
        folded.fractal = PseudoKleinian;
        assert_eq!(without_base, render_formula(&folded));

        assert_ne!(menger, mutated, "rotating or widening Menger tunnels should change the render");
        request.uniforms[80..83].copy_from_slice(&[1.0, 0.0, 0.0]);
        let red_background = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        assert_eq!(red_background.get_pixel(0, 0).0, [255, 0, 0, 255]);
        let mut encoded = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(tiled).write_to(&mut encoded, image::ImageFormat::Png).unwrap();
        let decoded = image::load_from_memory(&encoded.into_inner()).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (130, 130));
        cancel.store(true, Ordering::Relaxed);
        assert!(matches!(render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}), Err(ref e) if e == "Render cancelled"));
    }
}
