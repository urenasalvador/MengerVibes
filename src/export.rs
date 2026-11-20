use std::{fs::{self, OpenOptions}, path::PathBuf, sync::{Arc, OnceLock, atomic::{AtomicBool, Ordering}, mpsc::Sender}, time::{Duration, Instant, SystemTime, UNIX_EPOCH}};

const TILE_SIZE: u32 = 128;
const MAX_SAMPLE_BATCH: u32 = 8;
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

pub enum Update {
    Progress(f32),
    Finished(Result<PathBuf, String>),
}

pub struct Request {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
    pub uniforms: [f32; super::CAMERA_UNIFORM_FLOATS],
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
            let pipelines = pipelines.get_or_init(|| Pipelines::new(&device));
            render(&device, &queue, pipelines, &request, &cancel, |fraction| {
                let _ = updates.send(Update::Progress(fraction));
                window.request_redraw();
            })
        })).unwrap_or_else(|_| Err("Render failed unexpectedly".into()));
        let _ = updates.send(Update::Finished(outcome));
        window.request_redraw();
    });
}

pub struct Pipelines {
    samples: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
}

impl Pipelines {
    fn new(device: &wgpu::Device) -> Self {
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Export sample uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(super::CAMERA_UNIFORM_BYTES),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Export sample layout"), bind_group_layouts: &[&camera_layout], push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Export fractal"), source: wgpu::ShaderSource::Wgsl(super::FRACTAL_SHADER.into()),
        });
        let samples = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Export samples"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_export"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState { color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Constant, dst_factor: wgpu::BlendFactor::OneMinusConstant, operation: wgpu::BlendOperation::Add }, alpha: wgpu::BlendComponent::REPLACE }),
                write_mask: wgpu::ColorWrites::ALL,
            })] }), primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let shader_display = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Export display"), source: wgpu::ShaderSource::Wgsl(super::COMPOSITE_SHADER.into()),
        });
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Export output"), layout: None,
            vertex: wgpu::VertexState { module: &shader_display, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader_display, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        Self { samples, composite, camera_layout }
    }
}

fn render_pixels(
    device: &wgpu::Device, queue: &wgpu::Queue, pipelines: &Pipelines,
    request: &Request, cancel: &AtomicBool, tile: u32, mut progress: impl FnMut(f32),
) -> Result<image::RgbaImage, String> {
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
    let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None, layout: &pipelines.camera_layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &uniform, offset: 0, size: wgpu::BufferSize::new(super::CAMERA_UNIFORM_BYTES),
        }) }],
    });
    let tile_width = tile.min(request.width);
    let tile_height = tile.min(request.height);
    let (accum, composite_bind) = super::Renderer::create_accumulation(
        device, &pipelines.composite, &uniform, tile_width, tile_height,
    );
    let accum_view = accum.create_view(&Default::default());
    let out = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export tile"),
        size: wgpu::Extent3d { width: tile_width, height: tile_height, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    });
    let out_view = out.create_view(&Default::default());
    let row = (tile_width * 4).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export readback"), size: u64::from(row) * u64::from(tile_height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
    });

    let mut final_pixels = vec![0u8; request.width as usize * request.height as usize * 4];
    let tiles_x = request.width.div_ceil(tile);
    let tiles_y = request.height.div_ceil(tile);
    let total = u64::from(tiles_x) * u64::from(tiles_y) * u64::from(request.samples);
    let mut completed = 0u64;
    let mut last_progress = Instant::now();
    let mut watchdog = RenderWatchdog::default();
    let mut values = request.uniforms;
    values[3] = request.width as f32 / request.height as f32;
    values[10] = request.height as f32;
    values[11] = 0.0;
    for ty in 0..tiles_y {
        for tx in 0..tiles_x {
            let x = tx * tile;
            let y = ty * tile;
            let width = tile.min(request.width - x);
            let height = tile.min(request.height - y);
            values[36..40].copy_from_slice(&[x as f32, y as f32, request.width as f32, request.height as f32]);
            let mut sample = 0;
            // Probe each tile with one sample before batching. Cost can vary sharply
            // across the image, even when tiles have the same dimensions.
            let mut batch_limit = 1;
            while sample < request.samples {
                if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
                let batch = batch_limit.min(request.samples - sample);
                let final_batch = sample + batch == request.samples;
                let mut encoder = device.create_command_encoder(&Default::default());
                for slot in 0..batch {
                    let index = sample + slot;
                    values[22] = index as f32;
                    let offset = (stride * slot) as usize;
                    for (bytes, value) in batch_bytes[offset..][..super::CAMERA_UNIFORM_BYTES as usize].chunks_exact_mut(4).zip(values) {
                        bytes.copy_from_slice(&value.to_ne_bytes());
                    }
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Export sample"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &accum_view, resolve_target: None,
                            ops: wgpu::Operations { load: if index == 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
                        })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
                    });
                    pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                    pass.set_pipeline(&pipelines.samples);
                    pass.set_bind_group(0, &camera_bind, &[stride * slot]);
                    let weight = 1.0 / (index + 1) as f64;
                    pass.set_blend_constant(wgpu::Color { r: weight, g: weight, b: weight, a: 1.0 });
                    pass.draw(0..3, 0..1);
                }
                if final_batch {
                    {
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
                    encoder.copy_texture_to_buffer(
                        wgpu::TexelCopyTextureInfo { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) } },
                        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    );
                }
                queue.write_buffer(&uniform, 0, &batch_bytes[..(stride * batch) as usize]);
                let started = Instant::now();
                let submission = queue.submit(Some(encoder.finish()));
                let readback = final_batch.then(|| {
                    let (sender, receiver) = std::sync::mpsc::channel();
                    buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });
                    receiver
                });
                // Wait only for this bounded batch and check for cancellation before the next.
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
                if let Some(receiver) = readback {
                    receiver.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
                    let mapped = buffer.slice(..).get_mapped_range();
                    for local_y in 0..height as usize {
                        let dest = ((y as usize + local_y) * request.width as usize + x as usize) * 4;
                        let source = local_y * row as usize;
                        let bytes = width as usize * 4;
                        final_pixels[dest..dest + bytes].copy_from_slice(&mapped[source..source + bytes]);
                    }
                    drop(mapped);
                    buffer.unmap();
                }
                sample += batch;
                completed += u64::from(batch);
                if completed == total || last_progress.elapsed() >= PROGRESS_INTERVAL {
                    progress(completed as f32 / total as f32);
                    last_progress = Instant::now();
                }
            }
        }
    }
    if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
    image::RgbaImage::from_raw(request.width, request.height, final_pixels).ok_or_else(|| "Invalid output dimensions".into())
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
        let pipelines = Pipelines::new(&device);
        let mut uniforms = [0.0f32; crate::CAMERA_UNIFORM_FLOATS];
        uniforms[80..83].copy_from_slice(&crate::DEFAULT_BACKGROUND);
        uniforms[2] = 3.2; uniforms[8] = 8.0; uniforms[17] = 0.002; uniforms[18] = 1.0; uniforms[20] = 2.0;
        uniforms[40..48].copy_from_slice(&[1.8, 1.5, 0.035, 0.4, 0.16, 0.20, 0.25, 1.0]);
        let mut request = Request { width: 130, height: 130, samples: 4, uniforms };
        let cancel = AtomicBool::new(false);
        let tiled = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        let whole = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_eq!(tiled.dimensions(), (130, 130));
        let colors: std::collections::HashSet<_> = tiled.pixels().map(|p| p.0).collect();
        assert!(colors.len() > 8, "render did not contain a varied image");
        let max_difference = tiled.as_raw().iter().zip(whole.as_raw()).map(|(a, b)| a.abs_diff(*b)).max().unwrap();
        assert!(max_difference <= 2, "tile mismatch: {max_difference}");
        request.uniforms[51] = 1.25;
        let contrasted = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(whole, contrasted, "contrast should change the composited image");
        request.uniforms[51] = 1.0;
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
        request.uniforms[12] = 1.0; // Pearl matcap
        let unrotated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        request.uniforms[52] = std::f32::consts::FRAC_PI_2;
        let rotated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 256, |_| {}).unwrap();
        assert_ne!(unrotated, rotated, "rotating Pearl should change its shading");
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
        parameters.depth = 14;
        parameters.rotation = [18.0, -27.0, 138.0];
        parameters.tunnel_width = 1.2;
        request.uniforms[64..80].copy_from_slice(&parameters.uniforms());
        let mutated = render_pixels(&device, &queue, &pipelines, &request, &cancel, 128, |_| {}).unwrap();
        use crate::Fractal::{AmazingSurface, Kifs, Mandelbox, Mandelbulb, Menger, PseudoKleinian};
        let mut folded = crate::App::default();
        let render_formula = |app: &crate::App| {
            let request = Request {
                width: 64,
                height: 64,
                samples: 1,
                uniforms: app.camera_uniforms([64.0, 64.0], None, [0.0; 4], true),
            };
            render_pixels(&device, &queue, &pipelines, &request, &cancel, 64, |_| {}).unwrap()
        };
        for (base, secondary) in [
            (Kifs, AmazingSurface), (AmazingSurface, Kifs), (Mandelbulb, Kifs),
            (Mandelbox, AmazingSurface), (Menger, Kifs), (Mandelbox, Menger),
            (PseudoKleinian, AmazingSurface), (AmazingSurface, PseudoKleinian),
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
