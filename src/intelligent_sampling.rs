use wgpu::*;
use std::sync::mpsc;

const GEOMETRY_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

#[derive(Default)]
pub(super) struct IntelligentSampling {
    pub enabled: bool,
    resources: Option<Resources>,
}

pub(super) struct Resources {
    compute: ComputePipeline,
    allocation_layout: BindGroupLayout,
    textures: Textures,
    pipelines: [PipelineState; 2],
    view_changed: bool,
}

#[derive(Default)]
enum PipelineState {
    #[default]
    Unrequested,
    Compiling(mpsc::Receiver<Option<[RenderPipeline; 2]>>),
    Ready([RenderPipeline; 2]),
    Failed,
}

struct Textures {
    size: [u32; 2],
    geometry: TextureView,
    first_sample: Texture,
    compute_bind: BindGroup,
    allocation_bind: BindGroup,
    #[cfg(all(test, windows))]
    allocation: Texture,
}

impl IntelligentSampling {
    pub fn request(&mut self, device: &Device, camera: &BindGroupLayout, uniforms: &Buffer, accumulation: &Texture,
                   denoise: bool, wake: impl FnOnce() + Send + 'static) -> bool {
        let size = [accumulation.width(), accumulation.height()];
        let resources = self.resources.get_or_insert_with(|| Resources::new(device, uniforms, accumulation));
        if resources.view_changed || resources.textures.size != size {
            resources.textures = Textures::new(device, &resources.compute, &resources.allocation_layout, uniforms, accumulation);
            resources.view_changed = false;
        }
        let state = &mut resources.pipelines[usize::from(denoise)];
        match state {
            PipelineState::Unrequested => {
                let (send, receive) = mpsc::channel();
                let device = device.clone();
                let camera = camera.clone();
                let allocation = resources.allocation_layout.clone();
                std::thread::spawn(move || {
                    let pipelines = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        [false, true].map(|seed| Resources::create_pipeline(&device, &camera, &allocation, seed, denoise))
                    })).ok();
                    let _ = send.send(pipelines);
                    wake();
                });
                *state = PipelineState::Compiling(receive);
                false
            }
            PipelineState::Compiling(receiver) => match receiver.try_recv() {
                Ok(Some(pipelines)) => { *state = PipelineState::Ready(pipelines); true }
                Ok(None) | Err(mpsc::TryRecvError::Disconnected) => { *state = PipelineState::Failed; false }
                Err(mpsc::TryRecvError::Empty) => false,
            },
            PipelineState::Ready(_) => true,
            PipelineState::Failed => false,
        }
    }

    pub fn status(&self, denoise: bool) -> Option<&'static str> {
        let state = &self.resources.as_ref()?.pipelines[usize::from(denoise)];
        match state {
            PipelineState::Compiling(_) => Some("Preparing guided sampling"),
            PipelineState::Failed => Some("Guided sampling is unavailable"),
            _ => None,
        }
    }

    pub fn invalidate_view(&mut self) {
        if let Some(resources) = &mut self.resources { resources.view_changed = true; }
    }

    pub fn resources(&self) -> Option<&Resources> { self.resources.as_ref() }
}

#[cfg(all(test, windows))]
pub(super) fn compare_viewport_sampling(renderer: &mut super::Renderer, ctx: &egui::Context,
                                        mutation: &mut super::App, extent: [u32; 2]) -> Result<(), String> {
    use super::{App, Camera, Fractal, pixel_sampling::PixelSampling};
    use std::time::Instant;

    wait_for_pipelines(renderer, false)?;
    let mut measure = |app: &App, size: [u32; 2], guided: bool, samples: u32| -> Result<_, String> {
        renderer.intelligent_sampling.enabled = guided;
        let mut image = None;
        let start = Instant::now();
        for sample in 0..samples {
            let progress = Some(sample);
            let data = app.camera_uniforms(extent.map(|n| n as f32), progress, [0.0; 4], false);
            let gui = ctx.run(egui::RawInput::default(), |_| {});
            image = renderer.render(data, progress, true, None, size, gui, ctx, sample + 1 == samples, false, None);
            if !renderer.frame_submitted { return Err("Guided sampling test frame was not submitted".into()); }
        }
        let image = image.ok_or("Guided sampling capture unavailable")??;
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((image, ms))
    };
    let mut menger = App::default();
    menger.camera = Camera::framed_view(Fractal::Menger);
    let mut bulb = App::default();
    bulb.fractal = Fractal::Mandelbulb;
    bulb.camera = Camera::framed_view(bulb.fractal);
    for (name, scene) in [("Menger", &mut menger), ("Mandelbulb", &mut bulb), ("Mandelbox + Menger", &mut *mutation)] {
        scene.pixel_sampling = PixelSampling::Sobol;
        let (first, _) = measure(scene, extent, false, 1)?;
        let (guided_first, _) = measure(scene, extent, true, 1)?;
        let first_difference = first.as_raw().iter().zip(guided_first.as_raw())
            .map(|(a, b)| u64::from(a.abs_diff(*b))).sum::<u64>() as f64 / first.as_raw().len() as f64;
        if first_difference > 0.1 {
            return Err(format!("{name}: guided first-sample mean difference {first_difference:.3}"));
        }
        let _ = measure(scene, extent, false, 2)?;
        let _ = measure(scene, extent, true, 3)?;
        let (reference, _) = measure(scene, extent, false, 32)?;
        let (baseline, baseline_ms) = measure(scene, extent, false, 4)?;
        let (guided, guided_ms) = measure(scene, extent, true, 4)?;
        let (guided_refined, _) = measure(scene, extent, true, 16)?;
        if baseline == guided { return Err(format!("{name}: ray allocation did not affect the refinement")); }
        let rmse = |image: &image::RgbaImage| {
            let mut squared = 0.0;
            let mut edge_squared = 0.0;
            let mut edge_count = 0;
            for (x, y, expected) in reference.enumerate_pixels() {
                let actual = image.get_pixel(x, y);
                let error = (0..3).map(|channel| f64::from(expected[channel].abs_diff(actual[channel])).powi(2)).sum::<f64>();
                squared += error;
                let neighbors = [(x.saturating_sub(1), y), ((x + 1).min(extent[0] - 1), y),
                    (x, y.saturating_sub(1)), (x, (y + 1).min(extent[1] - 1))];
                if neighbors.into_iter().any(|(nx, ny)| {
                    let neighbor = reference.get_pixel(nx, ny);
                    (0..3).map(|channel| u32::from(expected[channel].abs_diff(neighbor[channel]))).sum::<u32>() >= 45
                }) {
                    edge_count += 1;
                    edge_squared += error;
                }
            }
            ((squared / (f64::from(extent[0]) * f64::from(extent[1]) * 3.0)).sqrt(),
             (edge_squared / f64::from(3 * edge_count.max(1))).sqrt())
        };
        let (baseline_all, baseline_edge) = rmse(&baseline);
        let (guided_all, guided_edge) = rmse(&guided);
        let (refined_all, _) = rmse(&guided_refined);
        if refined_all >= guided_all { return Err(format!("{name}: guided accumulation did not converge")); }
        println!("{name}: RMSE at 4 spp vs 32, all {baseline_all:.3}/{guided_all:.3}, edge {baseline_edge:.3}/{guided_edge:.3}; wall {baseline_ms:.1}/{guided_ms:.1} ms (uniform/guided); guided at 16 spp {refined_all:.3}");
    }
    let odd_size = extent.map(|n| (n / 2).saturating_sub(1).max(1));
    let _ = measure(mutation, odd_size, true, 2)?;
    let (favored, allocation_ms) = renderer.intelligent_sampling.validate_budget(&renderer.device, &renderer.queue, odd_size)?;
    if favored == 0 { return Err("No rays were redistributed in the odd-sized mutation preview".into()); }
    println!("{} × {}: {favored} pixels receive extra rays per guided pass; allocation GPU time {allocation_ms:?} ms", odd_size[0], odd_size[1]);
    renderer.intelligent_sampling.enabled = false;
    mutation.pixel_sampling = PixelSampling::R2;
    Ok(())
}

#[cfg(all(test, windows))]
pub(super) fn wait_for_pipelines(renderer: &mut super::Renderer, denoise: bool) -> Result<(), String> {
    let (done, ready) = mpsc::channel();
    if !renderer.prepare_intelligent_sampling(denoise, move || { let _ = done.send(()); }) {
        ready.recv_timeout(std::time::Duration::from_secs(20)).map_err(|e| e.to_string())?;
        if !renderer.prepare_intelligent_sampling(denoise, || {}) {
            return Err("Guided sampling pipelines did not compile".into());
        }
    }
    Ok(())
}

impl Resources {
    fn new(device: &Device, uniforms: &Buffer, accumulation: &Texture) -> Self {
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Ray allocation"),
            source: ShaderSource::Wgsl(include_str!("ray_allocation.wgsl").into()),
        });
        let compute = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Ray allocation"), layout: None, module: &shader, entry_point: Some("allocate"),
            compilation_options: Default::default(), cache: None,
        });
        let allocation_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Read ray allocation"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::FRAGMENT, count: None,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Uint,
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
            }],
        });
        let textures = Textures::new(device, &compute, &allocation_layout, uniforms, accumulation);
        Self { compute, allocation_layout, textures, pipelines: Default::default(), view_changed: false }
    }

    fn create_pipeline(device: &Device, camera: &BindGroupLayout, allocation: &BindGroupLayout,
                       seed: bool, denoise: bool) -> RenderPipeline {
        let mut groups = vec![camera];
        if !seed { groups.push(allocation); }
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Guided viewport layout"), bind_group_layouts: &groups, push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Guided viewport shader"), source: ShaderSource::Wgsl(super::FRACTAL_SHADER.into()),
        });
        let entry = match (seed, denoise) {
            (true, false) => "fs_guided_seed",
            (true, true) => "fs_guided_seed_denoise",
            (false, false) => "fs_guided",
            (false, true) => "fs_guided_denoise",
        };
        let target = Some(ColorTargetState {
            format: GEOMETRY_FORMAT,
            blend: (!seed).then_some(BlendState {
                color: BlendComponent { src_factor: BlendFactor::SrcAlpha, dst_factor: BlendFactor::OneMinusSrcAlpha,
                    operation: BlendOperation::Add },
                alpha: BlendComponent::REPLACE,
            }),
            write_mask: ColorWrites::ALL,
        });
        let targets = vec![target; 1 + usize::from(seed) + 2 * usize::from(denoise)];
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Guided viewport"), layout: Some(&layout),
            vertex: VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(FragmentState { module: &shader, entry_point: Some(entry), targets: &targets, compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        })
    }

    pub fn bind(&self, pass: &mut RenderPass<'_>, seed: bool, denoise: bool) {
        let PipelineState::Ready(pipelines) = &self.pipelines[usize::from(denoise)] else {
            unreachable!("Guided rendering starts only when its pipelines are ready")
        };
        pass.set_pipeline(&pipelines[usize::from(seed)]);
        if !seed { pass.set_bind_group(1, &self.textures.allocation_bind, &[]); }
    }

    pub fn geometry(&self) -> &TextureView { &self.textures.geometry }

    pub fn save_first_sample(&self, encoder: &mut CommandEncoder, accumulation: &Texture, [width, height]: [u32; 2]) {
        encoder.copy_texture_to_texture(accumulation.as_image_copy(), self.textures.first_sample.as_image_copy(),
            Extent3d { width, height, depth_or_array_layers: 1 });
    }

    pub fn allocate(&self, encoder: &mut CommandEncoder, size: [u32; 2]) {
        self.encode_allocation(encoder, size, None);
    }

    fn encode_allocation(&self, encoder: &mut CommandEncoder, [width, height]: [u32; 2],
                         timestamp_writes: Option<ComputePassTimestampWrites<'_>>) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("Allocate viewport rays"), timestamp_writes });
        pass.set_pipeline(&self.compute);
        pass.set_bind_group(0, &self.textures.compute_bind, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
}

impl Textures {
    fn new(device: &Device, compute: &ComputePipeline, allocation_layout: &BindGroupLayout, uniforms: &Buffer, accumulation: &Texture) -> Self {
        let [width, height] = [accumulation.width(), accumulation.height()];
        let texture = |label, format, usage| device.create_texture(&TextureDescriptor {
            label: Some(label), size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2,
            format, usage: usage | TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        let geometry = texture("Viewport guide geometry", GEOMETRY_FORMAT, TextureUsages::RENDER_ATTACHMENT)
            .create_view(&Default::default());
        let first_sample = texture("First uniform sample", GEOMETRY_FORMAT, TextureUsages::COPY_DST);
        let first_sample_view = first_sample.create_view(&Default::default());
        let mean_view = accumulation.create_view(&Default::default());
        let allocation = texture("Viewport ray counts", TextureFormat::R32Uint, TextureUsages::STORAGE_BINDING | TextureUsages::COPY_SRC);
        let allocation_view = allocation.create_view(&Default::default());
        let compute_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Generate ray allocation"), layout: &compute.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&geometry) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&allocation_view) },
                BindGroupEntry { binding: 2, resource: uniforms.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: BindingResource::TextureView(&first_sample_view) },
                BindGroupEntry { binding: 4, resource: BindingResource::TextureView(&mean_view) },
            ],
        });
        let allocation_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Read ray allocation"), layout: allocation_layout,
            entries: &[BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&allocation_view) }],
        });
        Self {
            size: [width, height], geometry, first_sample, compute_bind, allocation_bind,
            #[cfg(all(test, windows))]
            allocation,
        }
    }
}

#[cfg(all(test, windows))]
impl IntelligentSampling {
    pub fn validate_budget(&self, device: &Device, queue: &Queue, [width, height]: [u32; 2]) -> Result<(usize, Option<f64>), String> {
        let resources = self.resources.as_ref().ok_or("Ray guide was not prepared")?;
        let allocation = &resources.textures.allocation;
        let bytes_per_row = (width * 4).div_ceil(COPY_BYTES_PER_ROW_ALIGNMENT) * COPY_BYTES_PER_ROW_ALIGNMENT;
        let ray_bytes = u64::from(bytes_per_row) * u64::from(height);
        let timing = device.features().contains(Features::TIMESTAMP_QUERY).then(|| {
            let query_set = device.create_query_set(&QuerySetDescriptor { label: None, ty: QueryType::Timestamp, count: 2 });
            let resolve = device.create_buffer(&BufferDescriptor {
                label: None, size: 16, usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC, mapped_at_creation: false,
            });
            (query_set, resolve)
        });
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Ray allocation validation"), size: ray_bytes + u64::from(timing.is_some()) * 16,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(allocation.as_image_copy(),
            TexelCopyBufferInfo { buffer: &buffer, layout: TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bytes_per_row), rows_per_image: None } },
            Extent3d { width, height, depth_or_array_layers: 1 });
        if let Some((queries, resolve)) = &timing {
            resources.encode_allocation(&mut encoder, [width, height], Some(ComputePassTimestampWrites {
                query_set: queries, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1),
            }));
            encoder.resolve_query_set(queries, 0..2, resolve, 0);
            encoder.copy_buffer_to_buffer(resolve, 0, &buffer, ray_bytes, 16);
        }
        queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        let (send, ready) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |result| { let _ = send.send(result); });
        let _ = device.poll(Maintain::Wait);
        ready.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        let data = slice.get_mapped_range();
        let counts: Vec<_> = data[..ray_bytes as usize].chunks_exact(bytes_per_row as usize).flat_map(|row| {
            row[..(width * 4) as usize].chunks_exact(4).map(|pixel| u32::from_le_bytes(pixel.try_into().unwrap()))
        }).collect();
        for y in 0..height {
            for x in 0..width {
                let vertical = (x / 4 ^ y / 4) % 2 == 1;
                let [other_x, other_y] = if vertical { [x, y ^ 2] } else { [x ^ 2, y] };
                let count = counts[(y * width + x) as usize];
                let expected_sum = if other_x < width && other_y < height {
                    count + counts[(other_y * width + other_x) as usize]
                } else { count * 2 };
                if count > 2 || expected_sum != 2 {
                    return Err(format!("Invalid guided ray budget at pixel ({x}, {y})"));
                }
            }
        }
        let allocation_ms = timing.map(|_| {
            let timestamps = &data[ray_bytes as usize..];
            let begin = u64::from_le_bytes(timestamps[..8].try_into().unwrap());
            let end = u64::from_le_bytes(timestamps[8..].try_into().unwrap());
            end.wrapping_sub(begin) as f64 * f64::from(queue.get_timestamp_period()) / 1e6
        });
        Ok((counts.iter().filter(|&&count| count == 2).count(), allocation_ms))
    }
}
