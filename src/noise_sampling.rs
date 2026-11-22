use std::sync::{OnceLock, mpsc};
use wgpu::*;

pub(super) const MAX_SAMPLES: u32 = 4096;
const CHECK_INTERVAL: u32 = 4;

pub(super) fn viewport_tile_grid([width, height]: [u32; 2]) -> [u32; 2] {
    [width.div_ceil(crate::REFINE_TILE_WIDTH), height.div_ceil(crate::REFINE_TILE_HEIGHT)]
}

pub(super) struct ActiveTiles(Vec<(u32, u32)>);

impl ActiveTiles {
    pub fn len(&self) -> u32 { self.0.len() as u32 }

    pub fn tile(&self, index: u32) -> u32 { self.0[index as usize].0 }

    pub fn batch_end(&self, start: u32, full_tiles: u32) -> u32 {
        let full_cost = crate::REFINE_TILE_WIDTH * crate::REFINE_TILE_HEIGHT;
        let mut remaining = full_tiles * full_cost;
        let mut end = start;
        for &(_, pixels) in &self.0[start as usize..] {
            // Sparse tiles still cost raster and submission work.
            let cost = pixels.max(full_cost / 4);
            if cost > remaining { break; }
            remaining -= cost;
            end += 1;
        }
        end
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(super) struct Settings {
    pub enabled: bool,
    pub threshold: f32,
    pub min_samples: u32,
}

impl Settings {
    pub const fn new(threshold: f32) -> Self {
        Self { enabled: false, threshold, min_samples: 0 }
    }

    pub fn plan(self, max_samples: u32) -> Option<Plan> {
        let threshold = if self.threshold == 0.0 {
            (0.25 / (max_samples.max(1) as f32).sqrt()).clamp(0.001, 0.1)
        } else { self.threshold };
        let min = if self.min_samples == 0 {
            threshold.sqrt().recip().ceil().clamp(4.0, 64.0) as u32
        } else { self.min_samples };
        let first_check = min.max(2).next_multiple_of(2);
        (self.enabled && first_check < max_samples).then_some(Plan { threshold, first_check })
    }
}

#[derive(Clone, Copy)]
pub(super) struct Plan {
    threshold: f32,
    first_check: u32,
}

impl Plan {
    pub fn next_check(self, completed: u32) -> u32 {
        if completed < self.first_check { return self.first_check; }
        self.first_check + ((completed - self.first_check) / CHECK_INTERVAL + 1) * CHECK_INTERVAL
    }
}

pub(super) struct Pipelines {
    mask_layout: BindGroupLayout,
    check: ComputePipeline,
    samples: [OnceLock<RenderPipeline>; 2],
    export: bool,
}

impl Pipelines {
    pub fn ready(&self, denoise: bool) -> bool { self.samples[usize::from(denoise)].get().is_some() }

    pub fn new(device: &Device, export: bool) -> Self {
        let mask_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Read adaptive sampling mask"),
            entries: &[BindGroupLayoutEntry {
                binding: 0, visibility: ShaderStages::FRAGMENT, count: None,
                ty: BindingType::Buffer { ty: BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            }],
        });
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Adaptive sampling convergence"), source: ShaderSource::Wgsl(include_str!("noise_convergence.wgsl").into()),
        });
        let check = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Adaptive sampling convergence"), layout: None, module: &shader, entry_point: Some("check_noise"),
            compilation_options: Default::default(), cache: None,
        });
        Self { mask_layout, check, samples: Default::default(), export }
    }

    pub fn sample_pipeline(&self, device: &Device, camera: &BindGroupLayout, denoise: bool) -> &RenderPipeline {
        self.samples[usize::from(denoise)].get_or_init(|| {
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Adaptive sampling"), bind_group_layouts: &[camera, &self.mask_layout], push_constant_ranges: &[],
            });
            let shader = device.create_shader_module(ShaderModuleDescriptor {
                label: Some("Adaptive sampling"), source: ShaderSource::Wgsl(super::FRACTAL_SHADER.into()),
            });
            let entry = match (self.export, denoise) {
                (true, false) => "fs_noise_export",
                (true, true) => "fs_noise_export_denoise",
                (false, false) => "fs_noise_viewport",
                (false, true) => "fs_noise_viewport_denoise",
            };
            let target = |src_factor, dst_factor, alpha| Some(ColorTargetState {
                format: TextureFormat::Rgba16Float, write_mask: ColorWrites::ALL,
                blend: Some(BlendState {
                    color: BlendComponent { src_factor, dst_factor, operation: BlendOperation::Add },
                    alpha,
                }),
            });
            let average = |alpha| target(BlendFactor::Constant, BlendFactor::OneMinusConstant, alpha);
            let ever_uncertain = BlendComponent {
                src_factor: BlendFactor::One, dst_factor: BlendFactor::One, operation: BlendOperation::Max,
            };
            let mut targets = vec![average(ever_uncertain), target(BlendFactor::SrcAlpha, BlendFactor::OneMinusSrcAlpha, BlendComponent::REPLACE)];
            if denoise { targets.extend([average(BlendComponent::REPLACE), average(BlendComponent::REPLACE)]); }
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Adaptive sampling"), layout: Some(&layout),
                vertex: VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(FragmentState { module: &shader, entry_point: Some(entry), compilation_options: Default::default(), targets: &targets }),
                primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
            })
        })
    }
}

pub(super) struct Buffers {
    pub half_view: TextureView,
    pub mask_bind: BindGroup,
    mask: Buffer,
    remaining: Buffer,
    settings: Buffer,
    check_bind: BindGroup,
}

impl Buffers {
    pub fn new(device: &Device, pipelines: &Pipelines, accumulation: &Texture) -> Self {
        let size = accumulation.size();
        let half_view = device.create_texture(&TextureDescriptor {
            label: Some("Half-sample estimate"), size, mip_level_count: 1, sample_count: 1,
            dimension: TextureDimension::D2, format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING, view_formats: &[],
        }).create_view(&Default::default());
        let buffer = |label, size, usage| device.create_buffer(&BufferDescriptor {
            label: Some(label), size, usage, mapped_at_creation: false,
        });
        let mask_usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        #[cfg(test)]
        let mask_usage = mask_usage | BufferUsages::COPY_SRC;
        let mask = buffer("Converged pixels", (u64::from(size.width) * u64::from(size.height)).div_ceil(32) * 4,
            mask_usage);
        let [columns, rows] = viewport_tile_grid([size.width, size.height]);
        let count_words = if pipelines.export { 1 } else { 1 + columns * rows };
        let remaining = buffer("Remaining pixels", u64::from(count_words) * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC);
        let settings = buffer("Convergence settings", 28, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        let mean_view = accumulation.create_view(&Default::default());
        let check_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Check adaptive sampling convergence"), layout: &pipelines.check.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: BindingResource::TextureView(&mean_view) },
                BindGroupEntry { binding: 1, resource: BindingResource::TextureView(&half_view) },
                BindGroupEntry { binding: 2, resource: mask.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: remaining.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: settings.as_entire_binding() },
            ],
        });
        let mask_bind = device.create_bind_group(&BindGroupDescriptor {
            label: Some("Read adaptive sampling mask"), layout: &pipelines.mask_layout,
            entries: &[BindGroupEntry { binding: 0, resource: mask.as_entire_binding() }],
        });
        Self { half_view, mask_bind, mask, remaining, settings, check_bind }
    }

    pub fn reset(&self, encoder: &mut CommandEncoder) {
        encoder.clear_buffer(&self.mask, 0, None);
    }

    pub fn check(&self, device: &Device, queue: &Queue, encoder: &mut CommandEncoder, pipelines: &Pipelines,
                 [width, height]: [u32; 2], plan: Plan, completed: u32) -> Buffer {
        let [columns, rows] = if pipelines.export { [0, 0] } else { viewport_tile_grid([width, height]) };
        let tile_count = columns * rows;
        let words = [width, height, plan.threshold.to_bits(), columns, crate::REFINE_TILE_WIDTH, crate::REFINE_TILE_HEIGHT, completed];
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_ne_bytes).collect();
        queue.write_buffer(&self.settings, 0, &bytes);
        encoder.clear_buffer(&self.remaining, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor { label: Some("Check noise"), timestamp_writes: None });
            pass.set_pipeline(&pipelines.check);
            pass.set_bind_group(0, &self.check_bind, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        let readback_size = u64::from(1 + tile_count) * 4;
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("Convergence readback"), size: readback_size, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.remaining, 0, &readback, 0, readback_size);
        readback
    }
}

pub(super) struct Readback {
    buffer: Buffer,
    done: mpsc::Receiver<Result<(), BufferAsyncError>>,
}

impl Readback {
    pub fn new(buffer: Buffer) -> Self {
        let (sender, done) = mpsc::channel();
        buffer.slice(..).map_async(MapMode::Read, move |result| { let _ = sender.send(result); });
        Self { buffer, done }
    }

    fn count(&self) -> u32 {
        let bytes = self.buffer.slice(..).get_mapped_range();
        u32::from_ne_bytes(bytes[..4].try_into().unwrap())
    }

    pub fn poll(&self) -> Option<Option<(u32, ActiveTiles)>> {
        match self.done.try_recv() {
            Ok(Ok(())) => {
                let data = self.buffer.slice(..).get_mapped_range();
                let mut words = data.chunks_exact(4).map(|bytes| u32::from_ne_bytes(bytes.try_into().unwrap()));
                let count = words.next().unwrap();
                let tiles = words.enumerate().filter_map(|(id, pixels)| (pixels != 0).then_some((id as u32, pixels))).collect();
                Some(Some((count, ActiveTiles(tiles))))
            }
            Ok(Err(_)) | Err(mpsc::TryRecvError::Disconnected) => Some(None),
            Err(mpsc::TryRecvError::Empty) => None,
        }
    }

    pub fn wait(self) -> Result<u32, String> {
        self.done.recv().map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        Ok(self.count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_active_tiles_share_the_refinement_budget() {
        let full = crate::REFINE_TILE_WIDTH * crate::REFINE_TILE_HEIGHT;
        let tiles = ActiveTiles(vec![(2, full), (3, 1), (7, 1), (8, 1), (9, 1), (11, 1)]);
        assert_eq!(tiles.batch_end(0, 1), 1);
        assert_eq!(tiles.batch_end(1, 1), 5);
        assert_eq!(tiles.batch_end(5, 1), 6);
        assert_eq!(tiles.tile(2), 7);
    }

    #[test]
    fn dark_path_surfaces_remain_active_through_the_minimum() {
        let instance = Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&RequestAdapterOptions::default())) else { return; };
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor::default(), None)).unwrap();
        let size = Extent3d { width: 8, height: 8, depth_or_array_layers: 1 };
        let texture = device.create_texture(&TextureDescriptor {
            label: None, size, mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2,
            format: TextureFormat::Rgba16Float, usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let pipelines = Pipelines::new(&device, false);
        let buffers = Buffers::new(&device, &pipelines, &texture);
        let plan = Settings { enabled: true, threshold: 0.01, min_samples: 4 }.plan(64).unwrap();
        for (alpha, samples, remaining) in [(0.0, 4, 0), (1.0, 4, 64), (1.0, 32, 0)] {
            let mut encoder = device.create_command_encoder(&Default::default());
            buffers.reset(&mut encoder);
            let attachment = |view, a| Some(RenderPassColorAttachment {
                view, resolve_target: None,
                ops: Operations { load: LoadOp::Clear(Color { r: 0.0, g: 0.0, b: 0.0, a }), store: StoreOp::Store },
            });
            drop(encoder.begin_render_pass(&RenderPassDescriptor {
                label: None, color_attachments: &[attachment(&view, alpha), attachment(&buffers.half_view, 0.0)],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            }));
            let readback = buffers.check(&device, &queue, &mut encoder, &pipelines, [8, 8], plan, samples);
            let submission = queue.submit(Some(encoder.finish()));
            let readback = Readback::new(readback);
            let _ = device.poll(Maintain::WaitForSubmissionIndex(submission));
            assert_eq!(readback.wait().unwrap(), remaining);
        }
    }

    #[test]
    #[ignore = "GPU timing with controlled active-pixel masks"]
    fn benchmark_active_pixels() {
        benchmark_noise_sampling(640, 360, false);
    }

    #[test]
    #[ignore = "GPU timing of subdivided viewport tiles"]
    fn benchmark_viewport_subtiles() {
        benchmark_noise_sampling(1920, 1080, true);
    }

    fn benchmark_noise_sampling(width: u32, height: u32, subtiles: bool) {
        let instance = Instance::new(&InstanceDescriptor { backends: Backends::VULKAN, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance, ..Default::default()
        })).expect("Vulkan GPU");
        println!("GPU: {}", adapter.get_info().name);
        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            required_features: Features::TIMESTAMP_QUERY, ..Default::default()
        }, None)).unwrap();
        let camera = device.create_buffer(&BufferDescriptor {
            label: None, size: crate::CAMERA_UNIFORM_BYTES, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST, mapped_at_creation: false,
        });
        let camera_layout = crate::hdri::layout(&device, false);
        let environment = crate::hdri::upload(&device, &queue, 1, 1, &[0.0; 3]);
        let blue_noise = crate::pixel_sampling::blue_noise_view(&device, &queue);
        let camera_bind = crate::hdri::bind(&device, &camera_layout, &camera, &environment, &blue_noise);
        let accumulation = device.create_texture(&TextureDescriptor {
            label: Some("Benchmark accumulation"),
            size: Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: TextureDimension::D2, format: TextureFormat::Rgba16Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        let image = accumulation.create_view(&Default::default());
        let pipelines = Pipelines::new(&device, false);
        let buffers = Buffers::new(&device, &pipelines, &accumulation);
        let adaptive = pipelines.sample_pipeline(&device, &camera_layout, false);
        let uniform = crate::export::sample_pipeline(&device, &camera_layout, "fs_progress", false);
        let queries = device.create_query_set(&QuerySetDescriptor { label: None, ty: QueryType::Timestamp, count: 2 });
        let resolve = device.create_buffer(&BufferDescriptor {
            label: None, size: 16, usage: BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC, mapped_at_creation: false,
        });
        let timing = device.create_buffer(&BufferDescriptor {
            label: None, size: 16, usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let mut app = crate::App::default();
        app.effects.dof_enabled = true;
        app.effects.aperture = 0.025;
        app.effects.focus = 7.0;
        app.lighting.shadows = false;
        app.fractal = crate::Fractal::Mandelbulb;
        app.camera = crate::Camera::framed_view(app.fractal);
        app.mutation.formula = Some(crate::Fractal::Menger);
        app.mutation.weight = 0.3;
        let upload_camera = |app: &crate::App, index| {
            let mut values = app.camera_uniforms([width as f32, height as f32], Some(index), [0.0; 4], false);
            values[crate::NOISE_SAMPLING_UNIFORM] = width as f32;
            queue.write_buffer(&camera, 0, &values.into_iter().flat_map(f32::to_ne_bytes).collect::<Vec<_>>());
        };
        let render = |index: u32, use_mask: bool, scissors: &[[u32; 4]]| {
            let encode_start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            let attachment = |view| Some(RenderPassColorAttachment {
                view, resolve_target: None,
                ops: Operations {
                    load: if index == 0 { LoadOp::Clear(Color::BLACK) } else { LoadOp::Load },
                    store: StoreOp::Store,
                },
            });
            let mut targets = vec![attachment(&image)];
            if use_mask { targets.push(attachment(&buffers.half_view)); }
            {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: None, color_attachments: &targets, depth_stencil_attachment: None, occlusion_query_set: None,
                    timestamp_writes: Some(RenderPassTimestampWrites {
                        query_set: &queries, beginning_of_pass_write_index: Some(0), end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(if use_mask { adaptive } else { &uniform });
                pass.set_bind_group(0, &camera_bind, &[]);
                if use_mask { pass.set_bind_group(1, &buffers.mask_bind, &[]); }
                let weight = 1.0 / f64::from(index + 1);
                pass.set_blend_constant(Color { r: weight, g: weight, b: weight, a: 1.0 });
                for &[x, y, w, h] in scissors {
                    pass.set_scissor_rect(x, y, w, h);
                    pass.draw(0..3, 0..1);
                }
            }
            encoder.resolve_query_set(&queries, 0..2, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &timing, 0, 16);
            let submission = queue.submit(Some(encoder.finish()));
            let encode_ms = encode_start.elapsed().as_secs_f64() * 1e3;
            let slice = timing.slice(..);
            let (send, ready) = mpsc::channel();
            slice.map_async(MapMode::Read, move |result| { let _ = send.send(result); });
            let _ = device.poll(Maintain::WaitForSubmissionIndex(submission));
            ready.recv().unwrap().unwrap();
            let data = slice.get_mapped_range();
            let begin = u64::from_ne_bytes(data[..8].try_into().unwrap());
            let end = u64::from_ne_bytes(data[8..].try_into().unwrap());
            drop(data);
            timing.unmap();
            ((end - begin) as f64 * f64::from(queue.get_timestamp_period()) / 1e6, encode_ms)
        };
        let full = [[0, 0, width, height]];
        if subtiles {
            compare_subtiles(&device, &queue, &pipelines, &buffers, [width, height], &upload_camera, &render);
            return;
        }
        upload_camera(&app, 8);

        let measure = |active_groups: Option<u32>, scattered: bool| {
            if let Some(groups) = active_groups {
                let mut mask = vec![0u32; (width * height).div_ceil(32) as usize];
                for y in 0..height {
                    for x in 0..width {
                        let group = if scattered { (x + y * 2) % 4 } else { (x / 16 + y / 10) % 4 };
                        let index = y * width + x;
                        if group >= groups { mask[(index / 32) as usize] |= 1 << (index % 32); }
                    }
                }
                queue.write_buffer(&buffers.mask, 0, &mask.into_iter().flat_map(u32::to_ne_bytes).collect::<Vec<_>>());
            }
            let mut measurements: Vec<_> = (0..12).map(|_| render(8, active_groups.is_some(), &full).0).skip(3).collect();
            median(&mut measurements)
        };

        let cases = [
            ("Uniform", None, false), ("100% adaptive", Some(4), false), ("75% grouped", Some(3), false),
            ("50% grouped", Some(2), false), ("25% grouped", Some(1), false), ("50% scattered", Some(2), true),
            ("0%", Some(0), false),
        ];
        let forwards = cases.map(|(_, groups, scattered)| measure(groups, scattered));
        let backwards: Vec<_> = cases.iter().rev().map(|&(_, groups, scattered)| measure(groups, scattered)).collect();
        println!("640x360 Mandelbulb + Menger, DOF, GPU medians in ms (forward / reverse)");
        for ((name, _, _), (forward, reverse)) in cases.into_iter().zip(forwards.into_iter().zip(backwards.into_iter().rev())) {
            println!("{name}: {forward:.3} / {reverse:.3}");
        }
        assert!(forwards[6] < forwards[1], "converged pixels should skip tracing");
    }

    fn median(samples: &mut [f64]) -> f64 {
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }

    fn compare_subtiles(
        device: &Device, queue: &Queue, pipelines: &Pipelines, buffers: &Buffers, [width, height]: [u32; 2],
        upload_camera: &impl Fn(&crate::App, u32), render: &impl Fn(u32, bool, &[[u32; 4]]) -> (f64, f64),
    ) {
        let readback = device.create_buffer(&BufferDescriptor {
            label: Some("Benchmark convergence mask"), size: buffers.mask.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        let plan = Settings { enabled: true, threshold: 0.01, min_samples: 4 }.plan(64).unwrap();
        let full = [[0, 0, width, height]];
        for (name, dof, samples) in [
            ("Menger silhouette", false, 8), ("Menger silhouette", false, 32),
            ("Mandelbulb + Menger DOF", true, 8), ("Mandelbulb + Menger DOF", true, 32),
        ] {
            let mut app = crate::App::default();
            if dof {
                app.fractal = crate::Fractal::Mandelbulb;
                app.camera = crate::Camera::framed_view(app.fractal);
                app.mutation.formula = Some(crate::Fractal::Menger);
                app.mutation.weight = 0.3;
                app.effects.dof_enabled = true;
                app.effects.aperture = 0.025;
                app.effects.focus = 7.0;
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            buffers.reset(&mut encoder);
            queue.submit(Some(encoder.finish()));
            for index in 0..samples {
                upload_camera(&app, index);
                render(index, true, &full);
                if index + 1 != plan.next_check(index) { continue; }
                let mut encoder = device.create_command_encoder(&Default::default());
                let counts = buffers.check(device, queue, &mut encoder, pipelines, [width, height], plan, index + 1);
                let checkpoint = index + 1 == samples;
                if checkpoint {
                    encoder.copy_buffer_to_buffer(&buffers.mask, 0, &readback, 0, readback.size());
                }
                let submission = queue.submit(Some(encoder.finish()));
                if !checkpoint { continue; }
                let count_readback = Readback::new(counts);
                let slice = readback.slice(..);
                let (send, ready) = mpsc::channel();
                slice.map_async(MapMode::Read, move |result| { let _ = send.send(result); });
                let _ = device.poll(Maintain::WaitForSubmissionIndex(submission));
                ready.recv().unwrap().unwrap();
                let data = slice.get_mapped_range();
                let mask: Vec<_> = data.chunks_exact(4).map(|word| u32::from_ne_bytes(word.try_into().unwrap())).collect();
                drop(data);
                readback.unmap();
                let pending = count_readback.wait().unwrap();
                let (active, plans) = subtile_scissors(&mask, width, height);
                assert_eq!(active, pending);
                assert!(active > 0 && active < width * height, "{name} must be partially converged");
                let sample = index + 1;
                upload_camera(&app, sample);
                let mut results = [[(0.0, 0.0); 15]; 3];
                for run in 0..18 {
                    for offset in 0..plans.len() {
                        let mode = (run + offset) % plans.len();
                        let time = render(sample, true, &plans[mode]);
                        if run >= 3 { results[mode][run - 3] = time; }
                    }
                }
                println!("{name}, {sample} spp, {:.1}% active ({active}/{})",
                    100.0 * f64::from(active) / f64::from(width * height), width * height);
                for ((name, rects), times) in ["256x128", "64x32", "Hybrid"].into_iter().zip(plans.iter()).zip(results) {
                    let area: u32 = rects.iter().map(|rect| rect[2] * rect[3]).sum();
                    let gpu = median(&mut times.map(|time| time.0));
                    let cpu = median(&mut times.map(|time| time.1));
                    println!("  {name}: {} draws, {area} raster pixels, {gpu:.3} ms GPU, {cpu:.3} ms CPU encode+submit", rects.len());
                }
            }
        }
    }

    fn subtile_scissors(mask: &[u32], width: u32, height: u32) -> (u32, [Vec<[u32; 4]>; 3]) {
        let mut plans = [Vec::new(), Vec::new(), Vec::new()];
        let mut active = 0;
        let tile_width = crate::REFINE_TILE_WIDTH;
        let tile_height = crate::REFINE_TILE_HEIGHT;
        let small_width = tile_width / 4;
        let small_height = tile_height / 4;
        for y in (0..height).step_by(tile_height as usize) {
            for x in (0..width).step_by(tile_width as usize) {
                let right = (x + tile_width).min(width);
                let bottom = (y + tile_height).min(height);
                let mut children = Vec::new();
                for sy in (y..bottom).step_by(small_height as usize) {
                    for sx in (x..right).step_by(small_width as usize) {
                        let sr = (sx + small_width).min(right);
                        let sb = (sy + small_height).min(bottom);
                        let count: u32 = (sy..sb).map(|py| {
                            (sx..sr).filter(|&px| {
                                let index = py * width + px;
                                mask[(index / 32) as usize] & (1 << (index % 32)) == 0
                            }).count() as u32
                        }).sum();
                        active += count;
                        if count > 0 { children.push([sx, sy, sr - sx, sb - sy]); }
                    }
                }
                if children.is_empty() { continue; }
                let parent = [x, y, right - x, bottom - y];
                plans[0].push(parent);
                plans[1].extend_from_slice(&children);
                let covered_area: u32 = children.iter().map(|rect| rect[2] * rect[3]).sum();
                if covered_area <= parent[2] * parent[3] / 4 {
                    plans[2].extend_from_slice(&children);
                } else {
                    plans[2].push(parent);
                }
            }
        }
        (active, plans)
    }
}
