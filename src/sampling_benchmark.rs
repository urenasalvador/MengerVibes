//! Offline comparison of the three primary-ray samplers.
use crate::{App, CAMERA_UNIFORM_BYTES, Camera, FRACTAL_SHADER, Fractal, MutationStage, hdri};
use std::path::Path;

const WIDTH: u32 = 256;
const HEIGHT: u32 = 144;
const REFERENCE: u32 = 256;
const MASK_SIDE: usize = 16;
const MASK_FRAMES: usize = 32;
const MASK_PLANE: usize = MASK_SIDE * MASK_SIDE;
const MASK_VOLUME: usize = MASK_PLANE * MASK_FRAMES;

const SAMPLER: &str = r#"
fn benchmark_offset(pixel: vec2<u32>, index: f32) -> vec2<f32> {
    if camera.four_branch_twist.z == 3.0 {
        let seed = pixel.x + 13763u * pixel.y;
        return vec2<f32>(pixel_random(seed + u32(index) * 71059u),
                         pixel_random(seed + u32(index) * 71059u + 98147u)) - 0.5;
    }
    return primary_offset(pixel, index, camera.four_branch_twist.w);
}
"#;

struct Mask {
    active: Vec<bool>,
    energy: Vec<f32>,
    neighbors: Vec<Vec<(usize, f32)>>,
}

impl Mask {
    fn new() -> Self {
        let neighbors = (0..MASK_VOLUME).map(|index| {
            let (x, y, t) = (index % MASK_SIDE, index / MASK_SIDE % MASK_SIDE, index / MASK_PLANE);
            let wrap = |value: usize, delta: i32, size: usize| (value as i32 + delta).rem_euclid(size as i32) as usize;
            let mut list = Vec::new();
            for dy in -4..=4 {
                for dx in -4..=4 {
                    if dx == 0 && dy == 0 { continue; }
                    let i = wrap(x, dx, MASK_SIDE) + wrap(y, dy, MASK_SIDE) * MASK_SIDE + t * MASK_PLANE;
                    list.push((i, (-(dx * dx + dy * dy) as f32 / 4.5).exp()));
                }
            }
            for dt in -4..=4 {
                if dt == 0 { continue; }
                let i = x + y * MASK_SIDE + wrap(t, dt, MASK_FRAMES) * MASK_PLANE;
                list.push((i, (-(dt * dt) as f32 / 4.5).exp()));
            }
            list
        }).collect();
        Self { neighbors, active: vec![false; MASK_VOLUME], energy: vec![0.0; MASK_VOLUME] }
    }

    fn set(&mut self, index: usize, active: bool) {
        if self.active[index] == active { return; }
        self.active[index] = active;
        let direction = if active { 1.0 } else { -1.0 };
        for &(neighbor, weight) in &self.neighbors[index] {
            self.energy[neighbor] += direction * weight;
        }
    }

    fn extremum(&self, active: bool, largest: bool) -> usize {
        let mut best = None;
        for (index, (&selected, &energy)) in self.active.iter().zip(&self.energy).enumerate() {
            if selected == active && best.is_none_or(|(_, previous)| {
                if largest { energy > previous } else { energy < previous }
            }) {
                best = Some((index, energy));
            }
        }
        best.unwrap().0
    }

    fn ranks(&mut self, mut seed: u32) -> Vec<u32> {
        let count = MASK_VOLUME / 10;
        let mut selected = 0;
        while selected < count {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let index = (seed >> 14) as usize % MASK_VOLUME;
            if !self.active[index] {
                self.set(index, true);
                selected += 1;
            }
        }
        // Interactions are spatial within one frame and temporal at one pixel.
        for _ in 0..MASK_VOLUME {
            let cluster = self.extremum(true, true);
            self.set(cluster, false);
            let void = self.extremum(false, false);
            self.set(void, true);
            if cluster == void { break; }
        }
        let pattern = self.active.clone();
        let mut ranks = vec![0; MASK_VOLUME];
        for rank in (0..count).rev() {
            let cluster = self.extremum(true, true);
            ranks[cluster] = rank as u32;
            self.set(cluster, false);
        }
        for (index, active) in pattern.into_iter().enumerate() {
            self.set(index, active);
        }
        for rank in count..MASK_VOLUME {
            let index = self.extremum(false, false);
            ranks[index] = rank as u32;
            self.set(index, true);
        }
        for index in 0..MASK_VOLUME { self.set(index, false); }
        self.energy.fill(0.0);
        ranks
    }
}

struct Probe {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    uniforms: wgpu::Buffer,
    camera: wgpu::BindGroup,
    target: wgpu::Texture,
    readback: wgpu::Buffer,
    stride: u32,
}

impl Probe {
    fn new() -> Self {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() });
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).expect("Vulkan adapter");
        println!("Adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let camera_layout = hdri::layout(&device, true);
        let entry = "primary_offset(position, index, 0.0)";
        assert!(FRACTAL_SHADER.contains(entry));
        let shader = FRACTAL_SHADER.replace(entry, "benchmark_offset(position, index)");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Local sampling benchmark"), source: wgpu::ShaderSource::Wgsl(format!("{shader}\n{SAMPLER}").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&camera_layout], push_constant_ranges: &[],
        });
        let format = wgpu::TextureFormat::Rgba16Float;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_export"), compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format, blend: Some(wgpu::BlendState { color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::Constant, dst_factor: wgpu::BlendFactor::OneMinusConstant,
                        operation: wgpu::BlendOperation::Add,
                    }, alpha: wgpu::BlendComponent::REPLACE }), write_mask: wgpu::ColorWrites::ALL,
                })] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let alignment = device.limits().min_uniform_buffer_offset_alignment;
        let stride = (CAMERA_UNIFORM_BYTES as u32).div_ceil(alignment) * alignment;
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: u64::from(stride * REFERENCE), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let environment = hdri::upload(&device, &queue, 1, 1, &[0.0; 3]);
        let blue_noise = crate::pixel_sampling::blue_noise_view(&device, &queue);
        let camera = hdri::bind(&device, &camera_layout, &uniforms, &environment, &blue_noise);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: u64::from(WIDTH * HEIGHT * 8),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
        });
        Self { device, queue, pipeline, uniforms, camera, target, readback, stride }
    }

    fn run(&self, app: &App, strategy: u32, samples: u32, moving: bool, frame: u32) -> Vec<[f32; 3]> {
        let mut values = app.camera_uniforms([WIDTH as f32, HEIGHT as f32], None, [0.0; 4], true);
        values[38..40].copy_from_slice(&[WIDTH as f32, HEIGHT as f32]);
        values[crate::PIXEL_SAMPLING_UNIFORM] = strategy as f32;
        values[crate::NAVIGATION_SAMPLE_UNIFORM] = (frame * samples) as f32;
        if moving {
            values[23] = 0.0;
            values[crate::FULL_PRECISION_PREVIEW_UNIFORM] = HEIGHT as f32;
            values[crate::MOVING_BOUNCES_UNIFORM] = app.moving_bounces as f32;
        }
        let mut data = vec![0; (self.stride * samples) as usize];
        for (i, slot) in data.chunks_exact_mut(self.stride as usize).enumerate() {
            values[22] = if strategy == 3 { (i + 16384) as f32 } else { i as f32 };
            for (bytes, value) in slot[..CAMERA_UNIFORM_BYTES as usize].chunks_exact_mut(4).zip(values) {
                bytes.copy_from_slice(&value.to_ne_bytes());
            }
        }
        self.queue.write_buffer(&self.uniforms, 0, &data);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let view = self.target.create_view(&Default::default());
        for i in 0..samples {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: if i == 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load },
                        store: wgpu::StoreOp::Store },
                })], ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera, &[i * self.stride]);
            let weight = 1.0 / f64::from(i + 1);
            pass.set_blend_constant(wgpu::Color { r: weight, g: weight, b: weight, a: 1.0 });
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(WIDTH * 8), rows_per_image: None } },
            self.target.size());
        self.queue.submit(Some(encoder.finish()));
        let slice = self.readback.slice(..);
        let (send, recv) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { let _ = send.send(result); });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        recv.recv().unwrap().unwrap();
        let map = slice.get_mapped_range();
        let rgb = map.chunks_exact(8).map(|pixel| std::array::from_fn(|channel| {
            half::f16::from_le_bytes([pixel[channel * 2], pixel[channel * 2 + 1]]).to_f32()
        })).collect();
        drop(map);
        self.readback.unmap();
        rgb
    }
}

fn errors(rgb: &[[f32; 3]], reference: &[[f32; 3]]) -> [f64; 2] {
    let sum = rgb.iter().zip(reference).flat_map(|(a, b)| a.iter().zip(b)).map(|(a,b)| f64::from((a-b).powi(2))).sum::<f64>();
    let mut coarse = 0.0;
    for y in (0..HEIGHT as usize).step_by(4) {
        for x in (0..WIDTH as usize).step_by(4) {
            let mut delta = [0.0f64; 3];
            for row in y..y+4 {
                for column in x..x+4 {
                    let i = row * WIDTH as usize + column;
                    for channel in 0..3 { delta[channel] += f64::from(rgb[i][channel] - reference[i][channel]) / 16.0; }
                }
            }
            coarse += delta.iter().map(|x| x*x).sum::<f64>();
        }
    }
    let n = f64::from(WIDTH * HEIGHT * 3);
    [(sum/n).sqrt(), (16.0*coarse/n).sqrt()]
}

fn display(rgb: &[[f32; 3]]) -> image::RgbImage {
    image::RgbImage::from_fn(WIDTH, HEIGHT, |x,y| {
        image::Rgb(rgb[(y * WIDTH + x) as usize].map(|v| {
            let v = v.clamp(0.0, 1.0);
            let srgb = if v < 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0/2.4) - 0.055 };
            (srgb * 255.0).round() as u8
        }))
    })
}

#[test]
#[ignore = "GPU comparison, writes local image panels"]
fn compare_local_sampling() {
    let probe = Probe::new();
    let output = Path::new("target/sampling");
    std::fs::create_dir_all(output).unwrap();
    println!("scene,mode,spp,RMSE,RMSE_filtered_4x4");
    for (label, fractal, mutation) in [("menger", Fractal::Menger, None),
        ("mandelbulb", Fractal::Mandelbulb, None), ("menger-box", Fractal::Menger, Some(Fractal::Mandelbox))]
    {
        let mut app = App::default();
        app.fractal = fractal;
        app.camera = Camera::framed_view(fractal);
        app.mutation = MutationStage { formula: mutation, weight: 0.5, ..Default::default() };
        app.lighting.path_traced = true;
        app.lighting.shadows = true;
        app.lighting.angular_radius = 24.0;
        let reference = probe.run(&app, 3, REFERENCE, false, 0);
        let sample_counts = [1, 4, 16, 64];
        let mut panel = image::RgbImage::new(WIDTH * 4, HEIGHT * sample_counts.len() as u32);
        for (row, samples) in sample_counts.into_iter().enumerate() {
            for (column, name) in ["current", "sobol", "stbn"].into_iter().enumerate() {
                let image = probe.run(&app, column as u32, samples, false, 0);
                let [full, filtered] = errors(&image, &reference);
                assert!(full.is_finite() && full > 0.0);
                println!("{label},{name},{samples},{full:.6},{filtered:.6}");
                image::imageops::replace(&mut panel, &display(&image), (column as u32 * WIDTH).into(), (row as u32 * HEIGHT).into());
            }
            image::imageops::replace(&mut panel, &display(&reference), (3*WIDTH).into(), (row as u32 * HEIGHT).into());
        }
        panel.save(output.join(format!("{label}.png"))).unwrap();
        if mutation.is_some() {
            let mut navigation = image::RgbImage::new(WIDTH * 4, HEIGHT * 4);
            let initial_yaw = app.camera.yaw;
            let mut previous: [Option<Vec<[f32; 3]>>; 3] = Default::default();
            let mut temporal_error = [0.0; 3];
            for frame in 0..4 {
                app.camera.yaw = initial_yaw + (frame as f32 - 1.5) * 0.025;
                let reference = probe.run(&app, 3, REFERENCE, true, frame);
                for (strategy, name) in ["current", "sobol", "stbn"].into_iter().enumerate() {
                    let image = probe.run(&app, strategy as u32, 1, true, frame);
                    let [full, filtered] = errors(&image, &reference);
                    println!("moving-{frame},{name},1,{full:.6},{filtered:.6}");
                    let residual: Vec<[f32; 3]> = image.iter().zip(&reference)
                        .map(|(a, b)| std::array::from_fn(|i| a[i] - b[i])).collect();
                    if let Some(before) = &previous[strategy] {
                        temporal_error[strategy] += errors(&residual, before)[1] / 3.0;
                    }
                    previous[strategy] = Some(residual);
                    image::imageops::replace(&mut navigation, &display(&image), (strategy as u32 * WIDTH).into(), (frame * HEIGHT).into());
                }
                image::imageops::replace(&mut navigation, &display(&reference), (3 * WIDTH).into(), (frame * HEIGHT).into());
            }
            println!("Mean change of 4×4-filtered residual across poses: {temporal_error:.6?}");
            navigation.save(output.join("menger-box-moving.png")).unwrap();
        }
    }
}

#[test]
#[ignore = "Regenerates the local STBN asset"]
fn generate_stbn_asset() {
    let mut generator = Mask::new();
    let x = generator.ranks(457103);
    let y = generator.ranks(991873);
    let bytes: Vec<_> = x.into_iter().zip(y).flat_map(|(x, y)| (x | y << 16).to_le_bytes()).collect();
    std::fs::create_dir_all("target/sampling").unwrap();
    std::fs::write("target/sampling/stbn-16x16x32.bin", bytes).unwrap();
}
