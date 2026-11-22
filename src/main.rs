mod about;
mod adaptive_viewport;
mod denoise;
mod viewport_denoise;
mod export;
mod camera_effects;
mod history;
mod hdri;
mod preset_gallery;
mod startup;
mod pixel_sampling;
mod intelligent_sampling;
mod noise_sampling;
#[cfg(test)]
mod sampling_benchmark;
use camera_effects::CameraEffects;
use std::{collections::HashSet, sync::Arc, time::{Duration, Instant}};
use winit::{application::ApplicationHandler, event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop, ControlFlow}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId}};

const APP_NAME: &str = "MengerVibes";
const MAX_MUTATIONS: usize = 8;
const EXTRA_MUTATION_UNIFORMS: usize = 20;
const MUTATION_PRESET_FIELDS: usize = 28;
const BASE_PRESET_FIELDS: usize = 103;
const LIGHTING_UNIFORM_OFFSET: usize = 100 + MAX_MUTATIONS * EXTRA_MUTATION_UNIFORMS;
const PATH_UNIFORM_OFFSET: usize = LIGHTING_UNIFORM_OFFSET + 4;
const SURFACE_UNIFORM_OFFSET: usize = PATH_UNIFORM_OFFSET + 8;
const SIERPINSKI_UNIFORM_OFFSET: usize = SURFACE_UNIFORM_OFFSET + 8;
const ORBIT_DEPTH_UNIFORM_OFFSET: usize = SIERPINSKI_UNIFORM_OFFSET + 4;
const FOUR_BRANCH_UNIFORM_OFFSET: usize = ORBIT_DEPTH_UNIFORM_OFFSET + 4;
const FULL_PRECISION_PREVIEW_UNIFORM: usize = FOUR_BRANCH_UNIFORM_OFFSET + 5;
const PIXEL_SAMPLING_UNIFORM: usize = FOUR_BRANCH_UNIFORM_OFFSET + 6;
const NAVIGATION_SAMPLE_UNIFORM: usize = FOUR_BRANCH_UNIFORM_OFFSET + 7;
const NOISE_SAMPLING_UNIFORM: usize = FOUR_BRANCH_UNIFORM_OFFSET + 8;
const BOKEH_UNIFORM_OFFSET: usize = NOISE_SAMPLING_UNIFORM + 1;
const CAMERA_UNIFORM_FLOATS: usize = NOISE_SAMPLING_UNIFORM + 4;
const CAMERA_UNIFORM_BYTES: u64 = (CAMERA_UNIFORM_FLOATS * std::mem::size_of::<f32>()) as u64;
const CONTRAST_RANGE: std::ops::RangeInclusive<f32> = 0.5..=3.0;
const AO_STRENGTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=4.0;
const CONTROLS_MIN_WIDTH: f32 = 315.0;
const CASCADE_PRESET_FIELDS: usize = BASE_PRESET_FIELDS + 1 + (MAX_MUTATIONS - 1) * MUTATION_PRESET_FIELDS;
const PRESET_LIGHT_FLAGS: usize = CASCADE_PRESET_FIELDS + 2;
const PRESET_ENVIRONMENT_FIELDS: usize = PRESET_LIGHT_FLAGS + 2;
const PRESET_PATH_FIELDS: usize = PRESET_ENVIRONMENT_FIELDS + 4;
const PRESET_MOVING_SAMPLES: usize = PRESET_PATH_FIELDS + 6;
const PRESET_MOVING_BOUNCES: usize = PRESET_MOVING_SAMPLES + 1;
const PRESET_SURFACE_FIELDS: usize = PRESET_MOVING_BOUNCES + 1;
const PRESET_PRINCIPLED_FIELDS: usize = PRESET_SURFACE_FIELDS + 5;
const PRESET_MATCAP_FIELDS: usize = PRESET_PRINCIPLED_FIELDS + 16;
const PRESET_SIERPINSKI_FIELDS: usize = PRESET_MATCAP_FIELDS + 12;
const PRESET_ORBIT_DEPTH_FIELDS: usize = PRESET_SIERPINSKI_FIELDS + 1 + MAX_MUTATIONS;
const PRESET_FOUR_BRANCH_FIELDS: usize = PRESET_ORBIT_DEPTH_FIELDS + 4 * (1 + MAX_MUTATIONS);
const PRESET_BOKEH_FIELDS: usize = PRESET_FOUR_BRANCH_FIELDS + 4 * (1 + MAX_MUTATIONS);
const PRESET_VALUE_COUNT: usize = PRESET_BOKEH_FIELDS + 2;
const MAX_FORMULA_DEPTH: u32 = 14;
const LIGHT_ANGULAR_RADIUS: std::ops::RangeInclusive<f32> = 0.0..=45.0;
const DEFAULT_VIEWPORT_SAMPLES: u32 = 16;
const STARTING_FOCAL_MM: f32 = 60.0;
const MOVING_SAMPLE_COUNTS: [u32; 3] = [1, 2, 4];
const MOVING_SAMPLES_UNIFORM: usize = PATH_UNIFORM_OFFSET + 6;
const MOVING_BOUNCES_UNIFORM: usize = PATH_UNIFORM_OFFSET + 7;
const DEFAULT_BACKGROUND: [f32; 3] = [0.015, 0.02, 0.04];
const DETAIL_QUALITY: std::ops::RangeInclusive<f32> = 0.5..=8.0;
const SMOOTHNESS: std::ops::RangeInclusive<f32> = 0.0..=0.2;
const SMOOTHNESS_KEY_FLOOR: f32 = 0.000020;
const FRACTAL_SHADER: &str = concat!(include_str!("contrast.wgsl"), "\n", include_str!("fractal.wgsl"), "\n",
    include_str!("path_tracer.wgsl"), "\n", include_str!("pixel_sampling.wgsl"), "\n", include_str!("intelligent_sampling.wgsl"), "\n",
    include_str!("noise_sampling.wgsl"));
const COMPOSITE_SHADER: &str = concat!(include_str!("contrast.wgsl"), "\n", include_str!("composite.wgsl"));
const REFINE_TILE_WIDTH: u32 = 256;
const REFINE_TILE_HEIGHT: u32 = 128;
const MAX_REFINE_TILES_PER_FRAME: u32 = 4;
const STANDARD_MATERIAL: u32 = 4;
const DIFFUSE_GRAY_MATERIAL: u32 = 10;
const MATCAP_MATERIAL: u32 = 11;
const MATERIALS: [(u32, &str); 3] = [
    (DIFFUSE_GRAY_MATERIAL, "Diffuse"),
    (STANDARD_MATERIAL, "Standard"),
    (MATCAP_MATERIAL, "Matcap"),
];
const MATCAP_STYLES: [(u32, &str); 9] = [
    (0, "Original"), (1, "Pearl"), (2, "Copper"), (3, "Nebula"), (5, "Gold"),
    (6, "Steel"), (7, "Clay"), (8, "Jade"), (9, "Aurora"),
];
const DIFFUSE_DEFAULTS: [f32; 5] = [0.50, 0.50, 0.50, 1.0, 0.0];
// Older presets that predate editable materials still need their original base parameters.
const LEGACY_MATERIAL_DEFAULTS: [[f32; 5]; 11] = [
    // Base RGB, roughness, metallic.
    [0.18, 0.48, 0.62, 0.5, 0.0],
    [0.55, 0.61, 0.69, 0.3, 0.0],
    [0.72, 0.33, 0.17, 0.38, 0.82],
    [0.26, 0.14, 0.48, 0.32, 0.25],
    [0.35, 0.62, 0.72, 0.5, 0.0],
    [0.78, 0.51, 0.16, 0.32, 0.85],
    [0.30, 0.39, 0.49, 0.28, 0.82],
    [0.54, 0.27, 0.18, 0.78, 0.0],
    [0.06, 0.37, 0.26, 0.35, 0.0],
    [0.34, 0.16, 0.46, 0.36, 0.35],
    DIFFUSE_DEFAULTS,
];

struct MatcapMaterial {
    style: u32,
    rotation: f32,
    roughness: f32,
    metallic: f32,
    finish: SurfaceFinish,
}

impl Default for MatcapMaterial {
    fn default() -> Self {
        Self { style: 0, rotation: 0.0, roughness: 0.5, metallic: 0.0, finish: SurfaceFinish::default() }
    }
}

impl MatcapMaterial {
    fn style_name(&self) -> &'static str {
        MATCAP_STYLES.iter().find(|(id, _)| *id == self.style).unwrap_or(&MATCAP_STYLES[0]).1
    }

    fn values(&self) -> [f32; 12] {
        let mut values = [0.0; 12];
        values[..4].copy_from_slice(&[self.style as f32, self.rotation, self.roughness, self.metallic]);
        values[4..].copy_from_slice(&self.finish.values());
        values
    }

    fn from_values(values: &[f32]) -> Self {
        let [style, rotation, roughness, metallic, finish @ ..] = values else { return Self::default(); };
        let style = MATCAP_STYLES.iter().find(|(id, _)| *id as f32 == *style).map_or(0, |(id, _)| *id);
        Self {
            style,
            rotation: rotation.clamp(-180.0, 180.0),
            roughness: roughness.clamp(0.05, 1.0),
            metallic: metallic.clamp(0.0, 1.0),
            finish: SurfaceFinish::from_values(finish),
        }
    }
}

struct SurfaceFinish {
    ior: f32,
    specular: f32,
    coat: f32,
    coat_roughness: f32,
    emission_color: [f32; 3],
    emission_strength: f32,
}

impl Default for SurfaceFinish {
    fn default() -> Self {
        Self { ior: 1.5, specular: 0.5, coat: 0.0, coat_roughness: 0.2,
            emission_color: [1.0; 3], emission_strength: 0.0 }
    }
}

impl SurfaceFinish {
    fn values(&self) -> [f32; 8] {
        let [r, g, b] = self.emission_color;
        [self.ior, self.specular, self.coat, self.coat_roughness, r, g, b, self.emission_strength]
    }

    fn from_values(values: &[f32]) -> Self {
        match values {
            [ior, specular, coat, roughness, r, g, b, strength] => Self {
                ior: ior.clamp(1.0, 3.0), specular: specular.clamp(0.0, 1.0),
                coat: coat.clamp(0.0, 1.0), coat_roughness: roughness.clamp(0.05, 1.0),
                emission_color: [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
                emission_strength: strength.clamp(0.0, 20.0),
            },
            _ => Self::default(),
        }
    }
}
const HDRI_ENVIRONMENT: usize = 10;
const ENVIRONMENT_NAMES: [&str; HDRI_ENVIRONMENT + 1] = ["Uniform", "Studio", "Pearl", "Copper", "Nebula", "Gold", "Steel", "Clay", "Jade", "Aurora", "HDRI"];

struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind: wgpu::BindGroup,
    focus_pipeline: wgpu::RenderPipeline,
    focus_bind: wgpu::BindGroup,
    progress_pipeline: wgpu::RenderPipeline,
    camera_layout: wgpu::BindGroupLayout,
    blue_noise: wgpu::TextureView,
    fallback_environment: hdri::Environment,
    environment_view: Option<hdri::Environment>,
    composite_pipeline: wgpu::RenderPipeline,
    accumulation: wgpu::Texture,
    composite_bind: wgpu::BindGroup,
    preview: wgpu::Texture,
    preview_bind: wgpu::BindGroup,
    preview_dimensions: Option<[f32; 2]>,
    viewport_in_flight: Option<adaptive_viewport::InFlight>,
    poll_redraw: bool,
    viewport_timing: Option<adaptive_viewport::GpuTiming>,
    noise_pipelines: Arc<std::sync::OnceLock<noise_sampling::Pipelines>>,
    noise_compiling: [bool; 2],
    noise_buffers: Option<noise_sampling::Buffers>,
    noise_plan: Option<noise_sampling::Plan>,
    noise_pending: Option<noise_sampling::Readback>,
    noise_remaining: Option<u32>,
    noise_tiles: Option<noise_sampling::ActiveTiles>,
    viewport_denoise: viewport_denoise::ViewportDenoise,
    intelligent_sampling: intelligent_sampling::IntelligentSampling,
    frame_submitted: bool,
    egui_renderer: egui_wgpu::Renderer,
    export_pipelines: Arc<std::sync::OnceLock<export::Pipelines>>,
}

impl Renderer {
    async fn new(
        instance: wgpu::Instance,
        surface: wgpu::Surface<'static>,
        size: winit::dpi::PhysicalSize<u32>,
        stage: impl Fn(&'static str),
    ) -> Result<Self, String> {
        stage("Selecting a graphics adapter");
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.ok_or("No compatible GPU found")?;
        println!("GPU: {} ({:?})", adapter.get_info().name, adapter.get_info().backend);
        stage("Preparing the graphics device");
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor { required_features, ..Default::default() }, None).await
            .map_err(|error| format!("GPU device: {error}"))?;
        let mut config = surface.get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("The GPU does not support the window surface")?;
        config.desired_maximum_frame_latency = 1;
        if surface.get_capabilities(&adapter).usages.contains(wgpu::TextureUsages::COPY_SRC) {
            config.usage |= wgpu::TextureUsages::COPY_SRC;
        }
        surface.configure(&device, &config);
        stage("Compiling the viewport shader");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mandelbulb"),
            source: wgpu::ShaderSource::Wgsl(FRACTAL_SHADER.into()),
        });
        let camera_layout = hdri::layout(&device, false);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Viewport camera and HDRI"), bind_group_layouts: &[&camera_layout], push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fractal pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(config.format.into())] }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera"), size: CAMERA_UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let fallback_environment = hdri::upload(&device, &queue, 1, 1, &[0.0; 3]);
        let blue_noise = pixel_sampling::blue_noise_view(&device, &queue);
        let camera_bind = hdri::bind(&device, &camera_layout, &camera_buffer, &fallback_environment, &blue_noise);
        stage("Compiling focus and progressive shaders");
        let (focus_pipeline, focus_bind) = Self::create_focus_pipeline(&device, &shader, &camera_buffer, &fallback_environment, &blue_noise);
        let progress_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Progressive fractal"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_progress"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Constant, dst_factor: wgpu::BlendFactor::OneMinusConstant, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent::REPLACE,
                }), write_mask: wgpu::ColorWrites::ALL,
            })] }),
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
        });
        stage("Preparing display and controls");
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Progressive composite"), source: wgpu::ShaderSource::Wgsl(COMPOSITE_SHADER.into()),
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Progressive display"), layout: None,
            vertex: wgpu::VertexState { module: &composite_shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &composite_shader, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(config.format.into())] }),
            primitive: wgpu::PrimitiveState::default(), depth_stencil: None, multisample: wgpu::MultisampleState::default(), multiview: None, cache: None,
        });
        let (accumulation, composite_bind) = Self::create_accumulation(&device, &composite_pipeline, &camera_buffer, config.width, config.height);
        let (preview, preview_bind) = Self::create_accumulation(&device, &composite_pipeline, &camera_buffer, config.width, config.height);
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);
        let viewport_timing = required_features.contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| adaptive_viewport::GpuTiming::new(&device, &queue));
        Ok(Self { surface, device, queue, config, pipeline, camera_buffer, camera_bind, focus_pipeline, focus_bind,
            progress_pipeline, camera_layout, blue_noise, fallback_environment, environment_view: None, composite_pipeline, accumulation,
            composite_bind, preview, preview_bind, preview_dimensions: None, viewport_denoise: Default::default(),
            intelligent_sampling: Default::default(), frame_submitted: false, egui_renderer,
            viewport_in_flight: None, poll_redraw: false, viewport_timing, export_pipelines: Arc::default(),
            noise_pipelines: Arc::default(), noise_compiling: [false; 2], noise_buffers: None,
            noise_plan: None, noise_pending: None, noise_remaining: None, noise_tiles: None })
    }

    fn set_environment(&mut self, view: Option<&hdri::Environment>) {
        if self.environment_view.as_ref() == view { return; }
        self.environment_view = view.cloned();
        self.camera_bind = hdri::bind(&self.device, &self.camera_layout, &self.camera_buffer,
            view.unwrap_or(&self.fallback_environment), &self.blue_noise);
    }

    fn create_focus_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, camera_buffer: &wgpu::Buffer, environment: &hdri::Environment, blue_noise: &wgpu::TextureView) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
        let camera_layout = hdri::layout(device, false);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Focus camera"), bind_group_layouts: &[&camera_layout], push_constant_ranges: &[],
        });
        let focus_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pick focus distance"), layout: Some(&layout),
            vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: shader, entry_point: Some("fs_focus"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::R32Float.into())] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let focus_bind = hdri::bind(device, &camera_layout, camera_buffer, environment, blue_noise);
        (focus_pipeline, focus_bind)
    }

    fn pick_focus(
        device: &wgpu::Device, queue: &wgpu::Queue, camera_buffer: &wgpu::Buffer,
        pipeline: &wgpu::RenderPipeline, bind: &wgpu::BindGroup,
        mut uniforms: [f32; CAMERA_UNIFORM_FLOATS], point: [f32; 2],
    ) -> Option<f32> {
        uniforms[36..38].copy_from_slice(&point);
        let bytes: Vec<_> = uniforms.iter().flat_map(|value| value.to_ne_bytes()).collect();
        queue.write_buffer(camera_buffer, 0, &bytes);
        let size = wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Focus depth"), size, mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Focus depth readback"), size: 4, mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        let view = texture.create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Focus depth"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.draw(0..3, 0..1);
        }
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: Default::default() },
            size,
        );
        let submission = queue.submit(Some(encoder.finish()));
        let slice = buffer.slice(..);
        let (send, receive) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { let _ = send.send(result); });
        let _ = device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission));
        receive.try_recv().ok()?.ok()?;
        let mapped = slice.get_mapped_range();
        let depth = f32::from_ne_bytes(mapped[..4].try_into().ok()?);
        (depth.is_finite() && depth > 0.0).then(|| depth.clamp(0.0001, 30.0))
    }

    fn create_accumulation(device: &wgpu::Device, pipeline: &wgpu::RenderPipeline, camera: &wgpu::Buffer, width: u32, height: u32) -> (wgpu::Texture, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Accumulated samples"), size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
        });
        let bind = Self::bind_composite(device, pipeline, camera, &texture);
        (texture, bind)
    }

    fn bind_composite(device: &wgpu::Device, pipeline: &wgpu::RenderPipeline, camera: &wgpu::Buffer, texture: &wgpu::Texture) -> wgpu::BindGroup {
        let view = texture.create_view(&Default::default());
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Accumulated samples"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: camera.as_entire_binding() },
            ],
        })
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        (self.accumulation, self.composite_bind) = Self::create_accumulation(&self.device, &self.composite_pipeline, &self.camera_buffer, width, height);
        (self.preview, self.preview_bind) = Self::create_accumulation(&self.device, &self.composite_pipeline, &self.camera_buffer, width, height);
        self.preview_dimensions = None;
        self.viewport_denoise.resize();
        self.intelligent_sampling.invalidate_view();
        self.noise_buffers = None;
        self.noise_pending = None;
        self.noise_remaining = None;
        self.noise_tiles = None;
    }

    fn poll_viewport_denoise(&mut self) {
        self.viewport_denoise.poll(&self.device, &self.queue, &self.composite_pipeline, &self.camera_buffer, self.accumulation.size());
    }

    fn prepare_intelligent_sampling(&mut self, denoise: bool, wake: impl FnOnce() + Send + 'static) -> bool {
        self.intelligent_sampling.request(&self.device, &self.camera_layout, &self.camera_buffer, &self.accumulation, denoise, wake)
    }

    fn prepare_noise_sampling(&mut self, denoise: bool, window: Arc<Window>) -> bool {
        if self.noise_pipelines.get().is_some_and(|pipelines| pipelines.ready(denoise)) { return true; }
        let requested = &mut self.noise_compiling[usize::from(denoise)];
        if !*requested {
            *requested = true;
            let pipelines = self.noise_pipelines.clone();
            let device = self.device.clone();
            let camera = self.camera_layout.clone();
            std::thread::spawn(move || {
                pipelines.get_or_init(|| noise_sampling::Pipelines::new(&device, false)).sample_pipeline(&device, &camera, denoise);
                window.request_redraw();
            });
        }
        false
    }

    fn poll_noise_sampling(&mut self) {
        if let Some(pending) = &self.noise_pending {
            let _ = self.device.poll(wgpu::Maintain::Poll);
            if let Some(result) = pending.poll() {
                self.noise_pending = None;
                if let Some((remaining, tiles)) = result {
                    self.noise_remaining = Some(remaining);
                    self.noise_tiles = Some(tiles);
                }
            }
        }
    }

    fn poll_viewport_work(&mut self) -> Option<Option<adaptive_viewport::PassStats>> {
        let pending = self.viewport_in_flight.as_ref()?;
        let _ = self.device.poll(wgpu::Maintain::Poll);
        let result = pending.poll(self.viewport_timing.as_ref())?;
        self.viewport_in_flight = None;
        Some(result)
    }

    fn render(&mut self, mut data: [f32; CAMERA_UNIFORM_FLOATS], progress: Option<u32>, advance: bool, refine_tiles: Option<std::ops::Range<u32>>, render_size: [u32; 2], gui: egui::FullOutput, ctx: &egui::Context, capture: bool, adaptive: bool, preview_work: Option<adaptive_viewport::PreviewWork>) -> Option<Result<image::RgbaImage, String>> {
        self.frame_submitted = false;
        let full_size = [self.config.width, self.config.height];
        let preview_size = (render_size != full_size || preview_work.is_some()).then(|| render_size.map(|side| side as f32));
        if let Some([width, height]) = preview_size {
            if preview_work.is_some() {
                data[FULL_PRECISION_PREVIEW_UNIFORM] = self.config.height as f32;
            } else {
                data[10] *= height / self.config.height as f32;
            }
            data[53] = width;
        }
        let [columns, rows] = noise_sampling::viewport_tile_grid(render_size);
        let active_tiles = self.noise_tiles.as_ref().filter(|_| self.noise_plan.is_some());
        let tile_count = active_tiles.map_or(columns * rows, noise_sampling::ActiveTiles::len);
        let sample_incomplete = refine_tiles.as_ref().is_some_and(|tiles| tiles.end < tile_count);
        let needs_cached_preview = progress == Some(0) && sample_incomplete;
        let cached_preview = needs_cached_preview.then_some(self.preview_dimensions).flatten();
        // Composite dimensions are separate from the tracing controls. This
        // keeps the cached preview available while the first sample renders.
        let display_size = cached_preview.or(preview_size).unwrap_or([0.0; 2]);
        data[54] = preview_size.map_or(0.0, |size| size[1]);
        data[55] = display_size[0];
        data[83] = display_size[1];
        data[NOISE_SAMPLING_UNIFORM] = full_size[0] as f32;
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.queue.write_buffer(&self.camera_buffer, 0, &bytes);
        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return capture.then(|| Err("Surface unavailable; try saving again".into()));
            }
            Err(error) => { eprintln!("Surface: {error}"); return capture.then(|| Err(format!("Surface: {error}"))); }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let timing = self.viewport_timing.as_ref().filter(|_| preview_work.is_some());
        let mut convergence = None;
        if advance || (progress.is_none() && preview_size.is_some()) {
            let index = progress.unwrap_or(0);
            let first_tile = refine_tiles.as_ref().is_none_or(|tiles| tiles.start == 0);
            let target = if progress.is_some() { &self.accumulation } else { &self.preview };
            let accumulation_view = target.create_view(&Default::default());
            let denoise_guides = advance && self.viewport_denoise.prepare(&self.device, &self.camera_layout, target.size());
            let noise = self.noise_plan.filter(|_| advance).zip(self.noise_pipelines.get()).map(|(plan, pipelines)| {
                let buffers = self.noise_buffers.get_or_insert_with(|| noise_sampling::Buffers::new(&self.device, pipelines, target));
                if index == 0 && first_tile { buffers.reset(&mut encoder); }
                (plan, pipelines, buffers)
            });
            let guided_sampling = advance && self.intelligent_sampling.enabled;
            let ray_guides = self.intelligent_sampling.resources().filter(|_| guided_sampling && index != 1);
            let guide_views = self.viewport_denoise.guides.as_ref().filter(|_| denoise_guides)
                .map(|guides| guides.each_ref().map(|texture| texture.create_view(&Default::default())));
            let load = if index == 0 && first_tile { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load };
            let attachment = |view| Some(wgpu::RenderPassColorAttachment {
                view, resolve_target: None, ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            });
            let mut attachments = vec![attachment(&accumulation_view)];
            if let Some((_, _, buffers)) = &noise { attachments.push(attachment(&buffers.half_view)); }
            if let Some(guides) = ray_guides.filter(|_| index == 0) { attachments.push(attachment(guides.geometry())); }
            if let Some(views) = &guide_views { attachments.extend(views.iter().map(attachment)); }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render viewport sample"), color_attachments: &attachments,
                depth_stencil_attachment: None, timestamp_writes: timing.map(adaptive_viewport::GpuTiming::writes), occlusion_query_set: None,
            });
            if let Some((_, pipelines, buffers)) = &noise {
                pass.set_pipeline(pipelines.sample_pipeline(&self.device, &self.camera_layout, denoise_guides));
                pass.set_bind_group(1, &buffers.mask_bind, &[]);
            } else if let Some(guides) = ray_guides {
                guides.bind(&mut pass, index == 0, denoise_guides);
            } else {
                let pipeline = self.viewport_denoise.pipeline.as_ref().filter(|_| denoise_guides).unwrap_or(&self.progress_pipeline);
                pass.set_pipeline(pipeline);
            }
            pass.set_bind_group(0, &self.camera_bind, &[]);
            let weight = 1.0 / (index + 1) as f64;
            pass.set_blend_constant(wgpu::Color { r: weight, g: weight, b: weight, a: 1.0 });
            // The still preview starts with a fresh sample after interaction.
            if let Some([width, height]) = preview_size {
                pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
            }
            if let Some(tiles) = refine_tiles.or_else(|| active_tiles.filter(|_| advance).map(|tiles| 0..tiles.len())) {
                for index in tiles {
                    let tile = active_tiles.map_or(index, |tiles| tiles.tile(index));
                    let x = (tile % columns) * REFINE_TILE_WIDTH;
                    let y = (tile / columns) * REFINE_TILE_HEIGHT;
                    pass.set_scissor_rect(x, y, REFINE_TILE_WIDTH.min(render_size[0] - x), REFINE_TILE_HEIGHT.min(render_size[1] - y));
                    pass.draw(0..3, 0..1);
                }
            } else {
                pass.draw(0..3, 0..1);
            }
            drop(pass);
            if let Some((plan, pipelines, buffers)) = noise {
                if !sample_incomplete && index + 1 == plan.next_check(index) {
                    convergence = Some(buffers.check(&self.device, &self.queue, &mut encoder, pipelines, render_size, plan, index + 1));
                }
            }
            if guided_sampling && !sample_incomplete {
                if let Some(guides) = ray_guides.filter(|_| index == 0) {
                    guides.save_first_sample(&mut encoder, target, render_size);
                } else if index == 1 {
                    if let Some(guides) = self.intelligent_sampling.resources() {
                        guides.allocate(&mut encoder, render_size);
                    }
                }
            }
        }
        if let Some(timing) = timing { timing.resolve(&mut encoder, &self.device, &self.preview, &self.camera_buffer); }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fractal"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            if progress.is_some() || preview_size.is_some() {
                pass.set_pipeline(&self.composite_pipeline);
                let bind = if cached_preview.is_some() || progress.is_none() {
                    &self.preview_bind
                } else {
                    self.viewport_denoise.display.as_ref().unwrap_or(&self.composite_bind)
                };
                pass.set_bind_group(0, bind, &[]);
            } else {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.camera_bind, &[]);
            }
            pass.draw(0..3, 0..1);
        }
        let row_size = self.config.width * 4;
        let padded_row = row_size.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let capture_buffer = if capture && self.config.usage.contains(wgpu::TextureUsages::COPY_SRC) {
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Preset thumbnail"), size: u64::from(padded_row) * u64::from(self.config.height),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false,
            });
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &frame.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded_row), rows_per_image: Some(self.config.height) } },
                wgpu::Extent3d { width: self.config.width, height: self.config.height, depth_or_array_layers: 1 },
            );
            Some(buffer)
        } else { None };
        let jobs = ctx.tessellate(gui.shapes, gui.pixels_per_point);
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.config.width, self.config.height], pixels_per_point: gui.pixels_per_point };
        for (id, delta) in &gui.textures_delta.set { self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta); }
        let buffers = self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor { label: Some("Interface"), color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None });
            self.egui_renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
        }
        self.queue.submit(buffers.into_iter().chain(Some(encoder.finish())));
        if let Some(buffer) = convergence { self.noise_pending = Some(noise_sampling::Readback::new(buffer)); }
        if adaptive {
            self.viewport_in_flight = Some(adaptive_viewport::InFlight::submit(&self.queue, timing, preview_work));
        }
        self.frame_submitted = true;
        if progress.is_none() {
            self.preview_dimensions = preview_size;
        }
        for id in &gui.textures_delta.free { self.egui_renderer.free_texture(id); }
        frame.present();
        if let Some(samples) = progress.filter(|_| cached_preview.is_none() && self.viewport_denoise.display.is_some()) {
            let completed_sample = advance && !sample_incomplete;
            self.viewport_denoise.record_presentation(samples + u32::from(completed_sample));
        }
        if !capture { return None; }
        let Some(buffer) = capture_buffer else { return Some(Err("Screenshot is not supported by this graphics surface".into())); };
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });
        let _ = self.device.poll(wgpu::Maintain::Wait);
        if let Err(error) = receiver.recv().map_err(|e| e.to_string()).and_then(|r| r.map_err(|e| e.to_string())) {
            return Some(Err(error));
        }
        let mapped = slice.get_mapped_range();
        let mut rgba = Vec::with_capacity((row_size * self.config.height) as usize);
        for row in mapped.chunks(padded_row as usize).take(self.config.height as usize) {
            rgba.extend_from_slice(&row[..row_size as usize]);
        }
        drop(mapped);
        buffer.unmap();
        if matches!(self.config.format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb) {
            for pixel in rgba.chunks_exact_mut(4) { pixel.swap(0, 2); }
        } else if !matches!(self.config.format, wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb) {
            return Some(Err(format!("Unsupported screenshot format: {:?}", self.config.format)));
        }
        Some(image::RgbaImage::from_raw(self.config.width, self.config.height, rgba).ok_or_else(|| "Invalid screenshot dimensions".to_string()))
    }
}

struct Lighting {
    color: [f32; 3],
    roughness: f32,
    metallic: f32,
    light_color: [f32; 3],
    azimuth: f32,
    elevation: f32,
    intensity: f32,
    environment_strength: f32,
    environment_color: [f32; 3],
    angular_radius: f32,
    enabled: bool,
    shadows: bool,
    path_traced: bool,
    environment: usize,
    environment_rotation: f32,
    show_environment: bool,
    bounces: u32,
}
impl Default for Lighting {
    fn default() -> Self {
        Self {
            color: [0.35, 0.62, 0.72], roughness: 0.5, metallic: 0.0,
            light_color: [1.0; 3], azimuth: 0.6, elevation: 0.8, intensity: 1.0,
            environment_strength: 0.25, environment_color: [1.0; 3], angular_radius: 0.0,
            enabled: true, shadows: true,
            path_traced: false, environment: 0, environment_rotation: 0.0,
            show_environment: false, bounces: 4,
        }
    }
}

impl Lighting {
    fn environment_intensity_max(&self) -> f32 {
        if self.environment == HDRI_ENVIRONMENT { 10.0 } else { 2.0 }
    }

    fn path_preset(&self) -> [f32; 6] {
        // The last slot was the path-only diffuse material toggle.
        [flag(self.path_traced), self.environment as f32, self.environment_rotation,
            self.bounces as f32, flag(self.show_environment), 0.0]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Fractal {
    Mandelbulb = 0,
    Mandelbox = 1,
    #[default]
    Menger = 2,
    Kifs = 3,
    AmazingSurface = 4,
    PseudoKleinian = 5,
    Sierpinski = 6,
    FourBranch = 7,
}

impl Fractal {
    const ALL: [Self; 8] = [Self::Mandelbulb, Self::Mandelbox, Self::Menger, Self::Kifs, Self::AmazingSurface, Self::PseudoKleinian, Self::Sierpinski, Self::FourBranch];

    fn name(self) -> &'static str {
        match self {
            Self::Mandelbulb => "Mandelbulb",
            Self::Mandelbox => "Mandelbox",
            Self::Menger => "Menger",
            Self::Kifs => "KIFS",
            Self::AmazingSurface => "Amazing Surface",
            Self::PseudoKleinian => "Pseudo-Kleinian",
            Self::Sierpinski => "Sierpinski Tetrahedron",
            Self::FourBranch => "Four-Branch Knot",
        }
    }

    fn value(self) -> f32 { self as u8 as f32 }

    fn from_value(value: f32) -> Option<Self> {
        Self::ALL.into_iter().find(|formula| formula.value() == value)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct FourBranchParameters {
    depth: u32,
    thickness: f32,
    spread: f32,
    twist: f32,
}

impl Default for FourBranchParameters {
    fn default() -> Self {
        Self { depth: 5, thickness: 0.16, spread: 1.05, twist: 32.0 }
    }
}

impl FourBranchParameters {
    const MAX_DEPTH: u32 = MAX_FORMULA_DEPTH;
    const THICKNESS: std::ops::RangeInclusive<f32> = 0.055..=0.24;
    const SPREAD: std::ops::RangeInclusive<f32> = 0.72..=1.35;
    const TWIST: std::ops::RangeInclusive<f32> = -90.0..=90.0;

    fn ui(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Depth").selected_text(self.depth.to_string()).show_ui(ui, |ui| {
            for depth in 0..=Self::MAX_DEPTH {
                ui.selectable_value(&mut self.depth, depth, depth.to_string());
            }
        }).response.on_hover_text("Depth 0 shows the four original arms.");
        precision_slider(ui, &mut self.thickness, Self::THICKNESS, "Thickness", false, false);
        precision_slider(ui, &mut self.spread, Self::SPREAD, "Spread", false, false);
        precision_slider(ui, &mut self.twist, Self::TWIST, "Twist", false, true)
            .on_hover_text("Twist the smaller copies around their arms. The direction alternates at each level.");
        if ui.button("Reset Four-Branch Knot parameters").clicked() {
            *self = Self::default();
        }
    }

    fn values(self) -> [f32; 4] { [self.depth as f32, self.thickness, self.spread, self.twist] }

    fn uniforms(self) -> [f32; 5] {
        let (sine, cosine) = self.twist.to_radians().sin_cos();
        [self.depth as f32, self.thickness, self.spread, cosine, sine]
    }

    fn from_preset(values: Option<&[f32]>) -> Self {
        let Some([depth, thickness, spread, twist]) = values else { return Self::default(); };
        Self {
            depth: (*depth as u32).min(Self::MAX_DEPTH),
            thickness: thickness.clamp(*Self::THICKNESS.start(), *Self::THICKNESS.end()),
            spread: spread.clamp(*Self::SPREAD.start(), *Self::SPREAD.end()),
            twist: twist.clamp(*Self::TWIST.start(), *Self::TWIST.end()),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct SierpinskiParameters {
    depth: u32,
}

impl Default for SierpinskiParameters {
    fn default() -> Self { Self { depth: 5 } }
}

impl SierpinskiParameters {
    const MAX_DEPTH: u32 = MAX_FORMULA_DEPTH;

    fn ui(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Depth").selected_text(self.depth.to_string()).show_ui(ui, |ui| {
            for depth in 0..=Self::MAX_DEPTH {
                ui.selectable_value(&mut self.depth, depth, depth.to_string());
            }
        }).response.on_hover_text("Depth 0 shows the solid tetrahedron. Each level keeps four corner copies.");
        if ui.button("Reset Sierpinski parameters").clicked() {
            *self = Self::default();
        }
    }

    fn from_preset(depth: Option<&f32>) -> Self {
        Self { depth: depth.map_or(Self::default().depth, |depth| (*depth as u32).min(Self::MAX_DEPTH)) }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OrbitDepths([u32; 4]);

impl Default for OrbitDepths {
    fn default() -> Self { Self([Self::DEFAULT; 4]) }
}

impl OrbitDepths {
    const DEFAULT: u32 = 11;
    const RANGE: std::ops::RangeInclusive<u32> = 2..=MAX_FORMULA_DEPTH;

    fn index(formula: Fractal) -> Option<usize> {
        match formula {
            Fractal::Mandelbulb => Some(0),
            Fractal::Mandelbox => Some(1),
            Fractal::AmazingSurface => Some(2),
            Fractal::PseudoKleinian => Some(3),
            _ => None,
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, formula: Fractal) {
        if let Some(index) = Self::index(formula) {
            ui.add(egui::Slider::new(&mut self.0[index], Self::RANGE).text("Base depth"))
                .on_hover_text("Sets the base detail. Close surfaces can use extra iterations.");
        }
    }

    fn reset(&mut self, formula: Fractal) {
        if let Some(index) = Self::index(formula) {
            self.0[index] = Self::DEFAULT;
        }
    }

    fn values(self) -> [f32; 4] { self.0.map(|depth| depth as f32) }

    fn offsets(self) -> [f32; 4] { self.values().map(|depth| depth - Self::DEFAULT as f32) }

    fn from_preset(values: Option<&[f32]>) -> Self {
        let Some([bulb, mandelbox, surface, kleinian]) = values else { return Self::default(); };
        Self([*bulb, *mandelbox, *surface, *kleinian].map(|depth|
            (depth as u32).clamp(*Self::RANGE.start(), *Self::RANGE.end())))
    }
}

#[derive(Debug, PartialEq)]
struct FoldParameters {
    kifs: [f32; 4],
    amazing_surface: [f32; 4],
    pseudo_kleinian: [f32; 4],
}

impl Default for FoldParameters {
    fn default() -> Self {
        Self {
            kifs: [7.0, 2.45, 12.0, 1.0],
            amazing_surface: [-2.5, 1.0, 0.65, 0.0],
            pseudo_kleinian: [1.0, 0.88, 0.08, 0.16],
        }
    }
}

impl FoldParameters {
    const LEGACY_SURFACE_DEFAULTS: [f32; 4] = [-1.8, 1.0, 0.5, 0.0];
    const LEGACY_KLEINIAN_DEFAULTS: [f32; 4] = [0.80, 1.0, 0.12, 0.08];
    const KIFS_RANGES: [std::ops::RangeInclusive<f32>; 4] = [2.0..=MAX_FORMULA_DEPTH as f32, 2.0..=3.5, -180.0..=180.0, 0.3..=1.5];
    const SURFACE_RANGES: [std::ops::RangeInclusive<f32>; 4] = [-3.0..=-1.2, 0.25..=2.0, 0.2..=0.95, -0.2..=0.2];
    const KLEINIAN_RANGES: [std::ops::RangeInclusive<f32>; 4] = [0.2..=1.5, 0.35..=2.0, 0.02..=0.4, -0.5..=0.5];

    fn ui(&mut self, ui: &mut egui::Ui, formula: Fractal) -> bool {
        let default = Self::default();
        match formula {
            Fractal::Kifs => {
                ui.add(egui::Slider::new(&mut self.kifs[0], Self::KIFS_RANGES[0].clone()).step_by(1.0).text("Depth"));
                precision_slider(ui, &mut self.kifs[1], Self::KIFS_RANGES[1].clone(), "Scale", false, false);
                precision_slider(ui, &mut self.kifs[2], Self::KIFS_RANGES[2].clone(), "Fold rotation Y", false, true);
                precision_slider(ui, &mut self.kifs[3], Self::KIFS_RANGES[3].clone(), "Seed radius", false, false);
                let reset = ui.button("Reset KIFS parameters").clicked();
                if reset { self.kifs = default.kifs; }
                reset
            }
            Fractal::AmazingSurface => {
                for (i, label) in ["Scale", "Fold limit", "Inner radius", "Scale variation"].into_iter().enumerate() {
                    precision_slider(ui, &mut self.amazing_surface[i], Self::SURFACE_RANGES[i].clone(), label, false, false);
                }
                let reset = ui.button("Reset Amazing Surface parameters").clicked();
                if reset { self.amazing_surface = default.amazing_surface; }
                reset
            }
            Fractal::PseudoKleinian => {
                for (i, label) in ["Fold limit", "Inversion radius", "Wall thickness", "Offset Z"].into_iter().enumerate() {
                    precision_slider(ui, &mut self.pseudo_kleinian[i], Self::KLEINIAN_RANGES[i].clone(), label, false, false);
                }
                let reset = ui.button("Reset Pseudo-Kleinian parameters").clicked();
                if reset { self.pseudo_kleinian = default.pseudo_kleinian; }
                reset
            }
            _ => unreachable!("formula has a separate parameter panel"),
        }
    }

    fn values(&self) -> [f32; 12] {
        let [depth, scale, rotation, radius] = self.kifs;
        let [surface_scale, fold, inner, variation] = self.amazing_surface;
        let [limit, inversion, thickness, offset] = self.pseudo_kleinian;
        [depth, scale, rotation, radius, surface_scale, fold, inner, variation, limit, inversion, thickness, offset]
    }

    fn uniforms(&self) -> [f32; 12] {
        let mut values = self.values();
        values[2] = values[2].to_radians();
        values[6] *= values[6];
        values
    }

    fn from_preset(values: &[f32], kleinian: &[f32]) -> Self {
        let mut settings = Self {
            amazing_surface: Self::LEGACY_SURFACE_DEFAULTS,
            pseudo_kleinian: Self::LEGACY_KLEINIAN_DEFAULTS,
            ..Self::default()
        };
        if let [kifs @ .., scale, fold, inner, variation] = values {
            if kifs.len() == 4 {
                for (i, value) in kifs.iter().enumerate() {
                    let range = &Self::KIFS_RANGES[i];
                    settings.kifs[i] = value.clamp(*range.start(), *range.end());
                }
                settings.kifs[0] = settings.kifs[0].round();
                for (i, value) in [*scale, *fold, *inner, *variation].into_iter().enumerate() {
                    let range = &Self::SURFACE_RANGES[i];
                    settings.amazing_surface[i] = value.clamp(*range.start(), *range.end());
                }
            }
        }
        if kleinian.len() == 4 {
            for (i, value) in kleinian.iter().enumerate() {
                let range = &Self::KLEINIAN_RANGES[i];
                settings.pseudo_kleinian[i] = value.clamp(*range.start(), *range.end());
            }
        }
        settings
    }
}

#[derive(Debug, PartialEq)]
struct MengerParameters {
    depth: u32,
    tunnel_width: f32,
    rotation: [f32; 3],
}

impl Default for MengerParameters {
    fn default() -> Self {
        Self { depth: 5, tunnel_width: 1.0, rotation: [0.0; 3] }
    }
}

impl MengerParameters {
    const DEPTH: std::ops::RangeInclusive<u32> = 1..=MAX_FORMULA_DEPTH;
    const TUNNEL_WIDTH: std::ops::RangeInclusive<f32> = 0.6..=1.4;
    const ROTATION: std::ops::RangeInclusive<f32> = -360.0..=360.0;

    fn ui(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Depth").selected_text(self.depth.to_string()).show_ui(ui, |ui| {
            for depth in Self::DEPTH {
                ui.selectable_value(&mut self.depth, depth, depth.to_string());
            }
        });
        precision_slider(ui, &mut self.tunnel_width, Self::TUNNEL_WIDTH, "Tunnel width", false, false)
            .on_hover_text("1.0 gives the classic width of one third of a cell.");
        ui.label("Rotation per level")
            .on_hover_text("Rotates each finer tunnel grid, applying X, then Y, then Z.");
        for (angle, axis) in self.rotation.iter_mut().zip(["X", "Y", "Z"]) {
            precision_slider(ui, angle, Self::ROTATION, axis, false, true);
        }
        if ui.button("Reset Menger parameters").clicked() {
            *self = Self::default();
        }
    }

    fn values(&self) -> [f32; 5] {
        let [x, y, z] = self.rotation;
        // Keep Z in the original preset slot.
        [self.depth as f32, self.tunnel_width, z, x, y]
    }

    fn uniforms(&self) -> [f32; 16] {
        let [(sx, cx), (sy, cy), (sz, cz)] = self.rotation.map(|angle| {
            (angle % 360.0).to_radians().sin_cos()
        });
        // Sample in the inverse of the grid's rotation. The forward matrix's
        // rows become columns of its inverse, with uniform-buffer padding.
        [
            self.depth as f32, self.tunnel_width, 0.0, 0.0,
            cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, 0.0,
            sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, 0.0,
            -sy, cy * sx, cy * cx, 0.0,
        ]
    }

    fn from_preset(values: &[f32]) -> Self {
        let (depth, width, rotation) = match values {
            [depth, width, z] => (*depth, *width, [0.0, 0.0, *z]),
            [depth, width, z, x, y] => (*depth, *width, [*x, *y, *z]),
            _ => return Self::default(),
        };
        Self {
            depth: (depth as u32).clamp(*Self::DEPTH.start(), *Self::DEPTH.end()),
            tunnel_width: width.clamp(*Self::TUNNEL_WIDTH.start(), *Self::TUNNEL_WIDTH.end()),
            rotation: rotation.map(|angle| angle.clamp(*Self::ROTATION.start(), *Self::ROTATION.end())),
        }
    }
}

#[derive(Debug, PartialEq)]
struct MandelboxParameters {
    scale: f32,
    fold_limit: f32,
    inner_radius: f32,
    outer_radius: f32,
}

impl Default for MandelboxParameters {
    fn default() -> Self {
        Self { scale: -1.8, fold_limit: 1.0, inner_radius: 0.5, outer_radius: 1.0 }
    }
}

impl MandelboxParameters {
    const LEGACY_SCALE: f32 = -1.5;
    const SCALE: std::ops::RangeInclusive<f32> = -2.5..=-1.15;
    const FOLD_LIMIT: std::ops::RangeInclusive<f32> = 0.25..=2.0;
    const MIN_INNER_RADIUS: f32 = 0.2;
    const MIN_OUTER_RADIUS: f32 = 0.5;
    const MAX_OUTER_RADIUS: f32 = 1.5;

    fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        precision_slider(ui, &mut self.scale, Self::SCALE, "Scale", false, false)
            .on_hover_text("Changes the size ratio between successive levels of the fractal.");
        precision_slider(ui, &mut self.fold_limit, Self::FOLD_LIMIT, "Box fold limit", false, false)
            .on_hover_text("Sets how far from the center the box fold starts.");
        precision_slider(ui, &mut self.inner_radius, Self::MIN_INNER_RADIUS..=self.outer_radius, "Inner radius", false, false)
            .on_hover_text("Sets the inner boundary of the sphere fold.");
        precision_slider(ui, &mut self.outer_radius, self.inner_radius.max(Self::MIN_OUTER_RADIUS)..=Self::MAX_OUTER_RADIUS, "Outer radius", false, false)
            .on_hover_text("Sets the outer boundary of the sphere fold.");
        let reset = ui.button("Reset Mandelbox parameters").clicked();
        if reset {
            *self = Self::default();
        }
        reset
    }

    fn values(&self) -> [f32; 4] {
        [self.scale, self.fold_limit, self.inner_radius, self.outer_radius]
    }

    fn uniforms(&self) -> [f32; 4] {
        let outer_squared = self.outer_radius * self.outer_radius;
        let inner_squared = self.inner_radius * self.inner_radius;
        [self.scale, self.fold_limit, outer_squared, outer_squared / inner_squared]
    }

    fn from_preset(values: &[f32]) -> Self {
        let [scale, fold_limit, inner_radius, outer_radius] = values else {
            return Self { scale: Self::LEGACY_SCALE, ..Self::default() };
        };
        let outer_radius = outer_radius.clamp(Self::MIN_OUTER_RADIUS, Self::MAX_OUTER_RADIUS);
        Self {
            scale: scale.clamp(*Self::SCALE.start(), *Self::SCALE.end()),
            fold_limit: fold_limit.clamp(*Self::FOLD_LIMIT.start(), *Self::FOLD_LIMIT.end()),
            inner_radius: inner_radius.clamp(Self::MIN_INNER_RADIUS, outer_radius),
            outer_radius,
        }
    }
}

#[derive(Default, Debug, PartialEq)]
struct MandelbulbParameters {
    polar_offset: f32,
    azimuth_twist: f32,
    seed_offset: f32,
}

impl MandelbulbParameters {
    const POLAR_OFFSET: std::ops::RangeInclusive<f32> = -1.5..=1.5;
    const AZIMUTH_TWIST: std::ops::RangeInclusive<f32> = -4.0..=4.0;
    const SEED_OFFSET: std::ops::RangeInclusive<f32> = -1.0..=1.0;

    fn ui(&mut self, ui: &mut egui::Ui) {
        precision_slider(ui, &mut self.polar_offset, Self::POLAR_OFFSET, "Polar offset", false, false)
            .on_hover_text("Shifts the latitude of each spherical fold.");
        precision_slider(ui, &mut self.azimuth_twist, Self::AZIMUTH_TWIST, "Azimuth twist", false, false)
            .on_hover_text("Twists each fold by an amount that grows with its distance from the center.");
        precision_slider(ui, &mut self.seed_offset, Self::SEED_OFFSET, "Seed offset X", false, false)
            .on_hover_text("Changes the seed added at each iteration, breaking left-right symmetry.");
    }

    fn values(&self) -> [f32; 3] {
        [self.polar_offset, self.azimuth_twist, self.seed_offset]
    }

    fn from_preset(values: &[f32]) -> Self {
        let [polar, twist, seed] = values else { return Self::default(); };
        Self {
            polar_offset: polar.clamp(*Self::POLAR_OFFSET.start(), *Self::POLAR_OFFSET.end()),
            azimuth_twist: twist.clamp(*Self::AZIMUTH_TWIST.start(), *Self::AZIMUTH_TWIST.end()),
            seed_offset: seed.clamp(*Self::SEED_OFFSET.start(), *Self::SEED_OFFSET.end()),
        }
    }
}

#[derive(Debug, PartialEq)]
struct MutationStage {
    formula: Option<Fractal>,
    weight: f32,
    box_parameters: MandelboxParameters,
    bulb_parameters: MandelbulbParameters,
    bulb_power: f32,
    menger_parameters: MengerParameters,
    fold_parameters: FoldParameters,
    sierpinski: SierpinskiParameters,
    four_branch: FourBranchParameters,
    orbit_depths: OrbitDepths,
}

impl Default for MutationStage {
    fn default() -> Self {
        Self {
            formula: None, weight: 0.25,
            box_parameters: MandelboxParameters::default(),
            bulb_parameters: MandelbulbParameters::default(),
            bulb_power: 8.0,
            menger_parameters: MengerParameters::default(),
            fold_parameters: FoldParameters::default(),
            sierpinski: SierpinskiParameters::default(),
            four_branch: FourBranchParameters::default(),
            orbit_depths: OrbitDepths::default(),
        }
    }
}

impl MutationStage {
    fn selected(&self, base: Fractal) -> Option<Fractal> {
        self.formula.filter(|formula| *formula != base)
    }

    fn ui(&mut self, ui: &mut egui::Ui, base: Fractal) {
        let mut selected = self.selected(base);
        let previous = selected;
        egui::ComboBox::from_label("Mutation")
            .selected_text(selected.map_or("None", Fractal::name))
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut selected, None, "None");
                for fractal in Fractal::ALL {
                    if fractal != base {
                        ui.selectable_value(&mut selected, Some(fractal), fractal.name());
                    }
                }
            });
        if selected != previous { self.formula = selected; }
        if let Some(formula) = selected {
            precision_slider(ui, &mut self.weight, 0.0..=1.0, "Weight", false, false)
                .on_hover_text("Contribution relative to the base formula and the other active mutations.");
            ui.collapsing("Mutation parameters", |ui| {
                self.orbit_depths.ui(ui, formula);
                let reset = match formula {
                    Fractal::Mandelbox => self.box_parameters.ui(ui),
                    Fractal::Mandelbulb => {
                        precision_slider(ui, &mut self.bulb_power, 2.0..=23.0, "Power", false, false);
                        self.bulb_parameters.ui(ui);
                        let reset = ui.button("Reset mutation parameters").clicked();
                        if reset {
                            self.bulb_power = 8.0;
                            self.bulb_parameters = MandelbulbParameters::default();
                        }
                        reset
                    }
                    Fractal::Menger => { self.menger_parameters.ui(ui); false }
                    Fractal::Sierpinski => { self.sierpinski.ui(ui); false }
                    Fractal::FourBranch => { self.four_branch.ui(ui); false }
                    Fractal::Kifs | Fractal::AmazingSurface | Fractal::PseudoKleinian => self.fold_parameters.ui(ui, formula),
                };
                if reset { self.orbit_depths.reset(formula); }
            });
        }
    }

    fn values(&self) -> [f32; 16] {
        // The third slot held the legacy repeat count.
        let [scale, fold, inner, outer] = self.box_parameters.values();
        let [polar, twist, seed] = self.bulb_parameters.values();
        let [depth, width, z, x, y] = self.menger_parameters.values();
        [self.formula.map_or(-1.0, Fractal::value), self.weight, 1.0,
         scale, fold, inner, outer, polar, twist, seed, self.bulb_power, depth, width, z, x, y]
    }

    fn field_uniforms(&self, formula: Fractal, coefficient: f32) -> [f32; EXTRA_MUTATION_UNIFORMS] {
        let mut data = [0.0; EXTRA_MUTATION_UNIFORMS];
        data[0] = coefficient;
        data[1] = formula.value();
        if let Some(index) = OrbitDepths::index(formula) {
            data[2] = self.orbit_depths.offsets()[index];
        }
        let folds = self.fold_parameters.uniforms();
        let parameters = match formula {
            Fractal::Mandelbox => self.box_parameters.uniforms(),
            Fractal::Mandelbulb => {
                let [polar, twist, seed] = self.bulb_parameters.values();
                [polar, twist, seed, self.bulb_power]
            }
            Fractal::Menger => {
                data[4..].copy_from_slice(&self.menger_parameters.uniforms());
                return data;
            }
            Fractal::Kifs => folds[0..4].try_into().unwrap(),
            Fractal::AmazingSurface => folds[4..8].try_into().unwrap(),
            Fractal::PseudoKleinian => folds[8..12].try_into().unwrap(),
            Fractal::Sierpinski => [self.sierpinski.depth as f32, 0.0, 0.0, 0.0],
            Fractal::FourBranch => {
                data[4..9].copy_from_slice(&self.four_branch.uniforms());
                return data;
            }
        };
        data[4..8].copy_from_slice(&parameters);
        data
    }

    fn stacked_preset_values(&self) -> [f32; MUTATION_PRESET_FIELDS] {
        let mut values = [0.0; MUTATION_PRESET_FIELDS];
        values[..16].copy_from_slice(&self.values());
        values[16..].copy_from_slice(&self.fold_parameters.values());
        values
    }

    fn from_preset(values: &[f32]) -> Self {
        let [formula, weight, _, _, _, _, _, _, _, _, power, ..] = values else {
            return Self::default();
        };
        Self {
            formula: Fractal::from_value(*formula),
            weight: weight.clamp(0.0, 1.0),
            box_parameters: MandelboxParameters::from_preset(&values[3..7]),
            bulb_parameters: MandelbulbParameters::from_preset(&values[7..10]),
            bulb_power: power.clamp(2.0, 23.0),
            menger_parameters: MengerParameters::from_preset(&values[11..]),
            fold_parameters: FoldParameters::default(),
            sierpinski: SierpinskiParameters::default(),
            four_branch: FourBranchParameters::default(),
            orbit_depths: OrbitDepths::default(),
        }
    }
}

fn color_edit_button(ui: &mut egui::Ui, color: &mut [f32; 3]) {
    // The RGB widget converts through HSV even when it only paints the swatch.
    let mut edited = *color;
    if ui.color_edit_button_rgb(&mut edited).changed() {
        *color = edited;
    }
}

fn precision_slider(ui: &mut egui::Ui, value: &mut f32, range: std::ops::RangeInclusive<f32>, label: &str, logarithmic: bool, degree_step: bool) -> egui::Response {
    let lo = *range.start();
    let hi = *range.end();
    let log_floor = if lo == 0.0 { 0.0001 } else { lo };
    let (shift, dx) = ui.input(|i| (i.modifiers.shift, i.pointer.delta().x));
    let track_origin = ui.next_widget_position();
    let track_width = ui.spacing().slider_width.max(1.0);
    let ctx = ui.ctx();
    // The slider and its numeric field are separate drag targets. Only the track
    // needs custom handling. The numeric field already supports Shift.
    let track = ctx.dragged_id().or_else(|| ctx.drag_stopped_id())
        .and_then(|id| ctx.read_response(id))
        .filter(|r| {
            (r.rect.left() - track_origin.x).abs() < 1.0
                && (r.rect.width() - track_width).abs() < 1.0
                && r.rect.y_range().contains(track_origin.y + 1.0)
        });
    let fine_value = track.and_then(|track| {
        let fine_id = track.id.with("precision_drag");
        let started = ctx.drag_started_id() == Some(track.id);
        let stopped = ctx.drag_stopped_id() == Some(track.id);
        let relative = ctx.data_mut(|data| {
            let relative = shift || (!started && data.get_temp::<bool>(fine_id).unwrap_or(false));
            data.insert_temp(fine_id, relative && !stopped);
            relative
        });
        relative.then(|| {
            let sensitivity = if shift { 0.1 } else { 1.0 };
            let motion = if started { 0.0 } else { dx * sensitivity / track_width };
            if motion == 0.0 {
                *value
            } else if logarithmic {
                let next = value.max(log_floor) * ((hi / log_floor).ln() * motion).exp();
                if lo == 0.0 && next < log_floor { 0.0 } else { next }
            } else {
                *value + (hi - lo) * motion
            }
        })
    });
    let mut slider = egui::Slider::from_get_set(f64::from(lo)..=f64::from(hi), |edit| {
        if let Some(edited) = edit {
            *value = fine_value.unwrap_or(edited as f32).clamp(lo, hi);
        }
        f64::from(*value)
    }).text(label).logarithmic(logarithmic).clamping(egui::SliderClamping::Edits);
    if logarithmic && lo == 0.0 {
        slider = slider.smallest_positive(f64::from(log_floor)).smart_aim(false);
    }
    if degree_step { slider = slider.step_by(if shift { 0.1 } else { 1.0 }).fixed_decimals(1).suffix("°"); }
    ui.add(slider)
}
struct Camera {
    position: [f32; 3],
    velocity: [f32; 3],
    yaw: f32,
    pitch: f32,
    roll: f32,
    // Offset between the transported view frame and the world-up frame.
    look_roll: f32,
    speed: f32,
}
impl Default for Camera {
    fn default() -> Self { Self { position: [0.0, 0.0, 3.2], velocity: [0.0; 3], yaw: 0.0, pitch: 0.0, roll: 0.0, look_roll: 0.0, speed: 1.0 } }
}
impl Camera {
    fn starting_view() -> Self {
        // Equal offsets along the three axes frame the Menger like the app icon.
        Self {
            position: [6.0; 3],
            yaw: -std::f32::consts::FRAC_PI_4,
            pitch: -(1.0 / 2.0_f32.sqrt()).atan(),
            ..Self::default()
        }
    }

    fn framed_view(fractal: Fractal) -> Self {
        let position: [f32; 3] = match fractal {
            Fractal::Mandelbulb => [4.0; 3],
            Fractal::Mandelbox => [12.0; 3],
            Fractal::AmazingSurface => [2.8, 1.0, 4.9],
            Fractal::PseudoKleinian => [1.1, 0.6, 3.4],
            Fractal::FourBranch => [2.8, 1.8, 4.2],
            _ => return Self::starting_view(),
        };
        let [x, y, z] = position;
        Self { position, yaw: (-x).atan2(z), pitch: (-y).atan2(x.hypot(z)), ..Self::default() }
    }

    fn view_roll(&self) -> f32 {
        self.roll + self.look_roll
    }

    fn basis(&self, roll: f32) -> [[f32; 3]; 3] {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        let forward = [sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch];
        let base_right = [cos_yaw, 0.0, sin_yaw];
        let base_up = [-sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch];
        let (sin_roll, cos_roll) = roll.sin_cos();
        let right = std::array::from_fn(|i| base_right[i] * cos_roll + base_up[i] * sin_roll);
        let up = std::array::from_fn(|i| base_up[i] * cos_roll - base_right[i] * sin_roll);
        [forward, right, up]
    }

    fn look(&mut self, dx: f32, dy: f32) {
        if self.roll == 0.0 {
            self.yaw += dx / self.pitch.cos().abs().max(0.1);
            self.pitch = (self.pitch - dy).clamp(-1.55, 1.55);
            return;
        }
        let angle = dx.hypot(dy);
        if angle == 0.0 { return; }
        let [forward, right, up] = self.basis(self.view_roll());
        let (sin, cos) = angle.sin_cos();
        let horizontal = dx / angle;
        let tangent = std::array::from_fn::<_, 3, _>(|i| horizontal * right[i] - (dy / angle) * up[i]);
        // Rotate the whole view frame in the plane of the pointer motion.
        let [x, y, z] = std::array::from_fn(|i| forward[i] * cos + tangent[i] * sin);
        let rotated_right: [f32; 3] = std::array::from_fn(|i| {
            right[i] + horizontal * (tangent[i] * (cos - 1.0) - forward[i] * sin)
        });
        let horizontal_length = x.hypot(z);
        if horizontal_length > 1.0e-7 { self.yaw = x.atan2(-z); }
        self.pitch = y.atan2(horizontal_length);
        let [_, base_right, base_up] = self.basis(0.0);
        let project = |axis: [f32; 3]| rotated_right.iter().zip(axis).map(|(r, a)| r * a).sum::<f32>();
        self.look_roll = project(base_up).atan2(project(base_right)) - self.roll;
    }

    fn set_roll(&mut self, roll: f32) {
        self.roll = roll;
        if roll == 0.0 {
            self.look_roll = 0.0;
            self.pitch = self.pitch.clamp(-1.55, 1.55);
        }
    }

    fn update(&mut self, keys: &HashSet<KeyCode>, dt: f32) -> bool {
        let [forward, right, up] = self.basis(self.view_roll());
        let axis = |positive, negative| (keys.contains(&positive) as i32 - keys.contains(&negative) as i32) as f32;
        let f = axis(KeyCode::KeyW, KeyCode::KeyS);
        let r = axis(KeyCode::KeyD, KeyCode::KeyA);
        let u = axis(KeyCode::KeyE, KeyCode::KeyQ);
        let direction = std::array::from_fn::<_, 3, _>(|i| forward[i] * f + right[i] * r + up[i] * u);
        let length = direction.iter().map(|v| v * v).sum::<f32>().sqrt().max(1.0);
        let blend = 1.0 - (-5.0 * dt).exp();
        for i in 0..3 {
            self.velocity[i] += (direction[i] / length * self.speed - self.velocity[i]) * blend;
            self.position[i] += self.velocity[i] * dt;
        }
        let moving = self.velocity.iter().any(|v| v.abs() > 0.0001) || f != 0.0 || r != 0.0 || u != 0.0;
        if !moving { self.velocity = [0.0; 3]; }
        moving
    }
}
fn flag(enabled: bool) -> f32 { f32::from(u8::from(enabled)) }

fn text_edit_is_focused(ctx: &egui::Context) -> bool {
    ctx.memory(|memory| memory.focused())
        .is_some_and(|id| egui::TextEdit::load_state(ctx, id).is_some())
}

struct OutputSettings {
    width: u32, height: u32, samples: u32,
    denoise: Option<denoise::Backend>, viewport_denoise: Option<denoise::Backend>,
    viewport_denoise_interval: u32,
    intelligent_sampling: bool,
    viewport_noise: noise_sampling::Settings,
    render_noise: noise_sampling::Settings,
    time_limit: f32,
    status: String, progress: f32,
}
impl Default for OutputSettings { fn default() -> Self { Self {
    width: 1920, height: 1080, samples: 16, denoise: None, viewport_denoise: None,
    viewport_denoise_interval: viewport_denoise::DEFAULT_INTERVAL, intelligent_sampling: false,
    viewport_noise: noise_sampling::Settings::new(0.1), render_noise: noise_sampling::Settings::new(0.01),
    time_limit: 0.0, status: String::new(), progress: 0.0,
} } }
struct RenderJob { receiver: std::sync::mpsc::Receiver<export::Update>, cancel: Arc<std::sync::atomic::AtomicBool> }

#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
enum ControlsTab {
    #[default]
    Formula,
    Surface,
    Material,
    Lighting,
    Color,
    Camera,
    Presets,
}

impl ControlsTab {
    const ALL: [Self; 7] = [Self::Formula, Self::Surface, Self::Material, Self::Lighting, Self::Camera, Self::Color, Self::Presets];

    fn label(self) -> &'static str {
        match self {
            Self::Formula => "Formula",
            Self::Surface => "Surface",
            Self::Material => "Material",
            Self::Lighting => "Lighting",
            Self::Color => "Color",
            Self::Camera => "Camera",
            Self::Presets => "Presets",
        }
    }

    fn button(self, ui: &mut egui::Ui, selected: bool) -> egui::Response {
        let response = ui.add_sized([27.0, 27.0], egui::Button::new("").selected(selected));
        response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, ui.is_enabled(), selected, self.label()));
        let painter = ui.painter();
        let color = ui.style().interact_selectable(&response, selected).text_color();
        let stroke = egui::Stroke::new(1.35_f32, color);
        let center = response.rect.center();
        let point = |x, y| center + egui::vec2(x, y);
        let outline = |points: Vec<egui::Pos2>| painter.add(egui::Shape::closed_line(points, stroke));
        match self {
            Self::Formula => {
                outline(vec![point(0.0, -7.0), point(7.0, 6.0), point(-7.0, 6.0)]);
                outline(vec![point(-3.5, -0.5), point(3.5, -0.5), point(0.0, 6.0)]);
            }
            Self::Surface => {
                for y in [-4.5, 0.0, 4.5] {
                    painter.add(egui::Shape::line(vec![point(-7.0, y + 1.5), point(-2.0, y - 1.5), point(2.0, y + 1.5), point(7.0, y - 1.5)], stroke));
                }
            }
            Self::Material => {
                painter.circle_stroke(center, 6.5, stroke);
                painter.circle_filled(point(-2.0, -2.0), 1.8, color);
            }
            Self::Lighting => {
                painter.circle_stroke(center, 3.5, stroke);
                for i in 0..8 {
                    let angle = i as f32 * std::f32::consts::FRAC_PI_4;
                    let ray = egui::vec2(angle.cos(), angle.sin());
                    painter.line_segment([center + ray * 5.5, center + ray * 8.0], stroke);
                }
            }
            Self::Color => {
                for (y, handle) in [(-4.0, -2.5), (4.0, 2.5)] {
                    painter.line_segment([point(-7.0, y), point(7.0, y)], stroke);
                    painter.circle_filled(point(handle, y), 2.2, color);
                }
            }
            Self::Camera => {
                let body = egui::Rect::from_center_size(point(-2.5, 0.0), egui::vec2(10.0, 9.0));
                painter.rect_stroke(body, 1.0, stroke, egui::StrokeKind::Inside);
                outline(vec![point(2.5, -2.5), point(7.0, -5.0), point(7.0, 5.0), point(2.5, 2.5)]);
            }
            Self::Presets => {
                let rect = egui::Rect::from_center_size(center, egui::vec2(14.0, 12.0));
                painter.rect_stroke(rect, 2.0, stroke, egui::StrokeKind::Inside);
                painter.add(egui::Shape::line(vec![point(-5.0, 4.0), point(-1.0, 0.0), point(2.0, 3.0), point(5.0, 0.0)], stroke));
            }
        }
        response.on_hover_text(self.label())
    }
}
struct App { still_timing: StillTiming, pixel_sampling: pixel_sampling::PixelSampling, navigation_sample: u32, viewport_was_interacting: bool, adaptive_viewport: adaptive_viewport::AdaptiveViewport, hdri: Option<hdri::Panorama>, hdri_input: String, hdri_status: String, about: about::About, startup: startup::Startup, event_proxy: Option<winit::event_loop::EventLoopProxy<startup::Event>>, window: Option<Arc<Window>>, renderer: Option<Renderer>, camera: Camera, keys: HashSet<KeyCode>, looking: bool, last_frame: Instant, slice: Option<f32>, power: f32, fractal: Fractal, mutation: MutationStage, extra_mutations: Vec<MutationStage>, base_weight: f32, menger_parameters: MengerParameters, mandelbox_parameters: MandelboxParameters, mandelbulb_parameters: MandelbulbParameters, fold_parameters: FoldParameters, sierpinski: SierpinskiParameters, four_branch: FourBranchParameters, orbit_depths: OrbitDepths, diagnostic: bool, material: u32, material_settings: [f32; 5], material_finish: SurfaceFinish, custom_finish: SurfaceFinish, matcap: MatcapMaterial, ao: bool, ao_strength: f32, ao_radius: f32, exposure: f32, contrast: f32, normal_smoothing: f32, quality: f32, distant_filter: f32, lighting: Lighting, still_samples: u32, moving_samples: u32, moving_bounces: u32, progress_samples: u32, progress_tile: u32, refine_tiles_per_frame: u32, idle_deadline: Option<Instant>, idle_ready: bool, last_scene: Vec<f32>, egui_ctx: egui::Context, egui_state: Option<egui_winit::State>, preset_status: String, presets: preset_gallery::Gallery, capture_slot: Option<u64>, output: OutputSettings, render_job: Option<RenderJob>, start_export: bool, left_open: bool, controls_tab: ControlsTab, right_open: bool, effects: CameraEffects, background_color: [f32; 3], show_frame: bool, frame: Option<egui::Rect>, picking_focus: bool, focus_point: Option<[f32; 2]>, focus_status: String, history: history::History, history_actions: Vec<history::Action>, modifiers: winit::keyboard::ModifiersState }
impl Default for App {
    fn default() -> Self {
        let camera = Camera::starting_view();
        let effects = CameraEffects { focal: STARTING_FOCAL_MM, ..CameraEffects::default() };
        Self { still_timing: StillTiming::default(), pixel_sampling: pixel_sampling::PixelSampling::default(), navigation_sample: 0, viewport_was_interacting: false, adaptive_viewport: adaptive_viewport::AdaptiveViewport::default(), hdri: None, hdri_input: String::new(), hdri_status: String::new(), about: about::About::default(), startup: startup::Startup::default(), event_proxy: None, window: None, renderer: None, camera, keys: HashSet::new(), looking: false, last_frame: Instant::now(), slice: None, power: 8.0, fractal: Fractal::default(), mutation: MutationStage::default(), extra_mutations: Vec::new(), base_weight: 1.0, menger_parameters: MengerParameters::default(), mandelbox_parameters: MandelboxParameters::default(), mandelbulb_parameters: MandelbulbParameters::default(), fold_parameters: FoldParameters::default(), sierpinski: SierpinskiParameters::default(), four_branch: FourBranchParameters::default(), orbit_depths: OrbitDepths::default(), diagnostic: false, material: MATCAP_MATERIAL, material_settings: DIFFUSE_DEFAULTS, material_finish: SurfaceFinish::default(), custom_finish: SurfaceFinish::default(), matcap: MatcapMaterial::default(), ao: false, ao_strength: 0.5, ao_radius: 0.048, exposure: 0.0, contrast: 1.0, normal_smoothing: 0.002, quality: 1.0, distant_filter: 2.0, lighting: Lighting::default(), still_samples: DEFAULT_VIEWPORT_SAMPLES, moving_samples: 1, moving_bounces: 1, progress_samples: 0, progress_tile: 0, refine_tiles_per_frame: 1, idle_deadline: None, idle_ready: false, last_scene: Vec::new(), egui_ctx: egui::Context::default(), egui_state: None, preset_status: String::new(), presets: preset_gallery::Gallery::default(), capture_slot: None, output: OutputSettings::default(), render_job: None, start_export: false, left_open: true, controls_tab: ControlsTab::default(), right_open: true, effects, background_color: DEFAULT_BACKGROUND, show_frame: true, frame: None, picking_focus: false, focus_point: None, focus_status: String::new(), history: history::History::default(), history_actions: Vec::new(), modifiers: winit::keyboard::ModifiersState::empty() }
    }
}

#[derive(Default)]
struct StillTiming {
    started: Option<Instant>,
    pending: Option<std::sync::mpsc::Receiver<Instant>>,
    completed: Option<Duration>,
}

impl StillTiming {
    fn finish(&mut self, queue: &wgpu::Queue) {
        if self.started.is_none() || self.pending.is_some() || self.completed.is_some() { return; }
        let (send, pending) = std::sync::mpsc::channel();
        queue.on_submitted_work_done(move || { let _ = send.send(Instant::now()); });
        self.pending = Some(pending);
    }
}
impl App {
    fn section_active(&self) -> bool { self.fractal == Fractal::Mandelbulb && self.slice.is_some() }

    fn ray_traced_ao_preview(&self) -> bool {
        !self.lighting.path_traced && self.ao && self.ao_strength > 0.0 && !self.diagnostic && !self.section_active()
    }

    fn path_traced_preview(&self) -> bool {
        self.lighting.path_traced && !self.diagnostic && !self.section_active()
    }

    fn has_directional_shadows(&self) -> bool {
        self.lighting.enabled && self.lighting.shadows && self.lighting.intensity > 0.0
    }

    fn fractal_title(&self) -> String {
        format!("{APP_NAME} - {}", self.fractal.name())
    }

    fn viewport_scene(&self, size: winit::dpi::PhysicalSize<u32>, projection: [f32; 4]) -> Vec<f32> {
        let mut scene = vec![self.camera.position[0], self.camera.position[1], self.camera.position[2], self.camera.yaw, self.camera.pitch, self.camera.roll, self.slice.is_some() as u8 as f32, self.slice.unwrap_or(0.0), self.power, self.fractal.value(), self.diagnostic as u8 as f32, self.material as f32, self.ao as u8 as f32, self.ao_strength, self.ao_radius, self.exposure, self.normal_smoothing, self.quality, self.distant_filter, self.lighting.color[0], self.lighting.color[1], self.lighting.color[2], self.lighting.roughness, self.lighting.metallic, self.lighting.light_color[0], self.lighting.light_color[1], self.lighting.light_color[2], self.lighting.azimuth, self.lighting.elevation, self.lighting.intensity, size.width as f32, size.height as f32];
        scene.extend(self.lighting.environment_color);
        scene.extend([self.lighting.environment_strength, self.lighting.angular_radius]);
        scene.extend(self.lighting.path_preset());
        scene.extend([flag(self.lighting.enabled), flag(self.lighting.shadows)]);
        scene.extend(self.effects.preset());
        scene.extend(self.effects.bokeh());
        scene.extend(self.material_settings);
        scene.extend(self.material_finish.values());
        scene.extend(self.custom_finish.values());
        scene.extend(self.matcap.values());
        scene.push(self.camera.look_roll);
        scene.extend(self.mandelbox_parameters.values());
        scene.extend(self.mandelbulb_parameters.values());
        scene.extend(self.menger_parameters.values());
        scene.extend(self.background_color);
        scene.extend(self.mutation.values());
        scene.extend(self.fold_parameters.values());
        scene.extend(self.mutation.fold_parameters.values());
        scene.extend([self.sierpinski.depth as f32, self.mutation.sierpinski.depth as f32]);
        scene.extend(self.orbit_depths.values());
        scene.extend(self.mutation.orbit_depths.values());
        scene.extend(self.four_branch.values());
        scene.extend(self.mutation.four_branch.values());
        for stage in &self.extra_mutations {
            scene.extend(stage.stacked_preset_values());
            scene.push(stage.sierpinski.depth as f32);
            scene.extend(stage.orbit_depths.values());
            scene.extend(stage.four_branch.values());
        }
        scene.push(self.base_weight);
        scene.extend(projection);
        scene.push(flag(self.adaptive_viewport.enabled));
        scene.push(self.pixel_sampling.uniform());
        scene.push(self.adaptive_viewport.budget_ms as f32);
        scene.extend([self.moving_samples as f32, self.moving_bounces as f32]);
        scene
    }

    fn still_sample_target(&self) -> u32 {
        if self.diagnostic { 1 }
        else { self.still_samples }
    }

    fn ui(&mut self, ctx: &egui::Context) {
        if self.right_open {
            egui::SidePanel::right("render_output").default_width(255.0).min_width(255.0).resizable(true).show(ctx, |ui| {
                ui.set_min_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.heading("Render");
                    if ui.small_button("▶").on_hover_text("Hide render panel").clicked() { self.right_open = false; }
                    self.about_button(ui);
                });
                ui.separator();
                ui.add_enabled_ui(self.render_job.is_none(), |ui| {
                    if ui.button("Render PNG").clicked() { self.start_export = true; }
                });
                if let Some(job) = &self.render_job {
                    ui.add(egui::ProgressBar::new(self.output.progress).show_percentage());
                    if ui.button("Cancel render").clicked() { job.cancel.store(true, std::sync::atomic::Ordering::Relaxed); }
                }
                if !self.output.status.is_empty() { ui.label(&self.output.status); }
                ui.separator();
                let body = ui.available_rect_before_wrap();
                let settings_rect = body.shrink2(egui::vec2(4.0, 0.0));
                let mut settings_ui = ui.new_child(egui::UiBuilder::new().id_salt("render_body").max_rect(settings_rect));
                // Hovered widgets can paint outside their layout rectangle, inside the panel gutter.
                settings_ui.set_clip_rect(ui.clip_rect().intersect(body));
                ui.allocate_space(body.size());
                egui::ScrollArea::vertical().id_salt("render_settings").auto_shrink([false, false]).show(&mut settings_ui, |ui| {
                    ui.spacing_mut().slider_width = 64.0;
                    self.render_settings_ui(ui);
                });
            });
        } else {
            egui::Area::new("open_render".into()).anchor(egui::Align2::RIGHT_TOP, [-5.0, 5.0]).show(ctx, |ui| {
                ui.horizontal(|ui| {
                    self.about_button(ui);
                    if ui.small_button("◀").on_hover_text("Show render panel").clicked() { self.right_open = true; }
                });
            });
        }
        if self.left_open {
            egui::SidePanel::left("controls")
                .default_width(320.0)
                .min_width(CONTROLS_MIN_WIDTH)
                .resizable(true)
                .show(ctx, |ui| {
                    ui.set_min_width(ui.available_width());
                    ui.horizontal(|ui| {
                        if ui.small_button("◀").on_hover_text("Hide controls").clicked() {
                            self.left_open = false;
                        }
                        ui.heading("Controls");
                        if ui.add_enabled(self.history.can_undo(), egui::Button::new("Undo").small())
                            .on_hover_text("Undo (Ctrl+Z)").clicked()
                        {
                            self.history_actions.push(history::Action::Undo);
                        }
                        if ui.add_enabled(self.history.can_redo(), egui::Button::new("Redo").small())
                            .on_hover_text("Redo (Ctrl+Shift+Z or Ctrl+Y)").clicked()
                        {
                            self.history_actions.push(history::Action::Redo);
                        }
                    });
                    ui.separator();
                    // A tab's contents must not resize the panel and change the camera projection.
                    let body = ui.available_rect_before_wrap();
                    let mut tab_ui = ui.new_child(egui::UiBuilder::new().id_salt("tab_body").max_rect(body));
                    tab_ui.set_clip_rect(ui.clip_rect().intersect(body));
                    ui.allocate_space(body.size());
                    tab_ui.horizontal_top(|ui| {
                        ui.allocate_ui_with_layout(egui::vec2(27.0, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                            for tab in ControlsTab::ALL {
                                if tab.button(ui, self.controls_tab == tab).clicked() {
                                    self.controls_tab = tab;
                                }
                            }
                        });
                        ui.separator();
                        let tab = self.controls_tab;
                        ui.allocate_ui_with_layout(ui.available_size_before_wrap(), egui::Layout::top_down(egui::Align::Min), |ui| {
                            if tab == ControlsTab::Presets {
                                ui.heading(tab.label());
                                if let Some(action) = self.presets.ui(ui, &self.preset_status) {
                                    self.preset_action(action);
                                }
                                return;
                            }
                            egui::ScrollArea::vertical().id_salt(tab).auto_shrink([false, false]).show(ui, |ui| {
                                ui.spacing_mut().slider_width = 88.0;
                                ui.heading(tab.label());
                                match tab {
                                    ControlsTab::Formula => self.formula_ui(ui),
                                    ControlsTab::Surface => self.surface_ui(ui),
                                    ControlsTab::Material => self.material_ui(ui),
                                    ControlsTab::Lighting => self.lighting_ui(ui),
                                    ControlsTab::Color => self.color_ui(ui),
                                    ControlsTab::Camera => self.camera_ui(ui),
                                    ControlsTab::Presets => unreachable!(),
                                }
                            });
                        });
                    });
                });
        } else {
            egui::Area::new("open_controls".into()).anchor(egui::Align2::LEFT_TOP, [5.0, 5.0]).show(ctx, |ui| {
                if ui.small_button("▶").on_hover_text("Show controls").clicked() { self.left_open = true; }
            });
        }
        if !self.left_open || self.controls_tab != ControlsTab::Presets {
            self.presets.hide();
        }
        self.about.show(ctx, self.event_proxy.as_ref());
    }

    fn about_button(&mut self, ui: &mut egui::Ui) {
        if ui.small_button("About").clicked() {
            self.keys.clear();
            self.looking = false;
            self.camera.velocity = [0.0; 3];
            self.about.open = true;
        }
    }
    fn formula_ui(&mut self, ui: &mut egui::Ui) {
        let mut selected = self.fractal;
        egui::ComboBox::from_label("Fractal")
            .selected_text(self.fractal.name())
            .show_ui(ui, |ui| {
                for fractal in Fractal::ALL {
                    ui.selectable_value(&mut selected, fractal, fractal.name());
                }
            });
        if selected != self.fractal {
            self.select_fractal(selected);
        }
        self.orbit_depths.ui(ui, self.fractal);
        let reset = match self.fractal {
            Fractal::Mandelbulb => {
                precision_slider(ui, &mut self.power, 2.0..=23.0, "Power", false, false);
                self.mandelbulb_parameters.ui(ui);
                let reset = ui.button("Reset Mandelbulb parameters").clicked();
                if reset {
                    self.power = 8.0;
                    self.mandelbulb_parameters = MandelbulbParameters::default();
                }
                let mut enabled = self.slice.is_some();
                if ui.checkbox(&mut enabled, "Section").changed() { self.slice = enabled.then_some(0.0); }
                if let Some(z) = &mut self.slice { precision_slider(ui, z, -2.0..=2.0, "Section Z", false, false); }
                reset
            }
            Fractal::Mandelbox => self.mandelbox_parameters.ui(ui),
            Fractal::Menger => { self.menger_parameters.ui(ui); false }
            Fractal::Sierpinski => { self.sierpinski.ui(ui); false }
            Fractal::FourBranch => { self.four_branch.ui(ui); false }
            Fractal::Kifs | Fractal::AmazingSurface | Fractal::PseudoKleinian => self.fold_parameters.ui(ui, self.fractal),
        };
        if reset { self.orbit_depths.reset(self.fractal); }
        ui.separator();
        self.mutations_ui(ui);
    }

    fn select_fractal(&mut self, fractal: Fractal) {
        let starting = Camera::framed_view(self.fractal);
        let at_starting_view = self.camera.position == starting.position
            && self.camera.yaw == starting.yaw && self.camera.pitch == starting.pitch
            && self.camera.view_roll() == 0.0 && self.effects.focal == STARTING_FOCAL_MM;
        self.fractal = fractal;
        if at_starting_view {
            self.reset_framing();
        }
    }

    fn reset_framing(&mut self) {
        self.camera = Camera { speed: self.camera.speed, ..Camera::framed_view(self.fractal) };
        self.effects.set_focal(STARTING_FOCAL_MM);
    }

    fn mutations_ui(&mut self, ui: &mut egui::Ui) {
        precision_slider(ui, &mut self.base_weight, 0.0..=1.0, "Base weight", false, false)
            .on_hover_text("Weight of the base formula. If every weight is zero, the base formula is shown.");
        let has_first = self.mutation.formula.is_some() || !self.extra_mutations.is_empty();
        if has_first {
            ui.horizontal(|ui| {
                ui.strong("Mutation 1");
                if ui.small_button("Remove").clicked() {
                    self.mutation = if self.extra_mutations.is_empty() {
                        MutationStage::default()
                    } else {
                        self.extra_mutations.remove(0)
                    };
                }
            });
            self.mutation.ui(ui, self.fractal);
        }

        let mut remove = None;
        for (index, stage) in self.extra_mutations.iter_mut().enumerate() {
            ui.separator();
            ui.push_id(index, |ui| {
                ui.horizontal(|ui| {
                    ui.strong(format!("Mutation {}", index + 2));
                    if ui.small_button("Remove").clicked() {
                        remove = Some(index);
                    }
                });
                stage.ui(ui, self.fractal);
            });
        }
        if let Some(index) = remove {
            self.extra_mutations.remove(index);
        }
        if ui.add_enabled(
            self.extra_mutations.len() < MAX_MUTATIONS - 1, egui::Button::new("+ Add mutation"),
        ).on_disabled_hover_text("Up to 8 mutations").clicked() {
            let last = self.extra_mutations.last().unwrap_or(&self.mutation).formula;
            let formula = Fractal::ALL.into_iter().find(|&formula| {
                formula != self.fractal && Some(formula) != last
            });
            let stage = MutationStage { formula, ..MutationStage::default() };
            if has_first {
                self.extra_mutations.push(stage);
            } else {
                self.mutation = stage;
            }
        }
    }

    fn surface_ui(&mut self, ui: &mut egui::Ui) {
        precision_slider(ui, &mut self.quality, DETAIL_QUALITY, "Detail quality", false, false);
        ui.small("Higher quality can slow down close views.");
        precision_slider(ui, &mut self.normal_smoothing, SMOOTHNESS, "Smoothness", true, false);
        precision_slider(ui, &mut self.distant_filter, 0.5..=4.0, "Distant smoothing", false, false);
        ui.checkbox(&mut self.diagnostic, "Surface diagnostic");
        if self.diagnostic {
            ui.colored_label(egui::Color32::from_rgb(64, 200, 220), "Cyan: surface found");
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Green: found with extra steps (still)");
            ui.colored_label(egui::Color32::YELLOW, "Yellow: approximate surface");
            ui.colored_label(egui::Color32::LIGHT_RED, "Red: step limit reached");
            ui.small("Dark blue: no surface found within range.");
        }
    }

    fn material_ui(&mut self, ui: &mut egui::Ui) {
        let mut selected = self.material;
        egui::ComboBox::from_label("Material").selected_text(self.material_name()).show_ui(ui, |ui| {
            for (id, name) in MATERIALS {
                ui.selectable_value(&mut selected, id, name);
            }
        });
        if selected != self.material {
            self.select_material(selected);
        }
        ui.separator();
        if self.material == DIFFUSE_GRAY_MATERIAL {
            let [r, g, b, ..] = &mut self.material_settings;
            let mut color = [*r, *g, *b];
            Self::base_color_ui(ui, &mut color);
            [*r, *g, *b] = color;
            return;
        }
        ui.heading("Principled");
        let (roughness, metallic, finish) = match self.material {
            MATCAP_MATERIAL => {
                egui::ComboBox::from_label("Base color matcap").selected_text(self.matcap.style_name()).show_ui(ui, |ui| {
                    for (id, name) in MATCAP_STYLES {
                        ui.selectable_value(&mut self.matcap.style, id, name);
                    }
                });
                if self.matcap.style != 0 {
                    precision_slider(ui, &mut self.matcap.rotation, -180.0..=180.0, "Matcap rotation", false, true);
                }
                (&mut self.matcap.roughness, &mut self.matcap.metallic, &mut self.matcap.finish)
            }
            _ => {
                Self::base_color_ui(ui, &mut self.lighting.color);
                (&mut self.lighting.roughness, &mut self.lighting.metallic, &mut self.custom_finish)
            }
        };
        precision_slider(ui, roughness, 0.05..=1.0, "Roughness", false, false);
        precision_slider(ui, metallic, 0.0..=1.0, "Metallic", false, false);
        egui::CollapsingHeader::new("Specular").show(ui, |ui| {
            precision_slider(ui, &mut finish.ior, 1.0..=3.0, "IOR", false, false);
            precision_slider(ui, &mut finish.specular, 0.0..=1.0, "IOR level", false, false);
        });
        egui::CollapsingHeader::new("Coat").show(ui, |ui| {
            precision_slider(ui, &mut finish.coat, 0.0..=1.0, "Weight", false, false);
            precision_slider(ui, &mut finish.coat_roughness, 0.05..=1.0, "Roughness", false, false);
        });
        egui::CollapsingHeader::new("Emission").show(ui, |ui| {
            ui.horizontal(|ui| { ui.label("Color"); color_edit_button(ui, &mut finish.emission_color); });
            precision_slider(ui, &mut finish.emission_strength, 0.0..=20.0, "Strength", false, false);
        });
    }

    fn select_material(&mut self, material: u32) {
        self.material = material;
        self.material_settings = DIFFUSE_DEFAULTS;
        self.material_finish = SurfaceFinish::default();
    }

    fn material_name(&self) -> &'static str {
        MATERIALS.iter().find(|(id, _)| *id == self.material).unwrap_or(&MATERIALS[1]).1
    }

    fn base_color_ui(ui: &mut egui::Ui, color: &mut [f32; 3]) {
        ui.horizontal(|ui| { ui.label("Base color"); color_edit_button(ui, color); });
    }

    fn color_ui(&mut self, ui: &mut egui::Ui) {
        precision_slider(ui, &mut self.exposure, -5.0..=5.0, "Exposure", false, false);
        precision_slider(ui, &mut self.contrast, CONTRAST_RANGE, "Contrast", false, false)
            .on_hover_text("1.0 preserves the original contrast.");
        if ui.button("Reset color adjustments").clicked() {
            self.exposure = 0.0;
            self.contrast = 1.0;
        }
    }

    fn render_settings_ui(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Engine")
            .selected_text(if self.lighting.path_traced { "Path Traced" } else { "Ambient + AO" })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.lighting.path_traced, false, "Ambient + AO");
                ui.selectable_value(&mut self.lighting.path_traced, true, "Path Traced");
            });
        ui.separator();
        ui.heading("Sampling");
        egui::ComboBox::from_label("Pixel sampling")
            .selected_text(self.pixel_sampling.label())
            .show_ui(ui, |ui| {
                for mode in pixel_sampling::PixelSampling::ALL {
                    ui.selectable_value(&mut self.pixel_sampling, mode, mode.label());
                }
            }).response.on_hover_text("Subpixel sampling for the viewport and PNG renders.");
        ui.heading("Viewport");
        ui.add_enabled_ui(!self.diagnostic && !self.section_active(), |ui| {
            Self::noise_controls(ui, "viewport_noise", &mut self.output.viewport_noise, &mut self.still_samples);
            ui.add_enabled(!self.output.viewport_noise.enabled, egui::Checkbox::new(
                &mut self.output.intelligent_sampling, "Intelligent ray sampling",
            )).on_hover_text("Redistributes rays during still refinement. Available when Noise Threshold is off.");
        });
        ui.add_enabled_ui(!self.diagnostic && !self.section_active(), |ui| {
            denoise::selector(ui, "Denoise viewport", &mut self.output.viewport_denoise);
            ui.add_enabled(self.output.viewport_denoise.is_some(),
                egui::Slider::new(&mut self.output.viewport_denoise_interval, 1..=512)
                    .text("Denoise Interval").suffix(" spp").step_by(1.0).drag_value_speed(1.0).smart_aim(false))
                .on_hover_text("Filters the first sample, then at fixed multiples of this interval, and at the final sample.");
        });
        egui::CollapsingHeader::new("Navigation").default_open(true).show(ui, |ui| {
            let adaptive = &mut self.adaptive_viewport;
            let toggled = ui.checkbox(&mut adaptive.enabled, "Adaptive resolution").changed();
            let budget_changed = ui.add_enabled(adaptive.enabled,
                egui::Slider::new(&mut adaptive.budget_ms, 8..=66).text("Pass budget").suffix(" ms"))
                .on_hover_text("Target time for a complete navigation preview. The visor starts at quarter resolution per axis.").changed();
            if toggled || budget_changed { adaptive.restart(); }
            ui.add_enabled_ui(!self.diagnostic, |ui| {
                egui::ComboBox::from_label("Moving Samples").selected_text(self.moving_samples.to_string()).show_ui(ui, |ui| {
                    for count in MOVING_SAMPLE_COUNTS {
                        ui.selectable_value(&mut self.moving_samples, count, count.to_string());
                    }
                }).response.on_hover_text("Samples per frame while moving or looking around. Higher values can slow navigation.");
            });
            if self.lighting.path_traced {
                ui.add_enabled_ui(self.path_traced_preview(), |ui| {
                    ui.add(egui::Slider::new(&mut self.moving_bounces, 1..=8).text("Moving bounces"))
                        .on_hover_text("Bounces while moving or looking around. Higher values can slow navigation.");
                });
            }
        });
        ui.separator();
        ui.heading("Render PNG");
        ui.add_enabled_ui(self.render_job.is_none(), |ui| {
            Self::noise_controls(ui, "render_noise", &mut self.output.render_noise, &mut self.output.samples);
            ui.horizontal(|ui| {
                ui.label("Time Limit");
                ui.add(egui::DragValue::new(&mut self.output.time_limit).range(0.0..=86400.0).speed(0.1).suffix(" s"))
                    .on_hover_text("Time budget for tracing. 0 is unlimited. Finishes at least one pass over every tile before denoising and saving.");
            });
            denoise::selector(ui, "Denoise PNG", &mut self.output.denoise);
        });
        if !self.section_active() && self.effects.dof_enabled && self.effects.aperture > 0.0 && self.output.samples == 1 {
            ui.small("Use 4 or more PNG samples for smoother depth of field.");
        }
        if self.lighting.path_traced {
            ui.separator();
            ui.heading("Light Paths");
            ui.add(egui::Slider::new(&mut self.lighting.bounces, 1..=8).text("Max bounces"))
                .on_hover_text("Bounces for the still viewport and PNG. 1 lights the visible surface. Higher values add indirect light.");
        }
        ui.separator();
        ui.heading("Output");
        ui.add_enabled_ui(self.render_job.is_none(), |ui| {
            egui::Grid::new("output_dimensions").show(ui, |ui| {
                for (label, dimension) in [("Width", &mut self.output.width), ("Height", &mut self.output.height)] {
                    ui.label(label);
                    ui.add(egui::DragValue::new(dimension).range(64..=8192).speed(4).suffix(" px"));
                    ui.end_row();
                }
            });
            if ui.button("Swap width and height").clicked() {
                std::mem::swap(&mut self.output.width, &mut self.output.height);
            }
        });
        ui.checkbox(&mut self.show_frame, "Show render frame");
        ui.small("Images are saved in the renders folder.");

        ui.separator();
        egui::CollapsingHeader::new("Viewport diagnostics").show(ui, |ui| {
            self.viewport_diagnostics_ui(ui);
        });
    }

    fn noise_controls(ui: &mut egui::Ui, id: &'static str, settings: &mut noise_sampling::Settings, max_samples: &mut u32) {
        egui::Grid::new(id).show(ui, |ui| {
            ui.checkbox(&mut settings.enabled, "Noise Threshold")
                .on_hover_text("Stop sampling pixels when their estimated noise falls below the threshold.");
            ui.add_enabled(settings.enabled, egui::DragValue::new(&mut settings.threshold)
                .range(0.0..=1.0).speed(0.001).fixed_decimals(4))
                .on_hover_text("Lower values refine the image longer. 0 chooses a threshold based on Max Samples.");
            ui.end_row();
            ui.label(if settings.enabled { "Max Samples" } else { "Samples" });
            ui.add(egui::DragValue::new(max_samples).range(1..=noise_sampling::MAX_SAMPLES));
            ui.end_row();
            if settings.enabled {
                ui.label("Min Samples");
                ui.add(egui::DragValue::new(&mut settings.min_samples).range(0..=*max_samples))
                    .on_hover_text("Minimum samples before the convergence check. 0 chooses a value based on Noise Threshold.");
                ui.end_row();
            }
        });
    }

    fn viewport_diagnostics_ui(&self, ui: &mut egui::Ui) {
        // Keep the scroll area stable when measurements appear or disappear.
        let row_height = ui.text_style_height(&egui::TextStyle::Small) + ui.spacing().item_spacing.y;
        ui.set_min_height(12.0 * row_height);
        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Truncate);
        let still_time = self.still_timing.completed.map_or_else(|| "—".to_string(), |elapsed| {
            format!("{:.2} s", elapsed.as_secs_f64())
        });
        ui.small(format!("Still time: {still_time}"))
            .on_hover_text("Elapsed time from the first still pass through the final viewport frame, including denoising when enabled.");
        let adaptive = &self.adaptive_viewport;
        if adaptive.enabled {
            if let Some(stats) = &adaptive.last {
                let [width, height] = stats.size;
                let source = if stats.gpu { "GPU" } else { "completion" };
                ui.small(format!("{width} × {height} · {:.1} ms {source} · {:.2} M primary rays/s",
                    stats.ms, stats.rays as f64 / (stats.ms * 1000.0)));
                if let Some(valid) = stats.valid_fraction {
                    ui.small(format!("Estimated valid: {:.1}% · {:.2} M rays/s",
                        valid * 100.0, valid * stats.rays as f64 / (stats.ms * 1000.0)));
                }
                if let Some(metrics) = stats.metrics_ms {
                    ui.small(format!("{:.1} ms completion · {metrics:.3} ms GPU metrics", stats.completion_ms));
                }
            }
        }
        if self.path_traced_preview() || self.ray_traced_ao_preview() || self.environment_texture().is_some()
            || self.output.viewport_denoise.is_some() || self.output.viewport_noise.enabled
        {
            let target = self.still_sample_target();
            let samples = if self.idle_ready { self.progress_samples.min(target) } else { 0 };
            ui.small(format!("Completed: {samples} / {target} passes"))
                .on_hover_text("A pass can trace different numbers of rays per pixel with Noise Threshold or Intelligent ray sampling.");
        }
        if let Some(renderer) = &self.renderer {
            if let Some(remaining) = renderer.noise_remaining {
                if remaining == 0 { ui.small("Noise threshold reached"); }
                else if let Some(tiles) = &renderer.noise_tiles {
                    ui.small(format!("{remaining} active pixels in {} tiles", tiles.len()));
                }
            }
            if self.output.intelligent_sampling && self.still_sample_target() >= 3 {
                if let Some(status) = renderer.intelligent_sampling.status(self.output.viewport_denoise.is_some()) { ui.small(status); }
            }
            let denoiser = &renderer.viewport_denoise;
            if let Some((backend, samples)) = denoiser.active_snapshot() {
                ui.small(format!("Denoising {samples} spp on {}", backend.label()));
            }
            if let Some(frame) = &denoiser.presented {
                ui.small(format!("Denoised pass {}, shown at {}", frame.samples, frame.traced_samples));
                ui.small(format!("Readback {:.1} ms · Prepare {:.1} ms", frame.readback_ms, frame.preparation_ms))
                    .on_hover_text("Readback includes GPU queue wait and copying the snapshot to CPU memory.");
                ui.small(format!("OIDN {:.1} ms · Present {:.1} ms", frame.oidn_ms, frame.presentation_ms))
                    .on_hover_text("OIDN includes setup and internal transfers. Present is measured from OIDN completion to the window presentation call.");
            }
            if let Some(error) = &denoiser.error { ui.small(error); }
        }
    }

    fn lighting_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Directional light");
        ui.checkbox(&mut self.lighting.enabled, "Enabled");
        ui.add_enabled_ui(self.lighting.enabled, |ui| {
            ui.checkbox(&mut self.lighting.shadows, "Cast shadows");
            ui.horizontal(|ui| { ui.label("Light color"); color_edit_button(ui, &mut self.lighting.light_color); });
            for (angle, limit, label) in [
                (&mut self.lighting.azimuth, 180.0, "Azimuth"),
                (&mut self.lighting.elevation, 1.55_f32.to_degrees(), "Elevation"),
            ] {
                let mut degrees = angle.to_degrees();
                if precision_slider(ui, &mut degrees, -limit..=limit, label, false, true).changed() {
                    *angle = degrees.to_radians();
                }
            }
            precision_slider(ui, &mut self.lighting.intensity, 0.0..=4.0, "Intensity", false, false);
            ui.add_enabled_ui(self.lighting.shadows, |ui| {
                precision_slider(ui, &mut self.lighting.angular_radius, LIGHT_ANGULAR_RADIUS, "Angular radius (°)", false, false)
                    .on_hover_text("0° gives hard shadows. Larger angles soften the shadows as samples accumulate.");
            });
        });
        ui.separator(); ui.heading("Environment");
        let previous_source = self.lighting.environment;
        let mut source = if self.lighting.path_traced || previous_source == HDRI_ENVIRONMENT { previous_source } else { 0 };
        let shown_source = source;
        egui::ComboBox::from_label("Source")
            .selected_text(ENVIRONMENT_NAMES[source])
            .show_ui(ui, |ui| {
                for (index, name) in ENVIRONMENT_NAMES.into_iter().enumerate() {
                    if self.lighting.path_traced || index == 0 || index == HDRI_ENVIRONMENT {
                        ui.selectable_value(&mut source, index, name);
                    }
                }
            });
        if source != shown_source {
            self.lighting.environment = source;
            self.lighting.environment_strength = self.lighting.environment_strength.min(self.lighting.environment_intensity_max());
        }
        if source != 0 {
            precision_slider(ui, &mut self.lighting.environment_rotation, -180.0..=180.0, "Rotation (°)", false, false);
        }
        if source == HDRI_ENVIRONMENT {
            ui.add(egui::TextEdit::singleline(&mut self.hdri_input).hint_text("Path to a .hdr panorama"));
            ui.horizontal(|ui| {
                #[cfg(windows)]
                if ui.button("Browse…").clicked() {
                    match hdri::choose_file(self.window.as_deref()) {
                        Ok(Some(path)) => {
                            self.hdri_input = path.to_string_lossy().into_owned();
                            self.apply_hdri_path(&path);
                        }
                        Ok(None) => {}
                        Err(error) => self.hdri_status = error,
                    }
                }
                if ui.add_enabled(!self.hdri_input.trim().is_empty(), egui::Button::new("Load")).clicked() {
                    let path = std::path::PathBuf::from(self.hdri_input.trim().trim_matches('"'));
                    self.apply_hdri_path(&path);
                }
            });
            if !self.hdri_status.is_empty() { ui.small(&self.hdri_status); }
        }
        ui.horizontal(|ui| {
            ui.label(if self.lighting.environment == HDRI_ENVIRONMENT || (self.lighting.path_traced && self.lighting.environment != 0) { "Tint" } else { "Light color" });
            color_edit_button(ui, &mut self.lighting.environment_color);
        });
        let intensity_max = self.lighting.environment_intensity_max();
        precision_slider(ui, &mut self.lighting.environment_strength, 0.0..=intensity_max, "Intensity", false, false);
        let environment_supported = self.lighting.path_traced || self.lighting.environment == HDRI_ENVIRONMENT;
        if environment_supported {
            ui.checkbox(&mut self.lighting.show_environment, "Show environment as background");
        }
        let environment_visible = (self.lighting.path_traced || self.environment_texture().is_some())
            && self.lighting.show_environment && !self.section_active();
        ui.add_enabled_ui(!environment_visible, |ui| {
            ui.horizontal(|ui| {
                ui.label("Background color");
                color_edit_button(ui, &mut self.background_color);
            });
        }).response.on_disabled_hover_text("Turn off Show environment as background to use a solid color.");
        if !self.lighting.path_traced {
            ui.separator();
            ui.heading("Ambient occlusion");
            ui.checkbox(&mut self.ao, "Ray-traced AO")
                .on_hover_text("Applies to both the viewport and PNG renders.");
            ui.add_enabled_ui(self.ao, |ui| {
                precision_slider(ui, &mut self.ao_strength, AO_STRENGTH_RANGE, "Strength", false, false);
                precision_slider(ui, &mut self.ao_radius, 0.0001..=2.0, "Radius", true, false);
            });
        }
        ui.separator();
        ui.heading("Fog");
        precision_slider(ui, &mut self.effects.fog_density, 0.0..=20.0, "Density", false, false);
        precision_slider(ui, &mut self.effects.fog_start, 0.0..=30.0, "Start distance", true, false)
            .on_hover_text("Distance from the camera, in scene units. Hold Shift for fine adjustment.");
        ui.horizontal(|ui| {
            ui.label("Color");
            color_edit_button(ui, &mut self.effects.fog_color);
        });
    }

    fn load_hdri(&mut self, path: &std::path::Path) -> Result<(), String> {
        let renderer = self.renderer.as_ref().ok_or("The graphics device is still starting.")?;
        let panorama = hdri::Panorama::load(&renderer.device, &renderer.queue, path)?;
        self.set_hdri(Some(panorama));
        Ok(())
    }

    fn apply_hdri_path(&mut self, path: &std::path::Path) {
        if let Err(error) = self.load_hdri(path) {
            self.hdri_status = format!("HDRI load failed: {error}");
        }
    }

    fn set_hdri(&mut self, panorama: Option<hdri::Panorama>) {
        self.hdri_input = panorama.as_ref().map(|asset| asset.path.to_string_lossy().into_owned()).unwrap_or_default();
        self.hdri_status = panorama.as_ref().map(|asset| {
            format!("Loaded {}", asset.path.file_name().unwrap_or_default().to_string_lossy())
        }).unwrap_or_default();
        self.hdri = panorama;
    }

    fn environment_texture(&self) -> Option<&hdri::Environment> {
        if self.lighting.environment == HDRI_ENVIRONMENT {
            self.hdri.as_ref().map(|asset| &asset.texture)
        } else {
            None
        }
    }

    fn camera_ui(&mut self, ui: &mut egui::Ui) {
        ui.heading("Lens");
        let mut focal = self.effects.focal;
        precision_slider(ui, &mut focal, 8.0..=120.0, "Focal length (mm)", false, false);
        if focal != self.effects.focal {
            self.effects.set_focal(focal);
        }
        ui.separator();
        let section = self.section_active();
        ui.add_enabled_ui(!section, |ui| {
            if ui.checkbox(&mut self.effects.dof_enabled, "Depth of field").changed() {
                if !self.effects.dof_enabled {
                    self.picking_focus = false;
                } else if self.effects.aperture == 0.0 {
                    self.effects.set_f_stop(self.effects.f_stop());
                }
            }
            ui.add_enabled_ui(self.effects.dof_enabled, |ui| {
                let mut f_stop = self.effects.f_stop();
                precision_slider(ui, &mut f_stop, self.effects.f_stop_range(), "Aperture (f/)", true, false)
                    .on_hover_text("Lower f-numbers give more blur. Hold Shift for fine adjustment.");
                if f_stop != self.effects.f_stop() {
                    self.effects.set_f_stop(f_stop);
                }
                precision_slider(ui, &mut self.effects.focus, 0.0001..=30.0, "Focus distance", true, false);
                let blades = self.effects.aperture_blades;
                egui::ComboBox::from_label("Aperture blades")
                    .selected_text(if blades == 0 { "Circular".into() } else { blades.to_string() })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.effects.aperture_blades, 0, "Circular");
                        for blades in 3..=16 {
                            ui.selectable_value(&mut self.effects.aperture_blades, blades, blades.to_string());
                        }
                    });
                if self.effects.aperture_blades >= 3 {
                    precision_slider(ui, &mut self.effects.aperture_rotation, -180.0..=180.0, "Aperture rotation", false, true);
                }
                if ui.button(if self.picking_focus { "Cancel focus pick" } else { "Pick focus" }).clicked() {
                    self.picking_focus = !self.picking_focus;
                }
                if self.picking_focus { ui.small("Click a surface inside the render frame."); }
                else if !self.focus_status.is_empty() { ui.small(&self.focus_status); }
            });
        });
        ui.separator(); ui.heading("Navigation");
        let mut roll_degrees = self.camera.roll.to_degrees();
        precision_slider(ui, &mut roll_degrees, -180.0..=180.0, "Camera roll", false, true);
        if roll_degrees != self.camera.roll.to_degrees() { self.camera.set_roll(roll_degrees.to_radians()); }
        if ui.button("Reset rotation").clicked() { self.camera.yaw = 0.0; self.camera.pitch = 0.0; self.camera.set_roll(0.0); }
        precision_slider(ui, &mut self.camera.speed, 0.001..=100.0, "Speed", true, false);
        if ui.button("Reset framing").on_hover_text("Frame the selected fractal from its starting camera position.").clicked() {
            self.reset_framing();
        }
        ui.small("WASD: move · E / Q: up / down\nRight mouse: look · Wheel: speed");
    }

    fn preset_action(&mut self, action: preset_gallery::Action) {
        use preset_gallery::Action;
        match action {
            Action::ResetDefaults => self.reset_to_defaults(),
            Action::Add => {
                match self.presets.add(&self.preset_text()) {
                    Ok(id) => {
                        self.capture_slot = Some(id);
                        self.preset_status = "Preset added. Capturing preview…".into();
                    }
                    Err(error) => self.preset_status = format!("Add failed: {error}"),
                }
            }
            Action::Load(id) => self.load_preset(&self.presets.file(id, "txt")),
            Action::Update(id) => {
                if self.save_preset(&self.presets.file(id, "txt")) {
                    self.capture_slot = Some(id);
                }
            }
            Action::Remove(id) => {
                self.preset_status = match self.presets.remove(id) {
                    Ok(()) => "Preset removed".into(),
                    Err(error) => format!("Remove failed: {error}"),
                };
            }
            Action::Rename(id, name) => {
                self.preset_status = match self.presets.rename(id, name) {
                    Ok(()) => "Preset renamed".into(),
                    Err(error) => format!("Rename failed: {error}"),
                };
            }
        }
    }

    fn reset_to_defaults(&mut self) {
        self.observe_history(false);
        let mut defaults = Self::default();
        if self.render_job.is_some() {
            defaults.output.progress = self.output.progress;
            defaults.output.status = std::mem::take(&mut self.output.status);
        }

        // Preserve the live app, saved presets, and Undo while restoring the working scene.
        *self = Self {
            about: std::mem::take(&mut self.about),
            startup: std::mem::take(&mut self.startup),
            event_proxy: self.event_proxy.take(),
            window: self.window.take(),
            renderer: self.renderer.take(),
            egui_ctx: self.egui_ctx.clone(),
            egui_state: self.egui_state.take(),
            presets: std::mem::take(&mut self.presets),
            controls_tab: ControlsTab::Presets,
            preset_status: "Defaults restored.".into(),
            render_job: self.render_job.take(),
            history: std::mem::take(&mut self.history),
            modifiers: self.modifiers,
            ..defaults
        };
        self.observe_history(false);
        if let Some(window) = &self.window {
            window.set_title(&self.fractal_title());
            window.request_redraw();
        }
    }

    fn update_render_frame(&mut self, ctx: &egui::Context) {
        let available = ctx.available_rect();
        let aspect = self.output.width as f32 / self.output.height as f32;
        let frame = camera_effects::render_frame(available, aspect);
        self.frame = Some(frame);
        let painter = ctx.layer_painter(egui::LayerId::background());
        if self.show_frame || self.picking_focus {
            let shade = egui::Color32::from_black_alpha(160);
            for (min, max) in [
                (available.min, egui::pos2(available.max.x, frame.min.y)),
                (egui::pos2(available.min.x, frame.max.y), available.max),
                (egui::pos2(available.min.x, frame.min.y), egui::pos2(frame.min.x, frame.max.y)),
                (egui::pos2(frame.max.x, frame.min.y), egui::pos2(available.max.x, frame.max.y)),
            ] {
                let matte = egui::Rect::from_min_max(min, max);
                if matte.is_positive() { painter.rect_filled(matte, 0.0, shade); }
            }
            painter.with_clip_rect(available.shrink2(egui::vec2(2.0, 0.0)))
                .rect_stroke(frame, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(180)), egui::StrokeKind::Inside);
        }
        if self.section_active() { self.picking_focus = false; }
        if self.picking_focus && !ctx.is_pointer_over_area() {
            if let Some(point) = ctx.input(|input| input.pointer.interact_pos()).filter(|p| frame.contains(*p)) {
                ctx.set_cursor_icon(egui::CursorIcon::Crosshair);
                if ctx.input(|input| input.pointer.primary_clicked()) {
                    self.focus_point = Some([
                        2.0 * (point.x - frame.left()) / frame.width() - 1.0,
                        1.0 - 2.0 * (point.y - frame.top()) / frame.height(),
                    ]);
                    self.picking_focus = false;
                }
            }
        }
    }

    fn camera_uniforms(&self, size: [f32; 2], progress: Option<u32>, frame: [f32; 4], export: bool) -> [f32; CAMERA_UNIFORM_FLOATS] {
        let c = &self.camera;
        let a = &self.lighting;
        let [r, g, b, roughness, metallic] = match self.material {
            STANDARD_MATERIAL => [a.color[0], a.color[1], a.color[2], a.roughness, a.metallic],
            MATCAP_MATERIAL => [0.0, 0.0, 0.0, self.matcap.roughness, self.matcap.metallic],
            _ => {
                let [r, g, b, ..] = self.material_settings;
                [r, g, b, 1.0, 0.0]
            }
        };
        let matcap_style = if self.material == MATCAP_MATERIAL { (self.matcap.style + 1) as f32 } else { 0.0 };
        let refined = export || progress.is_some();
        let quality = if refined { (self.quality * 1.25).min(4.0).max(self.quality) } else { self.quality };
        let base = [
            c.position[0], c.position[1], c.position[2], size[0] / size[1].max(1.0),
            c.yaw, c.pitch, flag(self.section_active()), self.slice.unwrap_or(0.0),
            self.power, self.fractal.value(), size[1], flag(self.diagnostic && !export),
            matcap_style, flag(self.ao), self.ao_strength, self.ao_radius,
            self.exposure, self.normal_smoothing, quality, flag(self.has_directional_shadows()),
            self.distant_filter, c.view_roll(), progress.unwrap_or(0) as f32, flag(refined),
            r, g, b, roughness,
            a.light_color[0], a.light_color[1], a.light_color[2], metallic,
            a.azimuth, a.elevation, if a.enabled { a.intensity } else { 0.0 }, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut uniforms = [0.0; CAMERA_UNIFORM_FLOATS];
        uniforms[..40].copy_from_slice(&base);
        uniforms[40..51].copy_from_slice(&self.effects.uniforms(frame));
        uniforms[51] = if self.diagnostic && !export { 1.0 } else { self.contrast };
        uniforms[52] = self.matcap.rotation.to_radians();
        uniforms[56..60].copy_from_slice(&self.mandelbox_parameters.uniforms());
        uniforms[60..63].copy_from_slice(&self.mandelbulb_parameters.values());
        uniforms[64..80].copy_from_slice(&self.menger_parameters.uniforms());
        uniforms[80..83].copy_from_slice(&self.background_color);
        uniforms[84..96].copy_from_slice(&self.fold_parameters.uniforms());
        let mut base_coefficient = 1.0;
        let mut count = 0;
        let mut fields = uniforms[100..LIGHTING_UNIFORM_OFFSET].chunks_exact_mut(EXTRA_MUTATION_UNIFORMS);
        let active = std::iter::once(&self.mutation).chain(&self.extra_mutations)
            .filter_map(|stage| stage.selected(self.fractal).map(|formula| (stage, formula)))
            .filter(|(stage, _)| stage.weight > 0.0);
        let mut packed = [[0.0; EXTRA_MUTATION_UNIFORMS]; MAX_MUTATIONS];
        for ((stage, formula), field) in active.zip(&mut packed) {
            *field = stage.field_uniforms(formula, stage.weight);
            count += 1;
        }
        let active_fields = &mut packed[..count];
        // Keep floating-point accumulation stable when the same formulas change order.
        active_fields.sort_unstable_by(|a, b| {
            a[1].total_cmp(&b[1]).then_with(|| {
                a.iter().zip(b).map(|(left, right)| left.total_cmp(right))
                    .find(|order| !order.is_eq()).unwrap_or(std::cmp::Ordering::Equal)
            })
        });
        let total = f64::from(self.base_weight)
            + active_fields.iter().map(|field| f64::from(field[0])).sum::<f64>();
        if total > 0.0 {
            base_coefficient = (f64::from(self.base_weight) / total) as f32;
            for (field, output) in active_fields.iter_mut().zip(&mut fields) {
                field[0] = (f64::from(field[0]) / total) as f32;
                output.copy_from_slice(field);
            }
        }
        uniforms[96..98].copy_from_slice(&[base_coefficient, count as f32]);
        let environment = a.environment_color.map(|channel| channel * a.environment_strength);
        // Store the angular radius as the cone's span in cosine space.
        let half_angle_sine = (a.angular_radius.to_radians() * 0.5).sin();
        uniforms[LIGHTING_UNIFORM_OFFSET..PATH_UNIFORM_OFFSET].copy_from_slice(&[
            environment[0], environment[1], environment[2], 2.0 * half_angle_sine.powi(2),
        ]);
        let environment_style = if a.environment == HDRI_ENVIRONMENT && self.hdri.is_none() { 0 } else { a.environment };
        uniforms[PATH_UNIFORM_OFFSET..PATH_UNIFORM_OFFSET + 6].copy_from_slice(&[
            flag(a.path_traced), environment_style as f32, a.environment_rotation.to_radians(),
            a.bounces as f32, flag(a.show_environment), 0.0,
        ]);
        if self.material == DIFFUSE_GRAY_MATERIAL {
            // IOR 1 with zero specular, coat and emission gives a Lambertian surface.
            uniforms[SURFACE_UNIFORM_OFFSET..SIERPINSKI_UNIFORM_OFFSET].copy_from_slice(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        } else {
            let finish = if self.material == MATCAP_MATERIAL { &self.matcap.finish } else { &self.custom_finish };
            let [r, g, b] = finish.emission_color.map(|channel| channel * finish.emission_strength);
            uniforms[SURFACE_UNIFORM_OFFSET..SIERPINSKI_UNIFORM_OFFSET].copy_from_slice(&[
                finish.ior, finish.specular, finish.coat, finish.coat_roughness, r, g, b, 0.0,
            ]);
        }
        uniforms[SIERPINSKI_UNIFORM_OFFSET] = self.sierpinski.depth as f32;
        uniforms[ORBIT_DEPTH_UNIFORM_OFFSET..FOUR_BRANCH_UNIFORM_OFFSET].copy_from_slice(&self.orbit_depths.offsets());
        uniforms[FOUR_BRANCH_UNIFORM_OFFSET..FOUR_BRANCH_UNIFORM_OFFSET + 5].copy_from_slice(&self.four_branch.uniforms());
        let [blades, rotation] = self.effects.bokeh();
        uniforms[BOKEH_UNIFORM_OFFSET..BOKEH_UNIFORM_OFFSET + 2].copy_from_slice(&[blades, rotation.to_radians()]);
        uniforms[PIXEL_SAMPLING_UNIFORM] = self.pixel_sampling.uniform();
        uniforms
    }

    fn preset_values(&self) -> [f32; PRESET_VALUE_COUNT] {
        let c = &self.camera;
        let light = &self.lighting;
        let mut values = [0.0; PRESET_VALUE_COUNT];
        values[..35].copy_from_slice(&[
            c.position[0], c.position[1], c.position[2], c.yaw, c.pitch, c.speed,
            flag(self.slice.is_some()), self.slice.unwrap_or(0.0), self.power, self.fractal.value(),
            flag(self.diagnostic), self.material as f32, flag(self.ao), self.ao_strength, self.ao_radius,
            // Slots 18 and 20 retain the old AA settings for preset compatibility.
            self.exposure, self.normal_smoothing, self.quality, 0.0, self.distant_filter,
            1.0, light.color[0], light.color[1], light.color[2], light.roughness,
            light.metallic, light.light_color[0], light.light_color[1], light.light_color[2],
            light.azimuth, light.elevation, light.intensity, light.environment_strength, c.roll, self.still_samples as f32,
        ]);
        // Keep the former DOF contrast slot so existing preset fields retain their positions.
        values[35..44].copy_from_slice(&self.effects.preset());
        // Slots 44 and 45 retain the layout for legacy DOF contrast and matcap rotation.
        values[44..48].copy_from_slice(&[1.0, 0.0, c.look_roll, self.contrast]);
        values[48..52].copy_from_slice(&self.mandelbox_parameters.values());
        values[52..55].copy_from_slice(&self.mandelbulb_parameters.values());
        values[55..60].copy_from_slice(&self.menger_parameters.values());
        values[60..63].copy_from_slice(&self.background_color);
        values[63..79].copy_from_slice(&self.mutation.values());
        let base_folds = self.fold_parameters.values();
        let mutation_folds = self.mutation.fold_parameters.values();
        values[79..87].copy_from_slice(&base_folds[..8]);
        values[87..95].copy_from_slice(&mutation_folds[..8]);
        values[95..99].copy_from_slice(&base_folds[8..]);
        values[99..103].copy_from_slice(&mutation_folds[8..]);
        values[BASE_PRESET_FIELDS] = self.extra_mutations.len() as f32;
        for (stage, output) in self.extra_mutations.iter()
            .zip(values[BASE_PRESET_FIELDS + 1..].chunks_exact_mut(MUTATION_PRESET_FIELDS))
        {
            output.copy_from_slice(&stage.stacked_preset_values());
        }
        values[CASCADE_PRESET_FIELDS] = 1.0; // Former blend mode, retained for preset compatibility.
        values[CASCADE_PRESET_FIELDS + 1] = self.base_weight;
        values[PRESET_LIGHT_FLAGS..PRESET_ENVIRONMENT_FIELDS].copy_from_slice(&[flag(light.enabled), flag(light.shadows)]);
        values[PRESET_ENVIRONMENT_FIELDS..PRESET_ENVIRONMENT_FIELDS + 3].copy_from_slice(&light.environment_color);
        values[PRESET_ENVIRONMENT_FIELDS + 3] = light.angular_radius;
        values[PRESET_PATH_FIELDS..PRESET_MOVING_SAMPLES].copy_from_slice(&light.path_preset());
        values[PRESET_MOVING_SAMPLES] = self.moving_samples as f32;
        values[PRESET_MOVING_BOUNCES] = self.moving_bounces as f32;
        values[PRESET_SURFACE_FIELDS..PRESET_PRINCIPLED_FIELDS].copy_from_slice(&self.material_settings);
        values[PRESET_PRINCIPLED_FIELDS..PRESET_PRINCIPLED_FIELDS + 8].copy_from_slice(&self.material_finish.values());
        values[PRESET_PRINCIPLED_FIELDS + 8..PRESET_MATCAP_FIELDS].copy_from_slice(&self.custom_finish.values());
        values[PRESET_MATCAP_FIELDS..PRESET_SIERPINSKI_FIELDS].copy_from_slice(&self.matcap.values());
        values[PRESET_SIERPINSKI_FIELDS] = self.sierpinski.depth as f32;
        values[PRESET_ORBIT_DEPTH_FIELDS..PRESET_ORBIT_DEPTH_FIELDS + 4].copy_from_slice(&self.orbit_depths.values());
        values[PRESET_FOUR_BRANCH_FIELDS..PRESET_FOUR_BRANCH_FIELDS + 4].copy_from_slice(&self.four_branch.values());
        for (index, stage) in std::iter::once(&self.mutation).chain(&self.extra_mutations).enumerate()
        {
            values[PRESET_SIERPINSKI_FIELDS + 1 + index] = stage.sierpinski.depth as f32;
            let offset = PRESET_ORBIT_DEPTH_FIELDS + 4 * (index + 1);
            values[offset..offset + 4].copy_from_slice(&stage.orbit_depths.values());
            let offset = PRESET_FOUR_BRANCH_FIELDS + 4 * (index + 1);
            values[offset..offset + 4].copy_from_slice(&stage.four_branch.values());
        }
        values[PRESET_BOKEH_FIELDS..].copy_from_slice(&self.effects.bokeh());
        values
    }

    fn preset_text(&self) -> String {
        let values = self.preset_values().into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n");
        match &self.hdri {
            Some(asset) => format!("HDRI={}\n{values}", asset.path.display()),
            None => values,
        }
    }
    fn save_preset(&mut self, path: impl AsRef<std::path::Path>) -> bool {
        match std::fs::write(path, self.preset_text()) { Ok(()) => { self.preset_status = "Preset saved; capturing preview".into(); true }, Err(e) => { self.preset_status = format!("Save failed: {e}"); false } }
    }
    fn load_preset(&mut self, path: impl AsRef<std::path::Path>) {
        let result = std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|text| {
            let mut lines = text.lines();
            let hdri = lines.clone().next().and_then(|line| line.strip_prefix("HDRI=")).map(std::path::PathBuf::from);
            if hdri.is_some() { lines.next(); }
            let mut v: Vec<f32> = lines.map(str::parse).collect::<Result<_, _>>().map_err(|_| "Invalid preset values".to_string())?;
            // A previous version wrote the sample count twice, including once
            // before lighting. Accept those files without changing them.
            if v.len() == 36 && v[21] == v[35] { v.remove(21); }
            let material = v.get(11).copied().unwrap_or(-1.0);
            let valid_material = (0.0..LEGACY_MATERIAL_DEFAULTS.len() as f32).contains(&material)
                || (v.len() >= PRESET_SIERPINSKI_FIELDS && material == MATCAP_MATERIAL as f32);
            if !matches!(v.len(), 17..=21 | 33..=35 | 43 | 45..=48 | 52 | 55 | 58 | 60 | 63 | 74 | 79 | 95 | BASE_PRESET_FIELDS | CASCADE_PRESET_FIELDS | PRESET_LIGHT_FLAGS | PRESET_ENVIRONMENT_FIELDS | PRESET_PATH_FIELDS | PRESET_MOVING_SAMPLES | PRESET_MOVING_BOUNCES | PRESET_SURFACE_FIELDS | PRESET_PRINCIPLED_FIELDS | PRESET_MATCAP_FIELDS | PRESET_SIERPINSKI_FIELDS | PRESET_FOUR_BRANCH_FIELDS | PRESET_BOKEH_FIELDS | PRESET_VALUE_COUNT)
                || v.iter().any(|x| !x.is_finite())
                || !valid_material
                || material.fract() != 0.0
            {
                return Err("Invalid preset".into());
            }
            Ok((v, hdri))
        });
        match result {
            Ok((v, hdri)) => {
                self.restore_scene(&v);
                self.preset_status = "Preset loaded".into();
                if let Some(path) = hdri {
                    if !self.hdri.as_ref().is_some_and(|asset| asset.path == path) {
                        self.set_hdri(None);
                        if let Err(error) = self.load_hdri(&path) {
                            self.hdri_input = path.to_string_lossy().into_owned();
                            self.hdri_status = format!("HDRI load failed: {error}");
                            self.preset_status = "Preset loaded with a uniform environment. HDRI unavailable.".into();
                        }
                    }
                } else {
                    self.set_hdri(None);
                }
            }
            Err(e) => self.preset_status = format!("Load failed: {e}"),
        }
    }

    fn restore_scene(&mut self, v: &[f32]) {
        self.camera.position = [v[0], v[1], v[2]]; self.camera.velocity = [0.0; 3]; self.camera.yaw = v[3]; self.camera.speed = v[5].clamp(0.001, 100.0);
        self.slice = (v[6] != 0.0).then_some(v[7].clamp(-2.0, 2.0)); self.power = v[8].clamp(2.0, 23.0); self.fractal = match (v.len(), Fractal::from_value(v[9])) {
            (58.., Some(Fractal::Menger)) => Fractal::Menger,
            (95.., Some(formula @ (Fractal::Kifs | Fractal::AmazingSurface))) => formula,
            (BASE_PRESET_FIELDS.., Some(Fractal::PseudoKleinian)) => Fractal::PseudoKleinian,
            (PRESET_FOUR_BRANCH_FIELDS.., Some(Fractal::Sierpinski)) => Fractal::Sierpinski,
            (PRESET_BOKEH_FIELDS.., Some(Fractal::FourBranch)) => Fractal::FourBranch,
            (_, Some(Fractal::Mandelbulb)) => Fractal::Mandelbulb,
            _ => Fractal::Mandelbox,
        }; self.diagnostic = v[10] != 0.0; self.material = (v[11] as u32).min(MATCAP_MATERIAL); self.ao = v[12] != 0.0; self.ao_strength = v[13].clamp(*AO_STRENGTH_RANGE.start(), *AO_STRENGTH_RANGE.end()); self.ao_radius = v[14].clamp(0.0001, 2.0); self.exposure = v[15].clamp(-5.0, 5.0); self.normal_smoothing = v[16].clamp(*SMOOTHNESS.start(), *SMOOTHNESS.end()); self.quality = v.get(17).copied().unwrap_or(1.0).clamp(*DETAIL_QUALITY.start(), *DETAIL_QUALITY.end()); self.distant_filter = v.get(19).copied().unwrap_or(2.0).clamp(0.5, 4.0);
        if v.len() >= 33 {
            self.lighting = Lighting { color: [v[21].clamp(0.0,1.0), v[22].clamp(0.0,1.0), v[23].clamp(0.0,1.0)], roughness: v[24].clamp(0.05,1.0), metallic: v[25].clamp(0.0,1.0), light_color: [v[26].clamp(0.0,1.0),v[27].clamp(0.0,1.0),v[28].clamp(0.0,1.0)], azimuth: v[29].clamp(-std::f32::consts::PI,std::f32::consts::PI), elevation: v[30].clamp(-1.55,1.55), intensity: v[31].clamp(0.0,4.0), ..Lighting::default() };
        } else {
            self.lighting = Lighting::default();
        }
        self.lighting.enabled = v.get(PRESET_LIGHT_FLAGS).copied().unwrap_or(1.0) != 0.0;
        self.lighting.shadows = v.get(PRESET_LIGHT_FLAGS + 1).copied().unwrap_or(0.0) != 0.0;
        if let Some([r, g, b, radius]) = v.get(PRESET_ENVIRONMENT_FIELDS..PRESET_PATH_FIELDS) {
            self.lighting.environment_color = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
            self.lighting.angular_radius = radius.clamp(*LIGHT_ANGULAR_RADIUS.start(), *LIGHT_ANGULAR_RADIUS.end());
        }
        if let Some([enabled, environment, rotation, bounces, visible, diffuse]) = v.get(PRESET_PATH_FIELDS..PRESET_MOVING_SAMPLES) {
            self.lighting.path_traced = *enabled != 0.0;
            self.lighting.environment = (*environment as usize).min(ENVIRONMENT_NAMES.len() - 1);
            self.lighting.environment_rotation = rotation.clamp(-180.0, 180.0);
            self.lighting.bounces = (*bounces as u32).clamp(1, 8);
            self.lighting.show_environment = *visible != 0.0;
            if self.lighting.path_traced && v.len() < PRESET_PRINCIPLED_FIELDS {
                self.material = if *diffuse != 0.0 { DIFFUSE_GRAY_MATERIAL } else { STANDARD_MATERIAL };
            }
        }
        if let Some(&strength) = v.get(32) {
            self.lighting.environment_strength = strength.clamp(0.0, self.lighting.environment_intensity_max());
        }
        self.material_settings = LEGACY_MATERIAL_DEFAULTS.get(self.material as usize).copied().unwrap_or(DIFFUSE_DEFAULTS);
        if let Some([r, g, b, roughness, metallic]) = v.get(PRESET_SURFACE_FIELDS..PRESET_PRINCIPLED_FIELDS) {
            self.material_settings = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0),
                roughness.clamp(0.05, 1.0), metallic.clamp(0.0, 1.0)];
        }
        self.material_finish = SurfaceFinish::from_values(v.get(PRESET_PRINCIPLED_FIELDS..PRESET_PRINCIPLED_FIELDS + 8).unwrap_or_default());
        self.custom_finish = SurfaceFinish::from_values(v.get(PRESET_PRINCIPLED_FIELDS + 8..PRESET_MATCAP_FIELDS).unwrap_or_default());
        self.matcap = MatcapMaterial::from_values(v.get(PRESET_MATCAP_FIELDS..PRESET_SIERPINSKI_FIELDS).unwrap_or_default());
        if !MATERIALS.iter().any(|(id, _)| *id == self.material) {
            let [r, g, b, roughness, metallic] = self.material_settings;
            self.lighting.color = [r, g, b];
            self.lighting.roughness = roughness;
            self.lighting.metallic = metallic;
            self.custom_finish = std::mem::take(&mut self.material_finish);
            self.select_material(STANDARD_MATERIAL);
        }
        self.moving_samples = match v.get(PRESET_MOVING_SAMPLES).copied() {
            Some(2.0) => 2,
            Some(4.0) => 4,
            _ => 1,
        };
        self.moving_bounces = (v.get(PRESET_MOVING_BOUNCES).copied().unwrap_or(1.0) as u32).clamp(1, 8);
        self.camera.roll = v.get(33).copied().unwrap_or(0.0).clamp(-std::f32::consts::PI, std::f32::consts::PI);
        let pitch_limit = if v.len() >= 47 && self.camera.roll != 0.0 { std::f32::consts::FRAC_PI_2 } else { 1.55 };
        self.camera.pitch = v[4].clamp(-pitch_limit, pitch_limit);
        self.camera.look_roll = v.get(46).copied().unwrap_or(0.0)
            .clamp(-std::f32::consts::TAU, std::f32::consts::TAU);
        if self.camera.roll == 0.0 { self.camera.look_roll = 0.0; }
        // Reuse the old stationary sample count. A disabled legacy toggle used the default in path-traced mode.
        let samples = v.get(34).copied().unwrap_or(DEFAULT_VIEWPORT_SAMPLES as f32) as u32;
        self.still_samples = if v.get(20) != Some(&0.0) && (1..=noise_sampling::MAX_SAMPLES).contains(&samples) {
            samples
        } else {
            DEFAULT_VIEWPORT_SAMPLES
        };
        self.effects = CameraEffects::from_preset(v.get(35..v.len().min(44)).unwrap_or_default());
        self.effects.restore_bokeh(v.get(PRESET_BOKEH_FIELDS..).unwrap_or_default());
        let legacy_contrast = if self.effects.dof_enabled && self.effects.aperture > 0.0 {
            v.get(44).copied().unwrap_or(1.12)
        } else {
            1.0
        };
        self.contrast = v.get(47).copied().unwrap_or(legacy_contrast)
            .clamp(*CONTRAST_RANGE.start(), *CONTRAST_RANGE.end());
        self.mandelbox_parameters = MandelboxParameters::from_preset(v.get(48..52).unwrap_or_default());
        self.mandelbulb_parameters = MandelbulbParameters::from_preset(v.get(52..55).unwrap_or_default());
        self.menger_parameters = MengerParameters::from_preset(v.get(55..v.len().min(60)).unwrap_or_default());
        self.background_color = match v.get(60..63) {
            Some([r, g, b]) => [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
            _ => DEFAULT_BACKGROUND,
        };
        self.mutation = MutationStage::from_preset(v.get(63..v.len().min(79)).unwrap_or_default());
        self.sierpinski = SierpinskiParameters::from_preset(v.get(PRESET_SIERPINSKI_FIELDS));
        self.mutation.sierpinski = SierpinskiParameters::from_preset(v.get(PRESET_SIERPINSKI_FIELDS + 1));
        self.orbit_depths = OrbitDepths::from_preset(v.get(PRESET_ORBIT_DEPTH_FIELDS..PRESET_ORBIT_DEPTH_FIELDS + 4));
        self.mutation.orbit_depths = OrbitDepths::from_preset(v.get(PRESET_ORBIT_DEPTH_FIELDS + 4..PRESET_ORBIT_DEPTH_FIELDS + 8));
        self.four_branch = FourBranchParameters::from_preset(v.get(PRESET_FOUR_BRANCH_FIELDS..PRESET_FOUR_BRANCH_FIELDS + 4));
        self.mutation.four_branch = FourBranchParameters::from_preset(v.get(PRESET_FOUR_BRANCH_FIELDS + 4..PRESET_FOUR_BRANCH_FIELDS + 8));
        self.fold_parameters = FoldParameters::from_preset(v.get(79..87).unwrap_or_default(), v.get(95..99).unwrap_or_default());
        self.mutation.fold_parameters = FoldParameters::from_preset(v.get(87..95).unwrap_or_default(), v.get(99..103).unwrap_or_default());
        let count = v.get(BASE_PRESET_FIELDS).copied().unwrap_or(0.0) as usize;
        self.extra_mutations = v.get(BASE_PRESET_FIELDS + 1..).unwrap_or_default()
            .chunks_exact(MUTATION_PRESET_FIELDS).take(count.min(MAX_MUTATIONS - 1)).enumerate().map(|(index, values)| {
                let mut stage = MutationStage::from_preset(&values[..16]);
                stage.fold_parameters = FoldParameters::from_preset(&values[16..24], &values[24..]);
                stage.sierpinski = SierpinskiParameters::from_preset(v.get(PRESET_SIERPINSKI_FIELDS + 2 + index));
                let offset = PRESET_ORBIT_DEPTH_FIELDS + 4 * (index + 2);
                stage.orbit_depths = OrbitDepths::from_preset(v.get(offset..offset + 4));
                let offset = PRESET_FOUR_BRANCH_FIELDS + 4 * (index + 2);
                stage.four_branch = FourBranchParameters::from_preset(v.get(offset..offset + 4));
                stage
            }).collect();
        self.base_weight = v.get(CASCADE_PRESET_FIELDS + 1).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        self.picking_focus = false;
        self.focus_status.clear();
    }
}
impl ApplicationHandler<startup::Event> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let icon = image::load_from_memory_with_format(
            include_bytes!("../assets/mengervibes.png"),
            image::ImageFormat::Png,
        )
        .expect("Decode the embedded application icon")
        .into_rgba8();
        let (width, height) = icon.dimensions();
        let icon = winit::window::Icon::from_rgba(icon.into_raw(), width, height)
            .expect("Create the application icon");
        let attributes = Window::default_attributes()
            .with_title(format!("{APP_NAME} - Starting graphics"))
            .with_inner_size(winit::dpi::LogicalSize::new(960, 960))
            .with_window_icon(Some(icon.clone()));
        #[cfg(target_os = "windows")]
        let attributes = {
            use winit::platform::windows::WindowAttributesExtWindows;
            attributes.with_taskbar_icon(Some(icon))
        };
        let window = Arc::new(event_loop.create_window(attributes).expect("Window"));
        window.request_redraw();
        self.window = Some(window);
    }
    fn user_event(&mut self, _: &ActiveEventLoop, event: startup::Event) {
        let Some(window) = self.window.as_ref().cloned() else { return; };
        match event {
            startup::Event::Stage(stage) => self.startup.set_stage(stage),
            startup::Event::UpdateChecked(result) => self.about.finish(result),
            startup::Event::CreateSurface { instance, reply } => startup::create_surface(window.clone(), instance, reply),
            startup::Event::Finished(Err(error)) => self.startup.fail(error),
            startup::Event::Finished(Ok(renderer)) => {
                let mut renderer = *renderer;
                let size = window.inner_size();
                if size.width != renderer.config.width || size.height != renderer.config.height {
                    renderer.resize(size.width, size.height);
                }
                self.renderer = Some(renderer);
                self.egui_state = Some(egui_winit::State::new(
                    self.egui_ctx.clone(), egui::ViewportId::ROOT, &*window,
                    Some(window.scale_factor() as f32), window.theme(), None,
                ));
                if let Err(error) = self.presets.scan() {
                    self.preset_status = format!("Could not read presets: {error}");
                }
                self.last_frame = Instant::now();
                self.observe_history(false);
                window.set_title(&self.fractal_title());
                println!("Startup: viewer ready");
                println!("WASD: fly | E/Q: up/down | right mouse: look | wheel: speed | R: reset | C: bulb section | Up/Down: section | Page Up/Down: bulb power");
            }
        }
        window.request_redraw();
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        if self.renderer.is_none() {
            let Some(window) = self.window.as_ref() else { return; };
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => window.request_redraw(),
                WindowEvent::RedrawRequested => {
                    self.startup.paint(window);
                    if self.startup.started.is_none() {
                        self.startup.start(window.inner_size(), self.event_proxy.as_ref().expect("Event proxy").clone());
                        if self.startup.next_repaint.is_none() { window.request_redraw(); }
                    }
                }
                _ => {}
            }
            return;
        }
        if let WindowEvent::ModifiersChanged(modifiers) = &event {
            self.modifiers = modifiers.state();
        }
        let mut text_history_shortcut = false;
        if let WindowEvent::KeyboardInput { event: key, is_synthetic: false, .. } = &event {
            let modifiers = self.modifiers;
            if !self.about.open && modifiers.control_key() && !modifiers.alt_key() && !modifiers.super_key()
                && matches!(key.physical_key, PhysicalKey::Code(KeyCode::KeyZ | KeyCode::KeyY))
            {
                if !text_edit_is_focused(&self.egui_ctx) {
                    if key.state == ElementState::Pressed && !key.repeat {
                        self.history_actions.push(if modifiers.shift_key()
                            || key.physical_key == PhysicalKey::Code(KeyCode::KeyY)
                        {
                            history::Action::Redo
                        } else {
                            history::Action::Undo
                        });
                        if let Some(window) = &self.window { window.request_redraw(); }
                    }
                    return;
                }
                text_history_shortcut = true;
            }
        }
        if matches!(event, WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. })
            && self.egui_ctx.is_pointer_over_area() && !self.looking
            && !self.keys.iter().any(|key| matches!(key, KeyCode::KeyW | KeyCode::KeyA | KeyCode::KeyS | KeyCode::KeyD | KeyCode::KeyE | KeyCode::KeyQ))
        {
            // End camera coasting when an edit in the panel begins.
            self.camera.velocity = [0.0; 3];
            self.observe_history(false);
        }
        if let (Some(window), Some(state)) = (&self.window, &mut self.egui_state) {
            let response = state.on_window_event(window, &event);
            // egui reports that RedrawRequested itself needs repainting; asking
            // for another redraw here would make the viewport render forever.
            if response.repaint && !matches!(event, WindowEvent::RedrawRequested) { window.request_redraw(); }
            if text_history_shortcut { return; }
            if (response.consumed || self.about.open) && matches!(event, WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. }) {
                if let WindowEvent::KeyboardInput { ref event, .. } = event { if event.state == ElementState::Released { if let PhysicalKey::Code(key) = event.physical_key { self.keys.remove(&key); } } }
                if matches!(event, WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Right, .. }) { self.looking = false; }
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                    self.last_scene.clear();
                }
                if let Some(window) = &self.window { window.request_redraw(); }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys.insert(key);
                        if key == KeyCode::KeyR && !event.repeat { self.reset_framing(); }
                        if key == KeyCode::KeyG && !event.repeat { self.diagnostic = !self.diagnostic; println!("Flat diagnostic: {}", if self.diagnostic { "ON" } else { "OFF" }); }
                        if key == KeyCode::KeyX { self.exposure = (self.exposure + 0.25).min(5.0); }
                        if key == KeyCode::KeyZ { self.exposure = (self.exposure - 0.25).max(-5.0); }
                        if key == KeyCode::KeyN { self.normal_smoothing = (self.normal_smoothing * 1.25).clamp(SMOOTHNESS_KEY_FLOOR, *SMOOTHNESS.end()); }
                        if key == KeyCode::KeyB {
                            self.normal_smoothing = if self.normal_smoothing <= SMOOTHNESS_KEY_FLOOR { 0.0 }
                                else { (self.normal_smoothing / 1.25).max(SMOOTHNESS_KEY_FLOOR) };
                        }
                        if key == KeyCode::KeyO && !event.repeat { self.ao = !self.ao; println!("Ambient occlusion: {}", if self.ao { "ON" } else { "OFF" }); }
                        if key == KeyCode::KeyI { self.ao_strength = (self.ao_strength + 0.1).min(*AO_STRENGTH_RANGE.end()); self.ao = true; }
                        if key == KeyCode::KeyK { self.ao_strength = (self.ao_strength - 0.1).max(*AO_STRENGTH_RANGE.start()); self.ao = true; }
                        if key == KeyCode::KeyU { self.ao_radius = (self.ao_radius * 1.25).min(2.0); self.ao = true; }
                        if key == KeyCode::KeyJ { self.ao_radius = (self.ao_radius / 1.25).max(0.0001); self.ao = true; }
                        if matches!(key, KeyCode::KeyI | KeyCode::KeyK | KeyCode::KeyU | KeyCode::KeyJ) {
                            println!("AO strength: {:.1}, radius: {:.4}", self.ao_strength, self.ao_radius);
                        }
                        if key == KeyCode::KeyM && !event.repeat {
                            let index = MATERIALS.iter().position(|(id, _)| *id == self.material).unwrap_or(0);
                            self.select_material(MATERIALS[(index + 1) % MATERIALS.len()].0);
                            println!("Material: {}", self.material_name());
                        }
                        if key == KeyCode::PageUp { self.power = (self.power + 1.0).min(23.0); }
                        if key == KeyCode::PageDown { self.power = (self.power - 1.0).max(2.0); }
                        if key == KeyCode::KeyC && !event.repeat && self.fractal == Fractal::Mandelbulb {
                            self.slice = if self.slice.is_some() { None } else { Some(0.0) };
                        }
                        if let Some(z) = &mut self.slice {
                            if key == KeyCode::ArrowUp { *z = (*z + 0.05).min(2.0); }
                            if key == KeyCode::ArrowDown { *z = (*z - 0.05).max(-2.0); }
                        }
                        if let Some(window) = &self.window {
                            if matches!(key, KeyCode::KeyN | KeyCode::KeyB) { window.set_title(&format!("{APP_NAME} - Surface shading smoothness {:.6}", self.normal_smoothing)); }
                            else if matches!(key, KeyCode::KeyZ | KeyCode::KeyX) { window.set_title(&format!("{APP_NAME} - Exposure {:+.2} EV", self.exposure)); }
                            else if matches!(key, KeyCode::KeyI | KeyCode::KeyK | KeyCode::KeyU | KeyCode::KeyJ) { window.set_title(&format!("{APP_NAME} - AO strength {:.1} | radius {:.4}", self.ao_strength, self.ao_radius)); }
                            else if self.section_active() {
                                window.set_title(&format!("{APP_NAME} - Section z={:.2} | power {:.0} | C: close", self.slice.unwrap_or_default(), self.power));
                            } else if self.fractal == Fractal::Mandelbulb && matches!(key, KeyCode::PageUp | KeyCode::PageDown) {
                                window.set_title(&format!("{APP_NAME} - Mandelbulb | power {:.0} | Page Up/Down", self.power));
                            } else {
                                window.set_title(&self.fractal_title());
                            }
                        }
                    } else { self.keys.remove(&key); }
                    if let Some(window) = &self.window { window.request_redraw(); }
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                self.looking = state == ElementState::Pressed;
                if let Some(window) = &self.window { window.request_redraw(); }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let delta = match delta { MouseScrollDelta::LineDelta(_, y) => y, MouseScrollDelta::PixelDelta(p) => (p.y as f32) / 100.0 };
                self.camera.speed = (self.camera.speed * (delta * 0.2).exp()).clamp(0.001, 100.0);
                if self.slice.is_none() { if let Some(window) = &self.window { window.set_title(&format!("{APP_NAME} - speed {:.3}", self.camera.speed)); } }
                if let Some(window) = &self.window { window.request_redraw(); }
            }
            WindowEvent::Focused(false) => {
                self.keys.clear();
                self.looking = false;
                self.modifiers = winit::keyboard::ModifiersState::empty();
                if let Some(window) = &self.window { window.request_redraw(); }
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    if let Some(Some(stats)) = renderer.poll_viewport_work() {
                        self.adaptive_viewport.observe(stats);
                    }
                    if renderer.viewport_in_flight.is_some() { renderer.poll_redraw = true; return; }
                    renderer.poll_redraw = false;
                    renderer.poll_noise_sampling();
                    renderer.poll_viewport_denoise();
                }
                if let Some(job) = &self.render_job {
                    let mut finished = None;
                    while let Ok(update) = job.receiver.try_recv() {
                        match update {
                            export::Update::Progress(p) => self.output.progress = p,
                            export::Update::Denoising(backend) => {
                                self.output.progress = 1.0;
                                self.output.status = format!("Denoising PNG on {}…", backend.label());
                            }
                            export::Update::Finished(result) => finished = Some(result),
                        }
                    }
                    if let Some(result) = finished {
                        self.output.status = match result { Ok(path) => format!("Saved: {}", path.display()), Err(e) => e };
                        self.render_job = None;
                    }
                }
                let now = Instant::now();
                let frame_interval = now.duration_since(self.last_frame).as_secs_f32();
                let dt = frame_interval.min(0.05);
                self.last_frame = now;
                let mut moving = self.camera.update(&self.keys, dt);
                let window = self.window.as_ref().unwrap().clone();
                let input = self.egui_state.as_mut().unwrap().take_egui_input(&window);
                let ctx = self.egui_ctx.clone();
                let panels = (self.left_open, self.right_open);
                let previous_fractal = self.fractal;
                let output = ctx.run(input, |ctx| {
                    self.ui(ctx);
                    self.update_render_frame(ctx);
                });
                if panels != (self.left_open, self.right_open) { window.request_redraw(); }
                if let Some(point) = self.focus_point.take() {
                    let mut uniforms = self.camera_uniforms([self.output.width as f32, self.output.height as f32], Some(0), [0.0; 4], false);
                    uniforms[11] = 0.0;
                    if let Some(depth) = self.renderer.as_ref().and_then(|renderer| Renderer::pick_focus(&renderer.device, &renderer.queue, &renderer.camera_buffer, &renderer.focus_pipeline, &renderer.focus_bind, uniforms, point)) {
                        self.effects.focus = depth;
                        self.focus_status = format!("Focused at {depth:.5}");
                    } else {
                        self.focus_status = "No surface found at that point.".into();
                    }
                    window.request_redraw();
                }
                let dragging = ctx.input(|input| input.pointer.primary_down());
                self.observe_history(dragging || text_edit_is_focused(&ctx) || moving || self.looking);
                if self.apply_history_actions() {
                    moving = false;
                    window.request_redraw();
                }
                if self.fractal != previous_fractal { window.set_title(&self.fractal_title()); }
                self.egui_state.as_mut().unwrap().handle_platform_output(&window, output.platform_output.clone());
                if std::mem::take(&mut self.start_export) && self.render_job.is_none() {
                    if let Some(renderer) = &self.renderer {
                        let uniforms = self.camera_uniforms([self.output.width as f32, self.output.height as f32], None, [0.0; 4], true);
                        let request = export::Request {
                            width: self.output.width.clamp(64,8192), height: self.output.height.clamp(64,8192),
                            samples: self.output.samples, noise: self.output.render_noise,
                            time_limit: Duration::from_secs_f32(self.output.time_limit.max(0.0).min(86400.0)),
                            denoise: self.output.denoise, uniforms, environment: self.environment_texture().cloned(),
                        };
                        let (sender, receiver) = std::sync::mpsc::channel();
                        let cancel = Arc::new(std::sync::atomic::AtomicBool::new(false));
                        export::spawn(renderer.device.clone(), renderer.queue.clone(), renderer.export_pipelines.clone(), request, cancel.clone(), sender, window.clone());
                        self.render_job = Some(RenderJob { receiver, cancel });
                        self.output.progress = 0.0; self.output.status = "Rendering…".into();
                        window.request_redraw();
                    }
                }
                let size = window.inner_size();
                let screen = ctx.screen_rect();
                let frame = self.frame.unwrap_or(screen);
                let projection = camera_effects::viewport_projection(screen, frame);
                let mut scene = self.viewport_scene(size, projection);
                let target = self.still_sample_target();
                scene.push(target as f32);
                let denoise_viewport = if !self.diagnostic && !self.section_active() { self.output.viewport_denoise } else { None };
                scene.push(match denoise_viewport { None => 0.0, Some(denoise::Backend::Cpu) => 1.0, Some(denoise::Backend::Gpu) => 2.0 });
                let viewport_noise = self.output.viewport_noise;
                let noise_plan = if !self.diagnostic && !self.section_active() {
                    viewport_noise.plan(target).filter(|_| self.renderer.as_mut().is_some_and(|renderer| {
                        renderer.prepare_noise_sampling(denoise_viewport.is_some(), window.clone())
                    }))
                } else { None };
                scene.extend([flag(noise_plan.is_some()), viewport_noise.threshold, viewport_noise.min_samples as f32]);
                let wants_guiding = self.output.intelligent_sampling && !viewport_noise.enabled
                    && target >= 3 && !self.diagnostic && !self.section_active();
                let intelligent_sampling = wants_guiding && self.renderer.as_mut().is_some_and(|renderer| {
                    let wake_window = window.clone();
                    renderer.prepare_intelligent_sampling(denoise_viewport.is_some(), move || wake_window.request_redraw())
                });
                scene.push(flag(intelligent_sampling));
                let selected_mutation = self.mutation.selected(self.fractal);
                let extra_selected = self.extra_mutations.iter().any(|stage| stage.selected(self.fractal).is_some());
                let mutation_selected = selected_mutation.is_some() || extra_selected;
                let distance_blend = extra_selected || (self.mutation.weight > 0.0 && selected_mutation.is_some());
                let ray_ao_preview = self.ray_traced_ao_preview();
                let path_preview = self.path_traced_preview();
                let shadows_preview = self.has_directional_shadows() && !self.diagnostic && !self.section_active();
                let scene_changed = scene != self.last_scene;
                let editing = scene_changed && dragging && ctx.is_using_pointer()
                    && (mutation_selected || ray_ao_preview || shadows_preview || path_preview);
                let navigating = moving || self.looking;
                let interacting = navigating || editing;
                let environment = self.environment_texture().cloned();
                let environment_changed = self.renderer.as_ref().is_some_and(|renderer| renderer.environment_view != environment);
                let reset = scene_changed || interacting || environment_changed;
                if (interacting && !self.viewport_was_interacting)
                    || (!interacting && (scene_changed || environment_changed))
                {
                    self.adaptive_viewport.restart();
                }
                if !navigating || !self.viewport_was_interacting { self.navigation_sample = 0; }
                self.viewport_was_interacting = interacting;
                if reset {
                    self.last_scene = scene;
                    self.idle_ready = false;
                    self.progress_samples = 0;
                    self.still_timing = StillTiming::default();
                    self.progress_tile = 0;
                    self.refine_tiles_per_frame = 1;
                    self.idle_deadline = None;
                    if let Some(renderer) = &mut self.renderer {
                        renderer.noise_pending = None;
                        renderer.noise_remaining = None;
                        renderer.noise_tiles = None;
                        if !viewport_noise.enabled { renderer.noise_buffers = None; }
                    }
                }
                if !interacting && !self.idle_ready && self.idle_deadline.is_none() {
                    self.idle_deadline = Some(Instant::now() + Duration::from_millis(180));
                }
                let progressing = self.idle_ready && !interacting;
                let capture_slot = self.capture_slot.take();
                // Capture a preset from the accumulated image when a still is already visible.
                let composite_progress = progressing && (capture_slot.is_none() || self.progress_samples > 0);
                let target = if self.renderer.as_ref().is_some_and(|renderer| renderer.noise_remaining == Some(0)) {
                    self.progress_samples
                } else { target };
                let awaiting_noise = self.renderer.as_ref().is_some_and(|renderer| renderer.noise_pending.is_some());
                let awaiting_denoise = self.renderer.as_ref().is_some_and(|renderer| {
                    renderer.viewport_denoise.awaiting_first_preview()
                        || (self.progress_tile == 0 && renderer.viewport_denoise.snapshot_pending(
                            self.progress_samples, target, self.output.viewport_denoise_interval))
                });
                // Keep a complete checkpoint until OIDN can take its snapshot.
                let advance = composite_progress && capture_slot.is_none() && self.progress_samples < target
                    && !awaiting_denoise && !awaiting_noise;
                let progress = composite_progress.then_some(self.progress_samples);
                let mut data = self.camera_uniforms([size.width.max(1) as f32, size.height.max(1) as f32], progress, projection, false);
                if navigating && capture_slot.is_none() && !self.diagnostic {
                    data[MOVING_SAMPLES_UNIFORM] = self.moving_samples as f32;
                    data[MOVING_BOUNCES_UNIFORM] = self.moving_bounces as f32;
                    data[NAVIGATION_SAMPLE_UNIFORM] = self.navigation_sample as f32;
                }
                data[10] = frame.height() * output.pixels_per_point;
                let preview_divisor = if capture_slot.is_some() && !composite_progress {
                    4
                } else if self.adaptive_viewport.enabled {
                    if progressing { 1 } else { self.adaptive_viewport.divisor() }
                } else if progressing {
                    if ray_ao_preview { 2 } else { 1 }
                } else if ray_ao_preview || path_preview {
                    4
                } else if shadows_preview {
                    2
                } else if mutation_selected || !navigating {
                    if navigating && !distance_blend { 2 } else { 4 }
                } else {
                    1
                };
                let full_size = [size.width.max(1), size.height.max(1)];
                let render_size = full_size.map(|side| side.div_ceil(preview_divisor));
                let preview_work = (self.adaptive_viewport.enabled && !progressing && capture_slot.is_none())
                    .then_some(adaptive_viewport::PreviewWork {
                        size: render_size, divisor: preview_divisor, samples: data[MOVING_SAMPLES_UNIFORM].max(1.0) as u32,
                    });
                if preview_work.is_some() {
                    data[18] = (self.quality * 1.25).min(4.0).max(self.quality);
                }
                let active_tiles = self.renderer.as_ref().and_then(|renderer| renderer.noise_tiles.as_ref())
                    .filter(|_| noise_plan.is_some());
                let [columns, rows] = noise_sampling::viewport_tile_grid(render_size);
                let tile_count = active_tiles.map_or(columns * rows, noise_sampling::ActiveTiles::len);
                let refine_tiles = advance.then(|| {
                    if self.progress_tile > 0 || self.progress_samples > 0 {
                        // Back off quickly on hard tiles. Increase gradually when presentation stays responsive.
                        if frame_interval > 0.030 {
                            self.refine_tiles_per_frame = (self.refine_tiles_per_frame / 2).max(1);
                        } else if frame_interval < 0.023 && self.progress_tile % 8 == 0 {
                            self.refine_tiles_per_frame = (self.refine_tiles_per_frame + 1).min(MAX_REFINE_TILES_PER_FRAME);
                        }
                    }
                    let end = match active_tiles {
                        Some(tiles) => tiles.batch_end(self.progress_tile, self.refine_tiles_per_frame),
                        None => (self.progress_tile + self.refine_tiles_per_frame).min(tile_count),
                    };
                    self.progress_tile..end
                });
                let next_tile = refine_tiles.as_ref().map_or(tile_count, |tiles| tiles.end);
                let pass_started = advance.then(Instant::now);
                let capture_result = self.renderer.as_mut().and_then(|renderer| {
                    renderer.set_environment(environment.as_ref());
                    renderer.intelligent_sampling.enabled = intelligent_sampling;
                    renderer.noise_plan = noise_plan;
                    renderer.viewport_denoise.configure(denoise_viewport, reset);
                    let ready_to_present = renderer.viewport_denoise.ready_to_present();
                    let capture = renderer.render(data, progress, advance, refine_tiles, render_size, output, &ctx,
                        capture_slot.is_some(), self.adaptive_viewport.enabled, preview_work);
                    if ready_to_present && !renderer.viewport_denoise.ready_to_present() { window.request_redraw(); }
                    capture
                });
                if let Some(slot) = capture_slot {
                    match capture_result.unwrap_or_else(|| Err("Preview unavailable".into())) {
                        Ok(image) => {
                            let thumb = image::imageops::resize(&image, 200, 140, image::imageops::FilterType::Triangle);
                            match thumb.save(self.presets.file(slot, "png")) {
                                Ok(()) => {
                                    self.presets.preview_saved(slot);
                                    self.preset_status = "Preset saved".into();
                                }
                                Err(e) => self.preset_status = format!("Preset saved; preview failed: {e}"),
                            }
                        }
                        Err(e) => self.preset_status = format!("Preset saved; preview failed: {e}"),
                    }
                    window.request_redraw();
                }
                let submitted = self.renderer.as_ref().is_some_and(|renderer| renderer.frame_submitted);
                if submitted && navigating && capture_slot.is_none() {
                    self.navigation_sample = (self.navigation_sample + self.moving_samples) % 1024;
                }
                if let Some(started) = pass_started.filter(|_| submitted) {
                    self.still_timing.started.get_or_insert(started);
                    if next_tile < tile_count {
                        self.progress_tile = next_tile;
                    } else {
                        self.progress_tile = 0;
                        self.progress_samples += 1;
                    }
                }
                // Denoise only at the end of a complete pass over the viewport.
                if denoise_viewport.is_some() && composite_progress && self.progress_tile == 0 {
                    if let Some(renderer) = &mut self.renderer {
                        let wake_window = window.clone();
                        if renderer.viewport_denoise.start(&renderer.device, &renderer.queue, &renderer.accumulation, render_size,
                            self.progress_samples, target, self.output.viewport_denoise_interval, move || wake_window.request_redraw())
                        {
                            window.request_redraw();
                        }
                    }
                }
                if composite_progress && submitted && self.progress_tile == 0 && self.progress_samples == target {
                    if let Some(renderer) = &self.renderer {
                        let denoiser = &renderer.viewport_denoise;
                        let final_image = denoise_viewport.is_none() || denoiser.error.is_some()
                            || denoiser.presented.as_ref().is_some_and(|frame| frame.samples == target);
                        if final_image && renderer.noise_pending.is_none() {
                            self.still_timing.finish(&renderer.queue);
                        }
                    }
                }
                if moving || advance || awaiting_noise || (self.adaptive_viewport.enabled && interacting) { window.request_redraw(); }
            }
            _ => {}
        }
    }
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_none() {
            if let Some(mut deadline) = self.startup.next_repaint {
                let now = Instant::now();
                if now >= deadline {
                    if let Some(window) = &self.window { window.request_redraw(); }
                    deadline = now + Duration::from_secs(1);
                    self.startup.next_repaint = Some(deadline);
                }
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            } else {
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            return;
        }
        if let Some(renderer) = &mut self.renderer {
            if let Some(result) = renderer.poll_viewport_work() {
                if let Some(stats) = result { self.adaptive_viewport.observe(stats); }
                if let Some(window) = &self.window {
                    if renderer.poll_redraw { window.request_redraw(); }
                }
                renderer.poll_redraw = false;
            }
            if renderer.viewport_in_flight.is_some() {
                event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + adaptive_viewport::POLL_INTERVAL));
                return;
            }
            if let Some(pending) = &self.still_timing.pending {
                let _ = renderer.device.poll(wgpu::Maintain::Poll);
                match pending.try_recv() {
                    Ok(finished) => {
                        self.still_timing.completed = self.still_timing.started.map(|started| finished - started);
                        self.still_timing.pending = None;
                        if let Some(window) = &self.window { window.request_redraw(); }
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + adaptive_viewport::POLL_INTERVAL));
                        return;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => self.still_timing.pending = None,
                }
            }
        }
        if let Some(deadline) = self.idle_deadline {
            if Instant::now() >= deadline {
                self.idle_deadline = None;
                self.idle_ready = true;
                if let Some(window) = &self.window { window.request_redraw(); }
                event_loop.set_control_flow(ControlFlow::Wait);
            } else {
                event_loop.set_control_flow(ControlFlow::WaitUntil(deadline));
            }
        } else { event_loop.set_control_flow(ControlFlow::Wait); }
    }
    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.looking {
                let dx = delta.0 as f32 * 0.003;
                let dy = delta.1 as f32 * 0.003;
                self.camera.look(dx, dy);
                if let Some(window) = &self.window { window.request_redraw(); }
            }
        }
    }
}
fn main() {
    let event_loop = EventLoop::<startup::Event>::with_user_event().build().expect("Event loop");
    let mut app = App { event_proxy: Some(event_loop.create_proxy()), ..App::default() };
    event_loop.run_app(&mut app).expect("Application");
}

#[cfg(test)]
mod tests {
    #[test]
    fn render_noise_checkbox_fits_its_clip_when_hovered() {
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(900.0, 900.0));
        let render = |ctx: &egui::Context, app: &mut super::App, pointer: Option<egui::Pos2>| {
            ctx.run(egui::RawInput {
                screen_rect: Some(screen),
                events: pointer.into_iter().map(egui::Event::PointerMoved).collect(),
                ..Default::default()
            }, |ctx| app.ui(ctx))
        };
        let labels = |frame: &egui::FullOutput| {
            frame.shapes.iter().filter_map(|shape| match &shape.shape {
                egui::Shape::Text(text) if text.galley.text() == "Noise Threshold" => Some(text.visual_bounding_rect()),
                _ => None,
            }).collect::<Vec<_>>()
        };
        let checkbox = |frame: &egui::FullOutput, label: egui::Rect| {
            frame.shapes.iter().find_map(|shape| match &shape.shape {
                egui::Shape::Rect(rect) if (0.0..=8.0).contains(&(label.left() - rect.rect.right()))
                    && (rect.rect.center().y - label.center().y).abs() < 3.0
                    && (12.0..=21.0).contains(&rect.rect.width()) =>
                    Some((shape.shape.visual_bounding_rect(), shape.clip_rect)),
                _ => None,
            }).expect("noise checkbox")
        };
        for index in 0..2 {
            let mut app = super::App::default();
            let ctx = egui::Context::default();
            let idle = render(&ctx, &mut app, None);
            let label = labels(&idle)[index];
            let (idle_square, _) = checkbox(&idle, label);
            let hovered = render(&ctx, &mut app, Some(label.center()));
            let (square, clip) = checkbox(&hovered, labels(&hovered)[index]);
            assert!(square.width() > idle_square.width(), "checkbox {index} stayed at {square:?} from {idle_square:?}");
            assert!(clip.contains_rect(square), "checkbox {square:?} is clipped by {clip:?}");
        }
    }

    #[test]
    fn aperture_shape_round_trips_and_legacy_presets_keep_a_circular_aperture() {
        let mut app = super::App::default();
        app.fractal = super::Fractal::FourBranch;
        app.effects.dof_enabled = true;
        app.effects.aperture_blades = 6;
        app.effects.aperture_rotation = 30.0;
        let preset = app.preset_values();
        let mut restored = super::App::default();
        restored.restore_scene(&preset);
        assert_eq!(restored.effects.bokeh(), [6.0, 30.0]);
        let uniforms = restored.camera_uniforms([640.0, 480.0], None, [0.0; 4], true);
        assert_eq!(&uniforms[super::BOKEH_UNIFORM_OFFSET..][..2], &[6.0, 30.0_f32.to_radians()]);
        restored.restore_scene(&preset[..super::PRESET_BOKEH_FIELDS]);
        assert_eq!(restored.fractal, super::Fractal::FourBranch);
        assert_eq!(restored.effects.bokeh(), [0.0, 0.0]);
    }

    #[test]
    fn render_panel_labels_fit_at_minimum_width() {
        let mut app = super::App::default();
        app.lighting.path_traced = true;
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1000.0, 1000.0));
        let frame = ctx.run(egui::RawInput { screen_rect: Some(screen), ..Default::default() }, |ctx| {
            app.ui(ctx);
        });
        for label in ["Noise Threshold", "Intelligent ray sampling", "Denoise Interval", "Moving bounces", "Max bounces"] {
            let (bounds, clip) = frame.shapes.iter().find_map(|shape| match &shape.shape {
                egui::Shape::Text(text) if text.galley.text() == label => Some((text.visual_bounding_rect(), shape.clip_rect)),
                _ => None,
            }).unwrap_or_else(|| panic!("Missing Render panel label {label}"));
            assert!(clip.contains_rect(bounds), "{label} is clipped: {bounds:?} outside {clip:?}");
        }
    }

    #[test]
    fn hdr_environment_intensity_survives_preset_restore() {
        let mut app = super::App::default();
        app.lighting.path_traced = true;
        app.lighting.environment = super::HDRI_ENVIRONMENT;
        app.lighting.environment_strength = 10.0;
        let mut values = app.preset_values();
        let mut restored = super::App::default();
        restored.restore_scene(&values);
        assert_eq!(restored.lighting.environment_strength, 10.0);

        let uniforms = restored.camera_uniforms([640.0, 480.0], None, [0.0; 4], true);
        assert_eq!(&uniforms[super::LIGHTING_UNIFORM_OFFSET..][..3], &[10.0; 3]);

        values[super::PRESET_PATH_FIELDS] = 0.0;
        restored.restore_scene(&values);
        assert_eq!(restored.lighting.environment, super::HDRI_ENVIRONMENT);
        assert_eq!(restored.lighting.environment_strength, 10.0);

        values[super::PRESET_PATH_FIELDS + 1] = 0.0;
        restored.restore_scene(&values);
        assert_eq!(restored.lighting.environment_strength, 2.0);
    }

    #[test]
    fn png_uses_the_viewports_stationary_detail_settings() {
        let mut app = super::App::default();
        app.still_samples = 128;
        for (quality, stationary_quality) in [(0.7, 0.7 * 1.25), (4.0, 4.0), (8.0, 8.0)] {
            app.quality = quality;
            for path_traced in [false, true] {
                app.lighting.path_traced = path_traced;
                let viewport = app.camera_uniforms([1380.0, 900.0], Some(0), [0.0; 4], false);
                let moving = app.camera_uniforms([1380.0, 900.0], None, [0.0; 4], false);
                let export = app.camera_uniforms([3840.0, 2160.0], None, [0.0; 4], true);
                assert_eq!([export[18], export[23]], [stationary_quality, 1.0]);
                assert_eq!([viewport[18], viewport[23]], [stationary_quality, 1.0]);
                assert_eq!([moving[18], moving[23]], [quality, 0.0]);
                assert_eq!(app.still_sample_target(), 128);
            }
        }
        assert_eq!(app.output.samples, 16);
        let mut legacy = app.preset_values();
        legacy[20] = 0.0;
        legacy[34] = 64.0;
        app.restore_scene(&legacy[..super::PRESET_MOVING_SAMPLES]);
        assert_eq!(app.still_sample_target(), super::DEFAULT_VIEWPORT_SAMPLES);
        app.diagnostic = true;
        assert_eq!(app.still_sample_target(), 1);
    }

    #[test]
    fn shift_drag_tracks_relative_motion_and_respects_slider_limits() {
        for (range, start, logarithmic, degrees) in [
            (-2.5..=-1.15, -1.5, false, false),
            (0.2..=0.5234, 0.5, false, false),
            (0.0001..=30.0, 0.1, true, false),
            (0.0..=30.0, 0.1, true, false),
            (-180.0..=180.0, 20.0, false, true),
        ] {
            let ctx = egui::Context::default();
            let track_width = ctx.style().spacing.slider_width;
            let mut value = start;
            let mut frame = |shift, events| {
                let mut track = egui::Rect::NOTHING;
                let _ = ctx.run(egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 300.0))),
                    modifiers: egui::Modifiers { shift, ..Default::default() },
                    events,
                    ..Default::default()
                }, |ctx| {
                    egui::CentralPanel::default().show(ctx, |ui| {
                        track = super::precision_slider(ui, &mut value, range.clone(), "Test", logarithmic, degrees).rect;
                    });
                });
                (track, value)
            };
            let (rect, _) = frame(false, vec![]);
            let pos = egui::pos2(rect.left() + 8.0, rect.center().y);
            let button = |pos, pressed, shift| egui::Event::PointerButton {
                pos, pressed, button: egui::PointerButton::Primary,
                modifiers: egui::Modifiers { shift, ..Default::default() },
            };
            assert_eq!(frame(true, vec![egui::Event::PointerMoved(pos), button(pos, true, true)]).1, start);
            let next = pos + egui::vec2(track_width / 2.0, 0.0);
            let actual = frame(true, vec![egui::Event::PointerMoved(next)]).1;
            let (lo, hi) = (*range.start(), *range.end());
            let mid = if logarithmic {
                let floor = if lo == 0.0 { 0.0001 } else { lo };
                start * ((hi / floor).ln() * 0.05).exp()
            } else {
                start + (hi - lo) * 0.05
            };
            assert!((actual - mid).abs() < 1.0e-5, "range {range:?}: {actual} vs {mid}");
            assert_eq!(frame(false, vec![]).1, actual);
            assert_eq!(frame(false, vec![button(next, false, false)]).1, actual);

            frame(true, vec![egui::Event::PointerMoved(pos), button(pos, true, true)]);
            let right = pos + egui::vec2(2000.0, 0.0);
            let left = pos - egui::vec2(2000.0, 0.0);
            assert_eq!(frame(true, vec![egui::Event::PointerMoved(right)]).1, hi);
            assert_eq!(frame(true, vec![egui::Event::PointerMoved(left)]).1, lo);
            assert_eq!(frame(true, vec![button(left, false, true)]).1, lo);
        }
    }

    #[test]
    fn controls_tabs_fit_at_minimum_width_and_reopen() {
        let mut app = super::App {
            fractal: super::Fractal::Mandelbox,
            material: 4,
            ..Default::default()
        };
        let ctx = egui::Context::default();
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(760.0, 320.0));
        let input = || egui::RawInput { screen_rect: Some(screen), ..Default::default() };
        let panel_id = egui::Id::new("controls");
        for path_traced in [false, true] {
            app.lighting.path_traced = path_traced;
            let mut previous_scene = None;
            for tab in super::ControlsTab::ALL {
                app.controls_tab = tab;
                ctx.data_mut(|data| {
                    data.insert_persisted(panel_id, egui::containers::panel::PanelState {
                        rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, screen.height())),
                    });
                });
                let _ = ctx.run(input(), |ctx| {
                    app.ui(ctx);
                    app.update_render_frame(ctx);
                    let width = ctx.available_rect().left();
                    assert!((super::CONTROLS_MIN_WIDTH..=super::CONTROLS_MIN_WIDTH + 2.0).contains(&width), "{} tab width: {width}", tab.label());
                    assert!(ctx.available_rect().width() >= 150.0);
                });
                let projection = super::camera_effects::viewport_projection(screen, app.frame.unwrap());
                let scene = app.viewport_scene(winit::dpi::PhysicalSize::new(760, 320), projection);
                if let Some(previous) = &previous_scene {
                    let changes: Vec<_> = scene.iter().zip(previous).enumerate()
                        .filter_map(|(index, (value, old))| (value != old).then_some((index, old, value))).collect();
                    assert!(previous == &scene, "{} changed the viewport scene: {changes:?}", tab.label());
                }
                previous_scene = Some(scene);
            }
        }
        app.left_open = false;
        let _ = ctx.run(input(), |ctx| {
            app.ui(ctx);
            assert_eq!(ctx.available_rect().left(), 0.0);
        });
        app.left_open = true;
        let _ = ctx.run(input(), |ctx| {
            app.ui(ctx);
            assert!(ctx.available_rect().left() >= super::CONTROLS_MIN_WIDTH);
        });
    }

    #[test]
    fn startup_frames_menger_from_above_a_corner() {
        let app = super::App::default();
        assert_eq!(app.fractal, super::Fractal::Menger);
        assert_eq!(app.material, super::MATCAP_MATERIAL);
        assert_eq!(app.matcap.style_name(), "Original");
        assert_eq!(app.fractal_title(), "MengerVibes - Menger");
        let [x, y, z] = app.camera.position;
        assert!(x > 1.0 && x == y && y == z);
        assert_eq!(app.camera.view_roll(), 0.0);
        let [forward, _, _] = app.camera.basis(app.camera.view_roll());
        let expected = -1.0 / 3.0_f32.sqrt();
        assert!(forward.into_iter().all(|axis| (axis - expected).abs() < 1.0e-6));
    }

    #[test]
    fn rolled_camera_looks_horizontally_across_the_pole() {
        let mut camera = super::Camera {
            pitch: 1.54,
            roll: std::f32::consts::FRAC_PI_2,
            ..Default::default()
        };
        let [_, _, initial_up] = camera.basis(camera.view_roll());
        for _ in 0..8 {
            camera.look(0.01, 0.0);
            let [_, _, up] = camera.basis(camera.view_roll());
            for (actual, expected) in up.into_iter().zip(initial_up) {
                assert!((actual - expected).abs() < 1.0e-5);
            }
        }
        assert_eq!(camera.roll, std::f32::consts::FRAC_PI_2);
        assert!(camera.pitch < 1.54);
        camera.set_roll(0.0);
        assert_eq!(camera.view_roll(), 0.0);
    }

    #[test]
    fn switching_formulas_reframes_only_a_camera_at_its_starting_view() {
        use super::Fractal::{Mandelbox, Mandelbulb, Menger};
        let mut app = super::App::default();
        assert_eq!(app.fractal, Menger);
        app.select_fractal(Mandelbox);
        assert_eq!(app.camera.position, super::Camera::framed_view(Mandelbox).position);
        app.camera.position = [2.0, 4.0, 8.0];
        app.select_fractal(Mandelbulb);
        assert_eq!(app.camera.position, [2.0, 4.0, 8.0]);
        app.reset_framing();
        assert_eq!(app.camera.position, super::Camera::framed_view(Mandelbulb).position);
        assert_eq!(app.effects.focal, super::STARTING_FOCAL_MM);
    }

    #[test]
    fn reset_defaults_restores_the_startup_state_and_can_be_undone() {
        let mut app = super::App::default();
        app.fractal = super::Fractal::Mandelbox;
        app.camera.position = [4.0, 2.0, 6.0];
        app.exposure = 1.7;
        app.effects.focal = 55.0;
        app.output.width = 800;
        app.output.viewport_denoise = Some(super::denoise::Backend::Cpu);
        app.hdri_input = "environment.hdr".into();
        app.picking_focus = true;
        app.progress_samples = 12;
        app.show_frame = false;
        let edited = app.preset_values();

        app.preset_action(super::preset_gallery::Action::ResetDefaults);

        let defaults = super::App::default();
        assert_eq!(app.preset_values(), defaults.preset_values());
        assert_eq!(app.output.width, defaults.output.width);
        assert_eq!(app.output.viewport_denoise, defaults.output.viewport_denoise);
        assert_eq!(app.show_frame, defaults.show_frame);
        assert_eq!(app.progress_samples, 0);
        assert!(app.hdri_input.is_empty());
        assert!(!app.picking_focus);

        app.history_actions.push(super::history::Action::Undo);
        assert!(app.apply_history_actions());
        assert_eq!(app.preset_values(), edited);
        assert_eq!(app.output.width, 800);
        assert!(!app.show_frame);
    }

    #[test]
    fn formula_depths_round_trip_for_base_and_mutations() {
        use super::Fractal::{Mandelbox, Sierpinski};
        let mut app = super::App::default();
        app.fractal = Sierpinski;
        app.sierpinski.depth = 14;
        app.orbit_depths = super::OrbitDepths([5, 14, 10, 8]);
        app.mutation.formula = Some(Mandelbox);
        app.mutation.orbit_depths.0[1] = 14;
        app.extra_mutations.push(super::MutationStage::default());
        app.extra_mutations[0].sierpinski.depth = 3;
        app.extra_mutations[0].orbit_depths.0[2] = 7;

        let mut loaded = super::App::default();
        loaded.restore_scene(&app.preset_values());
        assert_eq!(loaded.fractal, Sierpinski);
        assert_eq!(loaded.preset_values(), app.preset_values());
        let uniforms = loaded.camera_uniforms([100.0; 2], None, [0.0; 4], true);
        assert_eq!(uniforms[super::SIERPINSKI_UNIFORM_OFFSET], 14.0);
        assert_eq!(uniforms[super::ORBIT_DEPTH_UNIFORM_OFFSET..super::FOUR_BRANCH_UNIFORM_OFFSET], [-6.0, 3.0, -1.0, -3.0]);
        assert_eq!(loaded.mutation.field_uniforms(Mandelbox, 0.5)[2], 3.0);
        assert_eq!(loaded.extra_mutations[0].field_uniforms(Sierpinski, 0.5)[4], 3.0);

        loaded.restore_scene(&super::App::default().preset_values()[..super::PRESET_SIERPINSKI_FIELDS]);
        assert_eq!(loaded.sierpinski, super::SierpinskiParameters::default());
        assert_eq!(loaded.orbit_depths, super::OrbitDepths::default());
    }

    #[test]
    fn four_branch_parameters_survive_presets_and_reach_the_shader() {
        use super::{App, FourBranchParameters, Fractal, MutationStage};
        let parameters = FourBranchParameters { depth: 14, thickness: 0.13, spread: 1.2, twist: -45.0 };
        let mut app = App::default();
        app.select_fractal(Fractal::FourBranch);
        app.four_branch = parameters;
        let mut restored = App::default();
        restored.restore_scene(&app.preset_values());
        assert_eq!(restored.fractal, Fractal::FourBranch);
        assert_eq!(restored.four_branch, parameters);
        let uniforms = restored.camera_uniforms([100.0; 2], None, [0.0; 4], true);
        let offset = super::FOUR_BRANCH_UNIFORM_OFFSET;
        assert_eq!(uniforms[offset..offset + 5], parameters.uniforms());

        app.fractal = Fractal::Sierpinski;
        app.mutation.formula = Some(Fractal::FourBranch);
        app.mutation.four_branch = parameters;
        app.extra_mutations.push(MutationStage { four_branch: parameters, ..MutationStage::default() });
        restored.restore_scene(&app.preset_values());
        assert_eq!(restored.mutation.field_uniforms(Fractal::FourBranch, 0.5)[4..9], parameters.uniforms());
        assert_eq!(restored.extra_mutations[0].four_branch, parameters);

        let old_preset = app.preset_values();
        restored.restore_scene(&old_preset[..super::PRESET_FOUR_BRANCH_FIELDS]);
        assert_eq!(restored.fractal, Fractal::Sierpinski);
        assert_eq!(restored.four_branch, FourBranchParameters::default());
        assert_eq!(restored.mutation.four_branch, FourBranchParameters::default());
    }

    #[test]
    fn matcap_preset_round_trips_with_separate_principled_settings() {
        let path = std::env::temp_dir().join(format!("mengervibes-matcap-test-{}.txt", std::process::id()));
        let mut app = super::App::default();
        app.select_material(super::MATCAP_MATERIAL);
        app.matcap.style = 3;
        app.matcap.rotation = 35.0;
        app.matcap.roughness = 0.42;
        app.matcap.metallic = 0.75;
        app.matcap.finish.coat = 0.3;
        app.custom_finish.coat = 0.5;
        assert!(app.save_preset(&path));
        let mut loaded = super::App::default();
        loaded.load_preset(&path);
        std::fs::remove_file(path).unwrap();
        assert_eq!(loaded.material, super::MATCAP_MATERIAL);
        assert_eq!(loaded.matcap.values(), app.matcap.values());
        assert_eq!(loaded.custom_finish.coat, 0.5);
        let uniforms = loaded.camera_uniforms([100.0; 2], None, [0.0; 4], true);
        assert_eq!(uniforms[12], 4.0);
        assert_eq!(uniforms[52], 35.0_f32.to_radians());
    }

    #[test]
    fn presets_round_trip_and_recover_duplicated_sample_count() {
        let path = std::env::temp_dir().join(format!("mengervibes-preset-test-{}.txt", std::process::id()));
        let path = path.to_str().unwrap();
        let mut original = super::App::default();
        original.camera.position = [1.0, 2.0, 3.0];
        original.camera.roll = 0.7;
        original.camera.look_roll = -0.3;
        original.camera.pitch = 1.56;
        original.exposure = 2.2;
        original.distant_filter = 3.3;
        original.still_samples = 64;
        original.moving_samples = 4;
        original.ao_strength = 4.0;
        original.moving_bounces = 3;
        original.lighting.color = [0.1, 0.2, 0.3];
        original.lighting.environment_color = [0.8, 0.5, 0.3];
        original.lighting.environment_strength = 0.45;
        original.lighting.angular_radius = 8.0;
        original.lighting.enabled = false;
        original.select_material(super::DIFFUSE_GRAY_MATERIAL);
        original.material_settings = [0.18, 0.38, 0.58, 0.71, 0.6];
        original.material_finish.coat = 0.4;
        original.material_finish.emission_strength = 2.0;
        original.custom_finish.ior = 2.0;
        original.effects.focal = 50.0;
        original.effects.focus = 0.15;
        original.effects.aperture = 0.002;
        original.effects.dof_enabled = true;
        original.contrast = 2.5;
        original.quality = 8.0;
        original.normal_smoothing = 0.0;
        original.fractal = super::Fractal::Menger;
        original.menger_parameters.depth = 14;
        original.menger_parameters.tunnel_width = 1.15;
        original.menger_parameters.rotation = [15.0, -25.0, 138.0];
        original.background_color = [0.32, 0.45, 0.88];
        original.mandelbox_parameters.scale = -1.7;
        original.mandelbox_parameters.inner_radius = 0.65;
        original.mandelbulb_parameters.polar_offset = 0.3;
        original.mandelbulb_parameters.seed_offset = -0.2;
        original.mutation.formula = Some(super::Fractal::Mandelbulb);
        original.mutation.weight = 0.6;
        original.mutation.bulb_power = 6.0;
        original.mutation.bulb_parameters.azimuth_twist = 0.22;
        original.mutation.menger_parameters.depth = 14;
        original.mutation.menger_parameters.tunnel_width = 0.8;
        original.mutation.menger_parameters.rotation = [12.0, 50.0, -26.0];
        original.effects.fog_density = 0.4;
        assert!(original.save_preset(path));
        let text = std::fs::read_to_string(path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), super::PRESET_VALUE_COUNT);
        let mut loaded = super::App::default();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.camera.position, original.camera.position);
        assert_eq!(loaded.camera.roll, original.camera.roll);
        assert_eq!(loaded.camera.look_roll, original.camera.look_roll);
        assert_eq!(loaded.camera.pitch, original.camera.pitch);
        assert_eq!(loaded.exposure, original.exposure);
        assert_eq!(loaded.distant_filter, original.distant_filter);
        assert_eq!(loaded.lighting.color, original.lighting.color);
        assert_eq!(loaded.lighting.environment_color, original.lighting.environment_color);
        assert_eq!(loaded.lighting.environment_strength, original.lighting.environment_strength);
        assert_eq!(loaded.lighting.angular_radius, original.lighting.angular_radius);
        assert_eq!(loaded.material, super::DIFFUSE_GRAY_MATERIAL);
        assert_eq!(loaded.material_settings, original.material_settings);
        assert_eq!(loaded.material_finish.values(), original.material_finish.values());
        assert_eq!(loaded.custom_finish.values(), original.custom_finish.values());
        let mut legacy_path = original.preset_values();
        legacy_path[super::PRESET_PATH_FIELDS] = 1.0;
        let mut old_path_scene = super::App::default();
        old_path_scene.restore_scene(&legacy_path[..super::PRESET_SURFACE_FIELDS]);
        assert_eq!(old_path_scene.material, super::STANDARD_MATERIAL);
        assert_eq!(old_path_scene.lighting.color, original.lighting.color);
        assert!(!loaded.lighting.enabled);
        assert!(loaded.lighting.shadows);
        for export in [false, true] {
            let uniforms = loaded.camera_uniforms([64.0, 64.0], None, [0.0; 4], export);
            assert_eq!(uniforms[34], 0.0);
            assert_eq!(uniforms[24..28], [0.18, 0.38, 0.58, 1.0]);
            assert_eq!(uniforms[31], 0.0);
            assert_eq!(uniforms[super::SURFACE_UNIFORM_OFFSET..super::SIERPINSKI_UNIFORM_OFFSET],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        }
        let mut legacy_lighting = super::App::default();
        legacy_lighting.restore_scene(&original.preset_values()[..super::PRESET_LIGHT_FLAGS]);
        assert!(legacy_lighting.lighting.enabled);
        assert!(!legacy_lighting.lighting.shadows);
        assert_eq!(legacy_lighting.lighting.environment_color, [1.0; 3]);
        assert_eq!(legacy_lighting.lighting.environment_strength, 0.45);
        assert_eq!(legacy_lighting.lighting.angular_radius, 0.0);
        assert_eq!(loaded.still_samples, 64);
        assert_eq!(loaded.moving_samples, 4);
        assert_eq!(loaded.ao_strength, 4.0);
        assert_eq!(loaded.moving_bounces, 3);
        assert_eq!(legacy_lighting.moving_samples, 1);
        assert_eq!(legacy_lighting.moving_bounces, 1);
        assert_eq!(loaded.effects.preset(), original.effects.preset());
        assert_eq!(loaded.contrast, 2.5);
        assert_eq!(loaded.quality, 8.0);
        assert_eq!(loaded.normal_smoothing, 0.0);
        assert_eq!(loaded.fractal, super::Fractal::Menger);
        assert_eq!(loaded.menger_parameters, original.menger_parameters);
        assert_eq!(loaded.background_color, original.background_color);
        assert_eq!(loaded.mandelbox_parameters, original.mandelbox_parameters);
        assert_eq!(loaded.mandelbulb_parameters, original.mandelbulb_parameters);
        assert_eq!(loaded.mutation.values(), original.mutation.values());
        let legacy_effects = super::CameraEffects::from_preset(&original.effects.preset()[..8]);
        assert!(legacy_effects.dof_enabled);
        assert_eq!(legacy_effects.aperture, original.effects.aperture);
        assert!((loaded.effects.f_stop() - 12.5).abs() < 0.0001);
        loaded.effects.set_focal(100.0);
        assert!((loaded.effects.f_stop() - 12.5).abs() < 0.0001);
        assert!((loaded.effects.aperture - 0.004).abs() < 0.000001);
        loaded.effects.set_focal(original.effects.focal);
        loaded.effects.dof_enabled = false;
        loaded.output.samples = 1;
        for export in [false, true] {
            let uniforms = loaded.camera_uniforms([1920.0, 1080.0], None, [0.0; 4], export);
            assert_eq!(uniforms[42], 0.0);
            assert_eq!(uniforms[51], 2.5);
            assert_eq!(uniforms[56..60], original.mandelbox_parameters.uniforms());
            assert_eq!(uniforms[60..63], original.mandelbulb_parameters.values());
            assert_eq!(uniforms[64..80], original.menger_parameters.uniforms());
            assert_eq!(uniforms[80..83], original.background_color);
            assert_eq!(uniforms[96..98], [0.625, 1.0]);
            assert_eq!(uniforms[100], 0.375);
        }
        assert!((loaded.effects.aperture - original.effects.aperture).abs() < 0.000001);
        loaded.fractal = super::Fractal::Mandelbox;
        loaded.mutation.formula = Some(super::Fractal::Menger);
        assert!(loaded.save_preset(path));
        let mut restored = super::App::default();
        restored.load_preset(path);
        assert_eq!(restored.mutation.selected(restored.fractal), Some(super::Fractal::Menger));
        assert_eq!(restored.mutation, loaded.mutation);

        for (base, secondary) in [
            (super::Fractal::Kifs, super::Fractal::AmazingSurface),
            (super::Fractal::AmazingSurface, super::Fractal::Kifs),
            (super::Fractal::PseudoKleinian, super::Fractal::Kifs),
            (super::Fractal::AmazingSurface, super::Fractal::PseudoKleinian),
        ] {
            restored.fractal = base;
            restored.mutation.formula = Some(secondary);
            restored.fold_parameters.kifs[2] = 36.0;
            restored.fold_parameters.amazing_surface[1] = 1.2;
            restored.fold_parameters.kifs[0] = 14.0;
            restored.mutation.fold_parameters.kifs[0] = 14.0;
            restored.mutation.fold_parameters.amazing_surface[0] = -1.55;
            restored.fold_parameters.pseudo_kleinian[2] = 0.26;
            restored.mutation.fold_parameters.pseudo_kleinian[3] = -0.17;
            assert!(restored.save_preset(path));
            loaded.load_preset(path);
            assert_eq!(loaded.preset_status, "Preset loaded");
            assert_eq!(loaded.fractal, base);
            assert_eq!(loaded.mutation.selected(base), Some(secondary));
            assert_eq!(loaded.fold_parameters, restored.fold_parameters);
            assert_eq!(loaded.mutation.fold_parameters, restored.mutation.fold_parameters);
        }
        std::fs::write(path, lines[..95].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.fold_parameters.kifs, original.fold_parameters.kifs);
        assert_eq!(loaded.fold_parameters.amazing_surface, original.fold_parameters.amazing_surface);
        assert_eq!(loaded.fold_parameters.pseudo_kleinian, super::FoldParameters::LEGACY_KLEINIAN_DEFAULTS);
        assert_eq!(loaded.mutation.values(), original.mutation.values());
        assert_eq!(loaded.mutation.fold_parameters.kifs, original.mutation.fold_parameters.kifs);
        assert_eq!(loaded.mutation.fold_parameters.amazing_surface, original.mutation.fold_parameters.amazing_surface);
        assert_eq!(loaded.mutation.fold_parameters.pseudo_kleinian, super::FoldParameters::LEGACY_KLEINIAN_DEFAULTS);
        restored.fractal = super::Fractal::PseudoKleinian;
        restored.lighting.enabled = true;
        restored.lighting.shadows = false;
        restored.lighting.environment_color = [1.0; 3];
        restored.lighting.angular_radius = 0.0;
        restored.select_material(super::DIFFUSE_GRAY_MATERIAL);
        restored.extra_mutations = vec![
            super::MutationStage {
                formula: Some(super::Fractal::Kifs), weight: 0.25,
                ..Default::default()
            },
            super::MutationStage {
                formula: Some(super::Fractal::AmazingSurface), weight: 0.5,
                ..Default::default()
            },
        ];
        restored.extra_mutations[0].fold_parameters.kifs[0] = 14.0;
        assert!(restored.save_preset(path));
        loaded.load_preset(path);
        assert_eq!(loaded.fractal, super::Fractal::PseudoKleinian);
        assert_eq!(loaded.extra_mutations, restored.extra_mutations);
        let uniforms = loaded.camera_uniforms([1920.0, 1080.0], None, [0.0; 4], false);
        assert_eq!(uniforms[96..98], [1.0 / 1.75, 2.0]);
        assert_eq!(uniforms[100], 0.25 / 1.75);
        assert_eq!(uniforms[120], 0.5 / 1.75);
        let previous = restored.preset_text();
        std::fs::write(path, previous.lines().take(super::CASCADE_PRESET_FIELDS).collect::<Vec<_>>().join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.camera_uniforms([1920.0, 1080.0], None, [0.0; 4], false), uniforms);

        restored.base_weight = 0.5;
        restored.mutation.formula = Some(super::Fractal::Menger);
        restored.mutation.weight = 1.0;
        restored.extra_mutations[0].weight = 1.0;
        restored.extra_mutations[1].weight = 0.5;
        assert!(restored.save_preset(path));
        loaded.load_preset(path);
        assert_eq!(loaded.base_weight, 0.5);
        let relative = loaded.camera_uniforms([1920.0, 1080.0], None, [0.0; 4], false);
        assert_eq!(relative[96..98], [1.0 / 6.0, 3.0]);
        assert_eq!([relative[100], relative[120], relative[140]], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0]);
        let mut former_mode = restored.preset_values();
        former_mode[super::CASCADE_PRESET_FIELDS] = 0.0;
        loaded.restore_scene(&former_mode);
        assert_eq!(loaded.camera_uniforms([1920.0, 1080.0], None, [0.0; 4], false), relative);

        std::fs::write(path, previous.lines().take(super::BASE_PRESET_FIELDS).collect::<Vec<_>>().join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.fractal, super::Fractal::PseudoKleinian);
        assert!(loaded.extra_mutations.is_empty());
        std::fs::write(path, lines[..79].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.fractal, original.fractal);
        let legacy_folds = super::FoldParameters::from_preset(&[], &[]);
        assert_eq!(loaded.fold_parameters, legacy_folds);
        assert_eq!(loaded.mutation.fold_parameters, legacy_folds);
        assert_eq!(loaded.mutation.values(), original.mutation.values());

        std::fs::write(path, lines[..74].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.mutation.formula, Some(super::Fractal::Mandelbulb));
        assert_eq!(loaded.mutation.menger_parameters, super::MengerParameters::default());
        std::fs::write(path, lines[..63].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.background_color, original.background_color);
        assert_eq!(loaded.mutation.formula, None);
        std::fs::write(path, lines[..60].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.menger_parameters, original.menger_parameters);
        assert_eq!(loaded.background_color, super::DEFAULT_BACKGROUND);
        std::fs::write(path, lines[..58].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.fractal, super::Fractal::Menger);
        assert_eq!(loaded.menger_parameters.rotation, [0.0, 0.0, 138.0]);
        let mut previous_preset = lines[..55].to_vec();
        previous_preset[9] = "1";
        std::fs::write(path, previous_preset.join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.fractal, super::Fractal::Mandelbox);
        assert_eq!(loaded.menger_parameters, super::MengerParameters::default());
        assert_eq!(loaded.mandelbulb_parameters, original.mandelbulb_parameters);
        std::fs::write(path, previous_preset[..52].join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.mandelbox_parameters, original.mandelbox_parameters);
        assert_eq!(loaded.mandelbulb_parameters, super::MandelbulbParameters::default());
        let mut legacy = lines[..45].to_vec();
        legacy[44] = "1.6";
        std::fs::write(path, legacy.join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.preset_status, "Preset loaded");
        assert_eq!(loaded.camera.look_roll, 0.0);
        assert_eq!(loaded.effects.preset(), original.effects.preset());
        assert_eq!(loaded.contrast, 1.6);
        assert_eq!(loaded.mandelbox_parameters.scale, super::MandelboxParameters::LEGACY_SCALE);
        legacy[43] = "0";
        std::fs::write(path, legacy.join("\n")).unwrap();
        loaded.load_preset(path);
        assert_eq!(loaded.contrast, 1.0);
        let mut old_lines = lines[..35].to_vec();
        old_lines.insert(21, lines[34]);
        std::fs::write(path, old_lines.join("\n")).unwrap();
        let mut recovered = super::App::default();
        recovered.load_preset(path);
        assert_eq!(recovered.preset_status, "Preset loaded");
        assert_eq!(recovered.camera.roll, original.camera.roll);
        assert_eq!(recovered.lighting.color, original.lighting.color);
        assert_eq!(recovered.still_samples, 64);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fractal_shader_parses_and_validates() {
        for source in [super::FRACTAL_SHADER, include_str!("viewport_metrics.wgsl")] {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        }
    }

    #[test]
    fn progressive_pipelines_validate_on_gpu() {
        check_progressive_pipelines(false);
    }

    #[test]
    #[ignore = "GPU benchmark with multiple scene and resolution combinations"]
    fn benchmark_adaptive_viewport() {
        check_progressive_pipelines(true);
    }

    fn check_progressive_pipelines(benchmark: bool) {
        let instance = if benchmark {
            wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: wgpu::Backends::VULKAN, ..Default::default() })
        } else {
            wgpu::Instance::default()
        };
        let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("No GPU adapter available; skipping GPU validation");
            return;
        };
        let required_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        if benchmark {
            println!("Benchmark adapter: {:?}; timestamp query: {}", adapter.get_info(),
                required_features.contains(wgpu::Features::TIMESTAMP_QUERY));
            assert!(required_features.contains(wgpu::Features::TIMESTAMP_QUERY), "GPU timestamps unavailable");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features, ..Default::default() }, None)).expect("GPU device");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Fractal test"), source: wgpu::ShaderSource::Wgsl(super::FRACTAL_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Progress test"), layout: None,
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs_progress"), compilation_options: Default::default(), targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float, blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Constant, dst_factor: wgpu::BlendFactor::OneMinusConstant, operation: wgpu::BlendOperation::Add },
                    alpha: wgpu::BlendComponent::REPLACE,
                }), write_mask: wgpu::ColorWrites::ALL,
            })] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Composite test"), source: wgpu::ShaderSource::Wgsl(super::COMPOSITE_SHADER.into()),
        });
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Composite test"), layout: None,
            vertex: wgpu::VertexState { module: &composite, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: &composite, entry_point: Some("fs_main"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::Bgra8UnormSrgb.into())] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let camera = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: super::CAMERA_UNIFORM_BYTES, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let (texture, bind) = super::Renderer::create_accumulation(&device, &composite_pipeline, &camera, 8, 8);
        let mut values = [0.0f32; super::CAMERA_UNIFORM_FLOATS];
        values[2] = 3.2; values[3] = 1.0; values[8] = 8.0; values[10] = 8.0; values[17] = 0.002; values[18] = 1.0; values[20] = 2.0;
        let environment = super::hdri::upload(&device, &queue, 1, 1, &[0.0; 3]);
        let blue_noise = super::pixel_sampling::blue_noise_view(&device, &queue);
        let (focus_pipeline, focus_bind) = super::Renderer::create_focus_pipeline(&device, &shader, &camera, &environment, &blue_noise);
        let depth = super::Renderer::pick_focus(&device, &queue, &camera, &focus_pipeline, &focus_bind, values, [0.0, 0.0]);
        assert!(matches!(depth, Some(d) if d > 1.0 && d < 3.2), "unexpected focus depth: {depth:?}");
        assert!(super::Renderer::pick_focus(&device, &queue, &camera, &focus_pipeline, &focus_bind, values, [1.0, 1.0]).is_none());
        let mut inside = values;
        inside[2] = 0.0;
        let exit = super::Renderer::pick_focus(&device, &queue, &camera, &focus_pipeline, &focus_bind, inside, [0.0, 0.0]);
        assert!(matches!(exit, Some(d) if d > 0.25 && d < 2.0), "unexpected interior exit depth: {exit:?}");
        let mut menger = values;
        menger[9] = super::Fractal::Menger.value();
        let parameters = super::MengerParameters { depth: 1, ..Default::default() };
        menger[64..80].copy_from_slice(&parameters.uniforms());
        let focus = |view| super::Renderer::pick_focus(&device, &queue, &camera, &focus_pipeline, &focus_bind, view, [0.0, 0.0]);
        assert!(focus(menger).is_none(), "the central Menger tunnel must be open");
        menger[0..3].copy_from_slice(&[0.7, 0.7, 3.2]);
        let front = focus(menger);
        assert!(matches!(front, Some(d) if (d - 2.2).abs() < 0.002), "unexpected Menger front depth: {front:?}");
        menger[0..3].copy_from_slice(&[0.0, 0.5, 0.0]);
        menger[4] = std::f32::consts::FRAC_PI_2;
        let wall = focus(menger);
        assert!(matches!(wall, Some(d) if (d - 1.0 / 3.0).abs() < 0.002), "unexpected Menger tunnel wall: {wall:?}");
        values[22] = 1.0;
        values[40..43].copy_from_slice(&[1.8, 1.5, 0.025]);
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        queue.write_buffer(&camera, 0, &bytes);
        let camera_bind = super::hdri::bind(&device, &pipeline.get_bind_group_layout(0), &camera, &environment, &blue_noise);
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: None, size: wgpu::Extent3d { width: 8, height: 8, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        for index in 0..2 {
            let view = texture.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: if index == 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline); pass.set_bind_group(0, &camera_bind, &[]);
            let weight = 1.0 / (index + 1) as f64;
            pass.set_blend_constant(wgpu::Color { r: weight, g: weight, b: weight, a: 1.0 });
            pass.draw(0..3, 0..1);
        }
        {
            let view = output.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None, color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            pass.set_pipeline(&composite_pipeline); pass.set_bind_group(0, &bind, &[]); pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        let _ = device.poll(wgpu::Maintain::Wait);
        if required_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            let sizes = if benchmark && std::env::var_os("MENGERVIBES_BENCHMARK_1080P").is_some() {
                [1920u32, 1080u32]
            } else {
                [640, 360]
            };
            let (preview, _) = super::Renderer::create_accumulation(&device, &composite_pipeline, &camera, sizes[0], sizes[1]);
            let view = preview.create_view(&Default::default());
            let timing = super::adaptive_viewport::GpuTiming::new(&device, &queue);
            let measure = |app: &super::App, divisor: u32, adaptive: bool| {
                let [width, height] = sizes.map(|side| side.div_ceil(divisor));
                let mut uniforms = app.camera_uniforms(sizes.map(|side| side as f32), None, [0.0; 4], false);
                if adaptive || divisor != 1 {
                    uniforms[53] = width as f32;
                    uniforms[54] = height as f32;
                }
                if adaptive {
                    uniforms[super::FULL_PRECISION_PREVIEW_UNIFORM] = sizes[1] as f32;
                    uniforms[18] = (app.quality * 1.25).min(4.0).max(app.quality);
                } else {
                    uniforms[10] /= divisor as f32;
                }
                let bytes: Vec<u8> = uniforms.iter().flat_map(|value| value.to_ne_bytes()).collect();
                queue.write_buffer(&camera, 0, &bytes);
                let mut encoder = device.create_command_encoder(&Default::default());
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Adaptive navigation comparison"), color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                    })], depth_stencil_attachment: None, timestamp_writes: Some(timing.writes()), occlusion_query_set: None,
                });
                pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &camera_bind, &[]);
                pass.set_blend_constant(wgpu::Color::WHITE);
                pass.draw(0..3, 0..1);
                drop(pass);
                timing.resolve(&mut encoder, &device, &preview, &camera);
                queue.submit(Some(encoder.finish()));
                let pending = super::adaptive_viewport::InFlight::submit(&queue, Some(&timing), Some(super::adaptive_viewport::PreviewWork {
                    size: [width, height], divisor, samples: 1,
                }));
                let _ = device.poll(wgpu::Maintain::Wait);
                let stats = pending.poll(Some(&timing)).flatten().expect("completed benchmark pass");
                assert!(stats.gpu && stats.ms > 0.0);
                assert!(stats.valid_fraction.is_some_and(|fraction| (0.0..=1.0).contains(&fraction)));
                stats
            };
            let mut app = super::App::default();
            app.lighting.shadows = false;
            let stats = measure(&app, 4, true);
            assert_eq!(stats.size, sizes.map(|side| side.div_ceil(4)));
            assert!(stats.valid_fraction.unwrap() > 0.9);
            if benchmark {
                use super::{Camera, Fractal, MutationStage};
                println!("{}×{} target; 1 primary ray/pixel; AO and shadows off", sizes[0], sizes[1]);
                for (name, base, position, mutations, path_traced) in [
                    ("Menger exterior", Fractal::Menger, None, &[][..], false),
                    ("Mandelbox exterior", Fractal::Mandelbox, None, &[][..], false),
                    ("Bulb exterior", Fractal::Mandelbulb, None, &[][..], false),
                    ("Bulb interior", Fractal::Mandelbulb, Some([0.0; 3]), &[][..], false),
                    ("Menger tunnel", Fractal::Menger, Some([0.0, 0.5, 0.0]), &[][..], false),
                    ("Bulb + Box", Fractal::Mandelbulb, None, &[Fractal::Mandelbox][..], false),
                    ("Bulb + three fields", Fractal::Mandelbulb, None, &[Fractal::Mandelbox, Fractal::AmazingSurface, Fractal::Menger][..], false),
                    ("Interior + paths", Fractal::Mandelbulb, Some([0.0; 3]), &[Fractal::Mandelbox, Fractal::Menger][..], true),
                ] {
                    app.fractal = base;
                    app.lighting.path_traced = path_traced;
                    app.camera = Camera::framed_view(base);
                    if let Some(position) = position { app.camera.position = position; }
                    app.mutation = MutationStage::default();
                    app.extra_mutations.clear();
                    for &formula in mutations {
                        app.extra_mutations.push(MutationStage { formula: Some(formula), ..Default::default() });
                    }
                    let legacy = if mutations.is_empty() { 1 } else { 4 };
                    for (label, divisor, adaptive) in [("Legacy", legacy, false), ("Quarter", 4, true), ("Half", 2, true), ("Full", 1, true)] {
                        let mut samples = Vec::new();
                        let yaw = app.camera.yaw;
                        for offset in [0.0, 0.26, -0.31, 0.0, 0.26, -0.31] {
                            app.camera.yaw = yaw + offset;
                            samples.push(measure(&app, divisor, adaptive));
                        }
                        app.camera.yaw = yaw;
                        samples.drain(..3);
                        samples.sort_by(|a, b| a.ms.total_cmp(&b.ms));
                        let median = &samples[1];
                        let valid = median.valid_fraction.unwrap();
                        println!("{name:22} {label:7} {:6.2} ms {:6.1}% valid {:5.2} M valid primary rays/s | {:.3} ms GPU metrics | {:.2} ms completion",
                            median.ms, valid * 100.0, valid * median.rays as f64 / (median.ms * 1000.0),
                            median.metrics_ms.unwrap_or_default(), median.completion_ms);
                    }
                }
            }
        }
    }
}
