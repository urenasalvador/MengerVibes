mod export;
mod camera_effects;
mod history;
mod preset_gallery;
mod startup;
use camera_effects::CameraEffects;
use std::{collections::HashSet, sync::Arc, time::{Duration, Instant}};
use winit::{application::ApplicationHandler, event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent}, event_loop::{ActiveEventLoop, EventLoop, ControlFlow}, keyboard::{KeyCode, PhysicalKey}, window::{Window, WindowId}};

const APP_NAME: &str = "MengerVibes";
const MAX_MUTATIONS: usize = 8;
const EXTRA_MUTATION_UNIFORMS: usize = 20;
const MUTATION_PRESET_FIELDS: usize = 28;
const BASE_PRESET_FIELDS: usize = 103;
const CAMERA_UNIFORM_FLOATS: usize = 100 + MAX_MUTATIONS * EXTRA_MUTATION_UNIFORMS;
const CAMERA_UNIFORM_BYTES: u64 = (CAMERA_UNIFORM_FLOATS * std::mem::size_of::<f32>()) as u64;
const CONTRAST_RANGE: std::ops::RangeInclusive<f32> = 0.5..=3.0;
const CONTROLS_MIN_WIDTH: f32 = 315.0;
const CASCADE_PRESET_FIELDS: usize = BASE_PRESET_FIELDS + 1 + (MAX_MUTATIONS - 1) * MUTATION_PRESET_FIELDS;
const PRESET_VALUE_COUNT: usize = CASCADE_PRESET_FIELDS + 2;
const DEFAULT_BACKGROUND: [f32; 3] = [0.015, 0.02, 0.04];
const DETAIL_QUALITY: std::ops::RangeInclusive<f32> = 0.5..=8.0;
const SMOOTHNESS: std::ops::RangeInclusive<f32> = 0.000020..=0.2;
const FRACTAL_SHADER: &str = concat!(include_str!("contrast.wgsl"), "\n", include_str!("fractal.wgsl"));
const COMPOSITE_SHADER: &str = concat!(include_str!("contrast.wgsl"), "\n", include_str!("composite.wgsl"));
const REFINE_TILE_WIDTH: u32 = 256;
const REFINE_TILE_HEIGHT: u32 = 128;

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
    progress_bind: wgpu::BindGroup,
    composite_pipeline: wgpu::RenderPipeline,
    accumulation: wgpu::Texture,
    composite_bind: wgpu::BindGroup,
    preview: wgpu::Texture,
    preview_bind: wgpu::BindGroup,
    preview_dimensions: Option<[f32; 2]>,
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
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default(), None).await
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
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Fractal pipeline"),
            layout: None,
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
        let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera bind group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });
        stage("Compiling focus and progressive shaders");
        let (focus_pipeline, focus_bind) = Self::create_focus_pipeline(&device, &shader, &camera_buffer);
        let progress_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Progressive fractal"), layout: None,
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
        let progress_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Progressive camera"), layout: &progress_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
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
        Ok(Self { surface, device, queue, config, pipeline, camera_buffer, camera_bind, focus_pipeline, focus_bind, progress_pipeline, progress_bind, composite_pipeline, accumulation, composite_bind, preview, preview_bind, preview_dimensions: None, frame_submitted: false, egui_renderer, export_pipelines: Arc::default() })
    }

    fn create_focus_pipeline(device: &wgpu::Device, shader: &wgpu::ShaderModule, camera_buffer: &wgpu::Buffer) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
        let focus_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pick focus distance"), layout: None,
            vertex: wgpu::VertexState { module: shader, entry_point: Some("vs_main"), compilation_options: Default::default(), buffers: &[] },
            fragment: Some(wgpu::FragmentState { module: shader, entry_point: Some("fs_focus"), compilation_options: Default::default(), targets: &[Some(wgpu::TextureFormat::R32Float.into())] }),
            primitive: Default::default(), depth_stencil: None, multisample: Default::default(), multiview: None, cache: None,
        });
        let focus_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &focus_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buffer.as_entire_binding() }],
        });
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Accumulated samples"), layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: camera.as_entire_binding() },
            ],
        });
        (texture, bind)
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 { return; }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        (self.accumulation, self.composite_bind) = Self::create_accumulation(&self.device, &self.composite_pipeline, &self.camera_buffer, width, height);
        (self.preview, self.preview_bind) = Self::create_accumulation(&self.device, &self.composite_pipeline, &self.camera_buffer, width, height);
        self.preview_dimensions = None;
    }

    fn render(&mut self, mut data: [f32; CAMERA_UNIFORM_FLOATS], progress: Option<u32>, advance: bool, refine_tile: Option<u32>, preview_divisor: u32, gui: egui::FullOutput, ctx: &egui::Context, capture: bool) -> Option<Result<image::RgbaImage, String>> {
        self.frame_submitted = false;
        let preview_size = (preview_divisor > 1).then(|| [
            self.config.width.div_ceil(preview_divisor) as f32,
            self.config.height.div_ceil(preview_divisor) as f32,
        ]);
        if let Some([width, height]) = preview_size {
            data[10] *= height / self.config.height as f32;
            data[53] = width;
        }
        let columns = self.config.width.div_ceil(REFINE_TILE_WIDTH);
        let rows = self.config.height.div_ceil(REFINE_TILE_HEIGHT);
        let cached_preview = refine_tile.filter(|&tile| progress == Some(0) && tile + 1 < columns * rows)
            .and(self.preview_dimensions);
        // Composite dimensions are separate from the tracing controls. This
        // keeps the cached preview available while full-size tiles render.
        let display_size = preview_size.or(cached_preview).unwrap_or([0.0; 2]);
        data[54] = preview_size.map_or(0.0, |size| size[1]);
        data[55] = display_size[0];
        data[83] = display_size[1];
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
        if advance || preview_size.is_some() {
            let index = progress.unwrap_or(0);
            let target = if preview_size.is_some() { &self.preview } else { &self.accumulation };
            let accumulation_view = target.create_view(&Default::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render viewport sample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &accumulation_view, resolve_target: None,
                    ops: wgpu::Operations { load: if index == 0 && refine_tile.unwrap_or(0) == 0 { wgpu::LoadOp::Clear(wgpu::Color::BLACK) } else { wgpu::LoadOp::Load }, store: wgpu::StoreOp::Store },
                })], depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None,
            });
            pass.set_pipeline(&self.progress_pipeline);
            pass.set_bind_group(0, &self.progress_bind, &[]);
            let weight = 1.0 / (index + 1) as f64;
            pass.set_blend_constant(wgpu::Color { r: weight, g: weight, b: weight, a: 1.0 });
            // The still preview starts with a fresh sample after interaction.
            if let Some([width, height]) = preview_size {
                pass.set_viewport(0.0, 0.0, width, height, 0.0, 1.0);
            }
            if let Some(tile) = refine_tile {
                let x = (tile % columns) * REFINE_TILE_WIDTH;
                let y = (tile / columns) * REFINE_TILE_HEIGHT;
                pass.set_scissor_rect(x, y, REFINE_TILE_WIDTH.min(self.config.width - x), REFINE_TILE_HEIGHT.min(self.config.height - y));
            }
            pass.draw(0..3, 0..1);
        }
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
                let bind = if preview_size.is_some() || cached_preview.is_some() { &self.preview_bind } else { &self.composite_bind };
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
        self.frame_submitted = true;
        if progress.is_none() { self.preview_dimensions = preview_size; }
        for id in &gui.textures_delta.free { self.egui_renderer.free_texture(id); }
        frame.present();
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

struct Lighting { color: [f32; 3], roughness: f32, metallic: f32, light_color: [f32; 3], azimuth: f32, elevation: f32, intensity: f32, ambient: f32 }
impl Default for Lighting {
    fn default() -> Self { Self { color: [0.35, 0.62, 0.72], roughness: 0.5, metallic: 0.0, light_color: [1.0; 3], azimuth: 0.6, elevation: 0.8, intensity: 1.0, ambient: 0.25 } }
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
}

impl Fractal {
    const ALL: [Self; 6] = [Self::Mandelbulb, Self::Mandelbox, Self::Menger, Self::Kifs, Self::AmazingSurface, Self::PseudoKleinian];

    fn name(self) -> &'static str {
        match self {
            Self::Mandelbulb => "Mandelbulb",
            Self::Mandelbox => "Mandelbox",
            Self::Menger => "Menger",
            Self::Kifs => "KIFS",
            Self::AmazingSurface => "Amazing Surface",
            Self::PseudoKleinian => "Pseudo-Kleinian",
        }
    }

    fn value(self) -> f32 { self as u8 as f32 }

    fn from_value(value: f32) -> Option<Self> {
        Self::ALL.into_iter().find(|formula| formula.value() == value)
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
            amazing_surface: [-1.8, 1.0, 0.5, 0.0],
            pseudo_kleinian: [0.80, 1.0, 0.12, 0.08],
        }
    }
}

impl FoldParameters {
    const KIFS_RANGES: [std::ops::RangeInclusive<f32>; 4] = [2.0..=14.0, 2.0..=3.5, -180.0..=180.0, 0.3..=1.5];
    const SURFACE_RANGES: [std::ops::RangeInclusive<f32>; 4] = [-3.0..=-1.2, 0.25..=2.0, 0.2..=0.95, -0.2..=0.2];
    const KLEINIAN_RANGES: [std::ops::RangeInclusive<f32>; 4] = [0.2..=1.5, 0.35..=2.0, 0.02..=0.4, -0.5..=0.5];

    fn ui(&mut self, ui: &mut egui::Ui, formula: Fractal) {
        let default = Self::default();
        match formula {
            Fractal::Kifs => {
                ui.add(egui::Slider::new(&mut self.kifs[0], Self::KIFS_RANGES[0].clone()).step_by(1.0).text("Depth"));
                precision_slider(ui, &mut self.kifs[1], Self::KIFS_RANGES[1].clone(), "Scale", false, false);
                precision_slider(ui, &mut self.kifs[2], Self::KIFS_RANGES[2].clone(), "Fold rotation Y", false, true);
                precision_slider(ui, &mut self.kifs[3], Self::KIFS_RANGES[3].clone(), "Seed radius", false, false);
                if ui.button("Reset KIFS parameters").clicked() { self.kifs = default.kifs; }
            }
            Fractal::AmazingSurface => {
                for (i, label) in ["Scale", "Fold limit", "Inner radius", "Scale variation"].into_iter().enumerate() {
                    precision_slider(ui, &mut self.amazing_surface[i], Self::SURFACE_RANGES[i].clone(), label, false, false);
                }
                if ui.button("Reset Amazing Surface parameters").clicked() { self.amazing_surface = default.amazing_surface; }
            }
            Fractal::PseudoKleinian => {
                for (i, label) in ["Fold limit", "Inversion radius", "Wall thickness", "Offset Z"].into_iter().enumerate() {
                    precision_slider(ui, &mut self.pseudo_kleinian[i], Self::KLEINIAN_RANGES[i].clone(), label, false, false);
                }
                if ui.button("Reset Pseudo-Kleinian parameters").clicked() { self.pseudo_kleinian = default.pseudo_kleinian; }
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
        let mut settings = Self::default();
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
    const DEPTH: std::ops::RangeInclusive<u32> = 1..=14;
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
        Self { scale: -1.5, fold_limit: 1.0, inner_radius: 0.5, outer_radius: 1.0 }
    }
}

impl MandelboxParameters {
    const SCALE: std::ops::RangeInclusive<f32> = -2.5..=-1.15;
    const FOLD_LIMIT: std::ops::RangeInclusive<f32> = 0.25..=2.0;
    const MIN_INNER_RADIUS: f32 = 0.2;
    const MIN_OUTER_RADIUS: f32 = 0.5;
    const MAX_OUTER_RADIUS: f32 = 1.5;

    fn ui(&mut self, ui: &mut egui::Ui) {
        precision_slider(ui, &mut self.scale, Self::SCALE, "Scale", false, false)
            .on_hover_text("Changes the size ratio between successive levels of the fractal.");
        precision_slider(ui, &mut self.fold_limit, Self::FOLD_LIMIT, "Box fold limit", false, false)
            .on_hover_text("Sets how far from the center the box fold starts.");
        precision_slider(ui, &mut self.inner_radius, Self::MIN_INNER_RADIUS..=self.outer_radius, "Inner radius", false, false)
            .on_hover_text("Sets the inner boundary of the sphere fold.");
        precision_slider(ui, &mut self.outer_radius, self.inner_radius.max(Self::MIN_OUTER_RADIUS)..=Self::MAX_OUTER_RADIUS, "Outer radius", false, false)
            .on_hover_text("Sets the outer boundary of the sphere fold.");
        if ui.button("Reset Mandelbox parameters").clicked() {
            *self = Self::default();
        }
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
            return Self::default();
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
            ui.collapsing("Mutation parameters", |ui| match formula {
                Fractal::Mandelbox => self.box_parameters.ui(ui),
                Fractal::Mandelbulb => {
                    precision_slider(ui, &mut self.bulb_power, 2.0..=23.0, "Power", false, false);
                    self.bulb_parameters.ui(ui);
                    if ui.button("Reset mutation parameters").clicked() {
                        self.bulb_power = 8.0;
                        self.bulb_parameters = MandelbulbParameters::default();
                    }
                }
                Fractal::Menger => self.menger_parameters.ui(ui),
                Fractal::Kifs | Fractal::AmazingSurface | Fractal::PseudoKleinian => self.fold_parameters.ui(ui, formula),
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
        }
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

struct OutputSettings { width: u32, height: u32, samples: u32, status: String, progress: f32 }
impl Default for OutputSettings { fn default() -> Self { Self { width: 1920, height: 1080, samples: 16, status: String::new(), progress: 0.0 } } }
struct RenderJob { receiver: std::sync::mpsc::Receiver<export::Update>, cancel: Arc<std::sync::atomic::AtomicBool> }

#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
enum ControlsTab {
    #[default]
    Formula,
    Surface,
    Material,
    View,
    Presets,
}

impl ControlsTab {
    const ALL: [Self; 5] = [Self::Formula, Self::Surface, Self::Material, Self::View, Self::Presets];

    fn label(self) -> &'static str {
        match self {
            Self::Formula => "Formula",
            Self::Surface => "Surface",
            Self::Material => "Material",
            Self::View => "View",
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
            Self::View => {
                outline(vec![point(-8.0, 0.0), point(-4.0, -4.0), point(4.0, -4.0), point(8.0, 0.0), point(4.0, 4.0), point(-4.0, 4.0)]);
                painter.circle_filled(center, 2.1, color);
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
struct App { startup: startup::Startup, event_proxy: Option<winit::event_loop::EventLoopProxy<startup::Event>>, window: Option<Arc<Window>>, renderer: Option<Renderer>, camera: Camera, keys: HashSet<KeyCode>, looking: bool, last_frame: Instant, slice: Option<f32>, power: f32, fractal: Fractal, mutation: MutationStage, extra_mutations: Vec<MutationStage>, base_weight: f32, menger_parameters: MengerParameters, mandelbox_parameters: MandelboxParameters, mandelbulb_parameters: MandelbulbParameters, fold_parameters: FoldParameters, diagnostic: bool, material: u32, matcap_rotation: f32, ao: bool, ao_strength: f32, ao_radius: f32, exposure: f32, contrast: f32, normal_smoothing: f32, quality: f32, antialias: bool, distant_filter: f32, lighting: Lighting, refine_idle: bool, refine_target: u32, progress_samples: u32, progress_tile: u32, motion_samples: u32, motion_recovery: u32, idle_deadline: Option<Instant>, idle_ready: bool, last_scene: Vec<f32>, egui_ctx: egui::Context, egui_state: Option<egui_winit::State>, preset_status: String, presets: preset_gallery::Gallery, capture_slot: Option<u64>, output: OutputSettings, render_job: Option<RenderJob>, start_export: bool, left_open: bool, controls_tab: ControlsTab, right_open: bool, effects: CameraEffects, background_color: [f32; 3], show_frame: bool, frame: Option<egui::Rect>, picking_focus: bool, focus_point: Option<[f32; 2]>, focus_status: String, history: history::History, history_actions: Vec<history::Action>, modifiers: winit::keyboard::ModifiersState }
impl Default for App {
    fn default() -> Self {
        // Equal offsets from the three axes give the Menger cube the icon's diagonal view.
        let camera = Camera {
            position: [6.0; 3],
            yaw: -std::f32::consts::FRAC_PI_4,
            pitch: -(1.0 / 2.0_f32.sqrt()).atan(),
            ..Camera::default()
        };
        let effects = CameraEffects { focal: 60.0, ..CameraEffects::default() };
        Self { startup: startup::Startup::default(), event_proxy: None, window: None, renderer: None, camera, keys: HashSet::new(), looking: false, last_frame: Instant::now(), slice: None, power: 8.0, fractal: Fractal::default(), mutation: MutationStage::default(), extra_mutations: Vec::new(), base_weight: 1.0, menger_parameters: MengerParameters::default(), mandelbox_parameters: MandelboxParameters::default(), mandelbulb_parameters: MandelbulbParameters::default(), fold_parameters: FoldParameters::default(), diagnostic: false, material: 0, matcap_rotation: 0.0, ao: false, ao_strength: 0.5, ao_radius: 0.048, exposure: 0.0, contrast: 1.0, normal_smoothing: 0.002, quality: 1.0, antialias: false, distant_filter: 2.0, lighting: Lighting::default(), refine_idle: true, refine_target: 16, progress_samples: 0, progress_tile: 0, motion_samples: 1, motion_recovery: 0, idle_deadline: None, idle_ready: false, last_scene: Vec::new(), egui_ctx: egui::Context::default(), egui_state: None, preset_status: String::new(), presets: preset_gallery::Gallery::default(), capture_slot: None, output: OutputSettings::default(), render_job: None, start_export: false, left_open: true, controls_tab: ControlsTab::default(), right_open: true, effects, background_color: DEFAULT_BACKGROUND, show_frame: true, frame: None, picking_focus: false, focus_point: None, focus_status: String::new(), history: history::History::default(), history_actions: Vec::new(), modifiers: winit::keyboard::ModifiersState::empty() }
    }
}
impl App {
    fn viewport_dof(&self) -> bool {
        self.effects.dof_enabled && self.effects.aperture > 0.0 && !self.diagnostic && !self.section_active()
    }

    fn section_active(&self) -> bool { self.fractal == Fractal::Mandelbulb && self.slice.is_some() }

    fn fractal_title(&self) -> String {
        format!("{APP_NAME} - {} | 1-6: Switch fractal", self.fractal.name())
    }

    fn still_sample_target(&self) -> u32 {
        if self.diagnostic { 1 }
        else if self.refine_idle { self.refine_target }
        else if self.viewport_dof() { 16 }
        else { 1 }
    }

    fn ui(&mut self, ctx: &egui::Context) {
        if self.right_open {
            egui::SidePanel::right("render_output").default_width(255.0).resizable(true).show(ctx, |ui| {
                ui.horizontal(|ui| { ui.heading("Render output"); if ui.small_button("▶").on_hover_text("Hide render panel").clicked() { self.right_open = false; } });
                ui.separator();
                ui.add_enabled_ui(self.render_job.is_none(), |ui| {
                    ui.horizontal(|ui| { ui.label("Width"); ui.add(egui::DragValue::new(&mut self.output.width).range(64..=8192).speed(4)); ui.label("px"); });
                    ui.horizontal(|ui| { ui.label("Height"); ui.add(egui::DragValue::new(&mut self.output.height).range(64..=8192).speed(4)); ui.label("px"); });
                    egui::ComboBox::from_label("Samples").selected_text(self.output.samples.to_string()).show_ui(ui, |ui| { for n in [1, 4, 8, 16, 32, 64, 128] { ui.selectable_value(&mut self.output.samples, n, n.to_string()); } });
                    ui.small("Uses the viewport's detail settings.");
                    if ui.button("Render PNG").clicked() { self.start_export = true; }
                });
                if let Some(job) = &self.render_job {
                    ui.add(egui::ProgressBar::new(self.output.progress).show_percentage());
                    if ui.button("Cancel render").clicked() { job.cancel.store(true, std::sync::atomic::Ordering::Relaxed); }
                }
                ui.label(&self.output.status);
                ui.small("Images are saved in the renders folder.");
            });
        } else {
            egui::Area::new("open_render".into()).anchor(egui::Align2::RIGHT_TOP, [-5.0, 5.0]).show(ctx, |ui| { if ui.small_button("◀").on_hover_text("Show render panel").clicked() { self.right_open = true; } });
        }
        if self.left_open {
            egui::SidePanel::left("controls")
                .default_width(320.0)
                .min_width(CONTROLS_MIN_WIDTH)
                .resizable(true)
                .show(ctx, |ui| {
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
                    ui.horizontal_top(|ui| {
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
                                    ControlsTab::View => self.view_ui(ui),
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
    }
    fn formula_ui(&mut self, ui: &mut egui::Ui) {
        egui::ComboBox::from_label("Fractal")
            .selected_text(self.fractal.name())
            .show_ui(ui, |ui| {
                for fractal in Fractal::ALL {
                    ui.selectable_value(&mut self.fractal, fractal, fractal.name());
                }
            });
        match self.fractal {
            Fractal::Mandelbulb => {
                precision_slider(ui, &mut self.power, 2.0..=23.0, "Power", false, false);
                self.mandelbulb_parameters.ui(ui);
                if ui.button("Reset Mandelbulb parameters").clicked() {
                    self.power = 8.0;
                    self.mandelbulb_parameters = MandelbulbParameters::default();
                }
                let mut enabled = self.slice.is_some();
                if ui.checkbox(&mut enabled, "Section").changed() { self.slice = enabled.then_some(0.0); }
                if let Some(z) = &mut self.slice { precision_slider(ui, z, -2.0..=2.0, "Section Z", false, false); }
            }
            Fractal::Mandelbox => self.mandelbox_parameters.ui(ui),
            Fractal::Menger => self.menger_parameters.ui(ui),
            Fractal::Kifs | Fractal::AmazingSurface | Fractal::PseudoKleinian => self.fold_parameters.ui(ui, self.fractal),
        }
        ui.separator();
        self.mutations_ui(ui);
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
        ui.checkbox(&mut self.antialias, "Antialiasing (4 samples)");
        ui.checkbox(&mut self.refine_idle, "Progressive AA when still");
        if self.refine_idle {
            egui::ComboBox::from_label("Still samples").selected_text(self.refine_target.to_string()).show_ui(ui, |ui| {
                for count in [8, 16, 32, 64] { ui.selectable_value(&mut self.refine_target, count, count.to_string()); }
            });
        }
        let display_target = self.still_sample_target();
        // Keep the status row in place when the aperture crosses zero.
        ui.small(if display_target == 1 {
            if self.progress_samples > 0 { "Still preview: ready" } else { "Still preview: pending" }.into()
        } else if self.idle_ready {
            format!("Samples: {} / {}", self.progress_samples, display_target)
        } else {
            format!("Samples: - / {}", display_target)
        });
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
        egui::ComboBox::from_label("Material").selected_text(["Original", "Pearl", "Copper", "Nebula", "Custom", "Gold", "Steel", "Clay", "Jade", "Aurora"][self.material as usize]).show_ui(ui, |ui| { for (i, name) in ["Original", "Pearl", "Copper", "Nebula", "Custom", "Gold", "Steel", "Clay", "Jade", "Aurora"].iter().enumerate() { ui.selectable_value(&mut self.material, i as u32, *name); } });
        if self.material != 0 && self.material != 4 {
            precision_slider(ui, &mut self.matcap_rotation, -180.0..=180.0, "Matcap Rotation", false, true)
                .on_hover_text("Rotate around the view Z axis. Hold Shift for fine adjustment.");
            if ui.small_button("Reset Matcap Rotation").clicked() {
                self.matcap_rotation = 0.0;
            }
        }
        precision_slider(ui, &mut self.exposure, -5.0..=5.0, "Exposure", false, false);
        precision_slider(ui, &mut self.contrast, CONTRAST_RANGE, "Contrast", false, false)
            .on_hover_text("1.0 preserves the original contrast.");
        if self.material == 4 {
            ui.separator(); ui.heading("Custom material");
            ui.horizontal(|ui| { ui.label("Base color"); ui.color_edit_button_rgb(&mut self.lighting.color); });
            precision_slider(ui, &mut self.lighting.roughness, 0.05..=1.0, "Roughness", false, false);
            precision_slider(ui, &mut self.lighting.metallic, 0.0..=1.0, "Metallic", false, false);
            ui.separator(); ui.heading("Lighting");
            ui.horizontal(|ui| { ui.label("Light color"); ui.color_edit_button_rgb(&mut self.lighting.light_color); });
            precision_slider(ui, &mut self.lighting.azimuth, -std::f32::consts::PI..=std::f32::consts::PI, "Direction", false, false);
            precision_slider(ui, &mut self.lighting.elevation, -1.55..=1.55, "Elevation", false, false);
            precision_slider(ui, &mut self.lighting.intensity, 0.0..=4.0, "Intensity", false, false);
            precision_slider(ui, &mut self.lighting.ambient, 0.0..=2.0, "Ambient", false, false);
        }
        ui.separator(); ui.heading("Ambient occlusion");
        ui.checkbox(&mut self.ao, "Enabled");
        ui.add_enabled_ui(self.ao, |ui| { precision_slider(ui, &mut self.ao_strength, 0.0..=2.0, "Strength", false, false); precision_slider(ui, &mut self.ao_radius, 0.0001..=2.0, "Radius", true, false); });
    }

    fn view_ui(&mut self, ui: &mut egui::Ui) {
        let mut focal = self.effects.focal;
        precision_slider(ui, &mut focal, 8.0..=120.0, "Focal length (mm)", false, false);
        if focal != self.effects.focal {
            self.effects.set_focal(focal);
        }
        ui.checkbox(&mut self.show_frame, "Show render frame");
        ui.horizontal(|ui| {
            ui.label("Background color");
            ui.color_edit_button_rgb(&mut self.background_color);
        });
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
                if self.effects.aperture > 0.0 && self.output.samples == 1 {
                    ui.small("Select 4 or more render samples for depth of field.");
                }
                precision_slider(ui, &mut self.effects.focus, 0.0001..=30.0, "Focus distance", true, false);
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
        if ui.button("Reset camera").clicked() { self.camera = Camera::default(); }
        ui.small("WASD: move · E / Q: up / down\nRight mouse: look · Wheel: speed");
        ui.separator();
        ui.heading("Fog");
        precision_slider(ui, &mut self.effects.fog_density, 0.0..=20.0, "Density", false, false);
        precision_slider(ui, &mut self.effects.fog_start, 0.0..=30.0, "Start distance", true, false)
            .on_hover_text("Distance from the camera, in scene units. Hold Shift for fine adjustment.");
        ui.horizontal(|ui| {
            ui.label("Color");
            ui.color_edit_button_rgb(&mut self.effects.fog_color);
        });
    }

    fn preset_action(&mut self, action: preset_gallery::Action) {
        use preset_gallery::Action;
        match action {
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
            painter.rect_stroke(frame, 0.0, egui::Stroke::new(1.0_f32, egui::Color32::from_gray(180)), egui::StrokeKind::Inside);
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
        let refined = export || progress.is_some();
        let quality = if refined { (self.quality * 1.25).min(4.0).max(self.quality) } else { self.quality };
        let grid_aa = self.antialias && (progress.is_none() || self.still_sample_target() == 1);
        let base = [
            c.position[0], c.position[1], c.position[2], size[0] / size[1].max(1.0),
            c.yaw, c.pitch, flag(self.section_active()), self.slice.unwrap_or(0.0),
            self.power, self.fractal.value(), size[1], flag(self.diagnostic && !export),
            self.material as f32, flag(self.ao), self.ao_strength, self.ao_radius,
            self.exposure, self.normal_smoothing, quality, flag(grid_aa),
            self.distant_filter, c.view_roll(), progress.unwrap_or(0) as f32, flag(refined),
            a.color[0], a.color[1], a.color[2], a.roughness,
            a.light_color[0], a.light_color[1], a.light_color[2], a.metallic,
            a.azimuth, a.elevation, a.intensity, a.ambient,
            0.0, 0.0, 0.0, 0.0,
        ];
        let mut uniforms = [0.0; CAMERA_UNIFORM_FLOATS];
        uniforms[..40].copy_from_slice(&base);
        uniforms[40..51].copy_from_slice(&self.effects.uniforms(frame));
        uniforms[51] = if self.diagnostic && !export { 1.0 } else { self.contrast };
        uniforms[52] = self.matcap_rotation.to_radians();
        uniforms[56..60].copy_from_slice(&self.mandelbox_parameters.uniforms());
        uniforms[60..63].copy_from_slice(&self.mandelbulb_parameters.values());
        uniforms[64..80].copy_from_slice(&self.menger_parameters.uniforms());
        uniforms[80..83].copy_from_slice(&self.background_color);
        uniforms[84..96].copy_from_slice(&self.fold_parameters.uniforms());
        let mut base_coefficient = 1.0;
        let mut count = 0;
        let mut fields = uniforms[100..].chunks_exact_mut(EXTRA_MUTATION_UNIFORMS);
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
            self.exposure, self.normal_smoothing, self.quality, flag(self.antialias), self.distant_filter,
            flag(self.refine_idle), light.color[0], light.color[1], light.color[2], light.roughness,
            light.metallic, light.light_color[0], light.light_color[1], light.light_color[2],
            light.azimuth, light.elevation, light.intensity, light.ambient, c.roll, self.refine_target as f32,
        ]);
        // Keep the former DOF contrast slot so existing preset fields retain their positions.
        values[35..44].copy_from_slice(&self.effects.preset());
        values[44..48].copy_from_slice(&[1.0, self.matcap_rotation, c.look_roll, self.contrast]);
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
        values
    }

    fn preset_text(&self) -> String {
        self.preset_values().into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n")
    }
    fn save_preset(&mut self, path: impl AsRef<std::path::Path>) -> bool {
        match std::fs::write(path, self.preset_text()) { Ok(()) => { self.preset_status = "Preset saved; capturing preview".into(); true }, Err(e) => { self.preset_status = format!("Save failed: {e}"); false } }
    }
    fn load_preset(&mut self, path: impl AsRef<std::path::Path>) {
        let result = std::fs::read_to_string(path).map_err(|e| e.to_string()).and_then(|text| {
            let mut v: Vec<f32> = text.lines().map(str::parse).collect::<Result<_, _>>().map_err(|_| "Invalid preset values".to_string())?;
            // A previous version wrote the sample count twice, including once
            // before lighting. Accept those files without changing them.
            if v.len() == 36 && v[21] == v[35] { v.remove(21); }
            if !matches!(v.len(), 17..=21 | 33..=35 | 43 | 45..=48 | 52 | 55 | 58 | 60 | 63 | 74 | 79 | 95 | BASE_PRESET_FIELDS | CASCADE_PRESET_FIELDS | PRESET_VALUE_COUNT) || v.iter().any(|x| !x.is_finite()) || !(0.0..=9.0).contains(&v[11]) || v[11].fract() != 0.0 { return Err("Invalid preset".into()); }
            Ok(v)
        });
        match result {
            Ok(v) => {
                self.restore_scene(&v);
                self.preset_status = "Preset loaded".into();
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
            (_, Some(Fractal::Mandelbulb)) => Fractal::Mandelbulb,
            _ => Fractal::Mandelbox,
        }; self.diagnostic = v[10] != 0.0; self.material = v[11] as u32; self.ao = v[12] != 0.0; self.ao_strength = v[13].clamp(0.0, 2.0); self.ao_radius = v[14].clamp(0.0001, 2.0); self.exposure = v[15].clamp(-5.0, 5.0); self.normal_smoothing = v[16].clamp(*SMOOTHNESS.start(), *SMOOTHNESS.end()); self.quality = v.get(17).copied().unwrap_or(1.0).clamp(*DETAIL_QUALITY.start(), *DETAIL_QUALITY.end()); self.antialias = v.get(18).copied().unwrap_or(0.0) != 0.0; self.distant_filter = v.get(19).copied().unwrap_or(2.0).clamp(0.5, 4.0); self.refine_idle = v.get(20).copied().unwrap_or(1.0) != 0.0;
        if v.len() >= 33 {
            self.lighting = Lighting { color: [v[21].clamp(0.0,1.0), v[22].clamp(0.0,1.0), v[23].clamp(0.0,1.0)], roughness: v[24].clamp(0.05,1.0), metallic: v[25].clamp(0.0,1.0), light_color: [v[26].clamp(0.0,1.0),v[27].clamp(0.0,1.0),v[28].clamp(0.0,1.0)], azimuth: v[29].clamp(-std::f32::consts::PI,std::f32::consts::PI), elevation: v[30].clamp(-1.55,1.55), intensity: v[31].clamp(0.0,4.0), ambient: v[32].clamp(0.0,2.0) };
        } else { self.lighting = Lighting::default(); }
        self.camera.roll = v.get(33).copied().unwrap_or(0.0).clamp(-std::f32::consts::PI, std::f32::consts::PI);
        let pitch_limit = if v.len() >= 47 && self.camera.roll != 0.0 { std::f32::consts::FRAC_PI_2 } else { 1.55 };
        self.camera.pitch = v[4].clamp(-pitch_limit, pitch_limit);
        self.camera.look_roll = v.get(46).copied().unwrap_or(0.0)
            .clamp(-std::f32::consts::TAU, std::f32::consts::TAU);
        if self.camera.roll == 0.0 { self.camera.look_roll = 0.0; }
        self.refine_target = match v.get(34).copied().unwrap_or(16.0) as u32 { 8 => 8, 32 => 32, 64 => 64, _ => 16 };
        self.effects = CameraEffects::from_preset(v.get(35..v.len().min(44)).unwrap_or_default());
        let legacy_contrast = if self.effects.dof_enabled && self.effects.aperture > 0.0 {
            v.get(44).copied().unwrap_or(1.12)
        } else {
            1.0
        };
        self.contrast = v.get(47).copied().unwrap_or(legacy_contrast)
            .clamp(*CONTRAST_RANGE.start(), *CONTRAST_RANGE.end());
        self.matcap_rotation = v.get(45).copied().unwrap_or(0.0).clamp(-180.0, 180.0);
        self.mandelbox_parameters = MandelboxParameters::from_preset(v.get(48..52).unwrap_or_default());
        self.mandelbulb_parameters = MandelbulbParameters::from_preset(v.get(52..55).unwrap_or_default());
        self.menger_parameters = MengerParameters::from_preset(v.get(55..v.len().min(60)).unwrap_or_default());
        self.background_color = match v.get(60..63) {
            Some([r, g, b]) => [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
            _ => DEFAULT_BACKGROUND,
        };
        self.mutation = MutationStage::from_preset(v.get(63..v.len().min(79)).unwrap_or_default());
        self.fold_parameters = FoldParameters::from_preset(v.get(79..87).unwrap_or_default(), v.get(95..99).unwrap_or_default());
        self.mutation.fold_parameters = FoldParameters::from_preset(v.get(87..95).unwrap_or_default(), v.get(99..103).unwrap_or_default());
        let count = v.get(BASE_PRESET_FIELDS).copied().unwrap_or(0.0) as usize;
        self.extra_mutations = v.get(BASE_PRESET_FIELDS + 1..).unwrap_or_default()
            .chunks_exact(MUTATION_PRESET_FIELDS).take(count.min(MAX_MUTATIONS - 1)).map(|values| {
                let mut stage = MutationStage::from_preset(&values[..16]);
                stage.fold_parameters = FoldParameters::from_preset(&values[16..24], &values[24..]);
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
                println!("1/2/3: fractal | WASD: fly | E/Q: up/down | right mouse: look | wheel: speed | R: reset | C: bulb section | Up/Down: section | Page Up/Down: bulb power");
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
            if modifiers.control_key() && !modifiers.alt_key() && !modifiers.super_key()
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
            if response.consumed && matches!(event, WindowEvent::KeyboardInput { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. }) {
                if let WindowEvent::KeyboardInput { ref event, .. } = event { if event.state == ElementState::Released { if let PhysicalKey::Code(key) = event.physical_key { self.keys.remove(&key); } } }
                if matches!(event, WindowEvent::MouseInput { state: ElementState::Released, button: MouseButton::Right, .. }) { self.looking = false; }
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = &mut self.renderer { renderer.resize(size.width, size.height); }
                if let Some(window) = &self.window { window.request_redraw(); }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(key) = event.physical_key {
                    if event.state == ElementState::Pressed {
                        self.keys.insert(key);
                        if key == KeyCode::KeyR && !event.repeat { self.camera = Camera::default(); }
                        if key == KeyCode::KeyG && !event.repeat { self.diagnostic = !self.diagnostic; println!("Flat diagnostic: {}", if self.diagnostic { "ON" } else { "OFF" }); }
                        if key == KeyCode::KeyX { self.exposure = (self.exposure + 0.25).min(5.0); }
                        if key == KeyCode::KeyZ { self.exposure = (self.exposure - 0.25).max(-5.0); }
                        if key == KeyCode::KeyN { self.normal_smoothing = (self.normal_smoothing * 1.25).min(*SMOOTHNESS.end()); }
                        if key == KeyCode::KeyB { self.normal_smoothing = (self.normal_smoothing / 1.25).max(*SMOOTHNESS.start()); }
                        if key == KeyCode::KeyO && !event.repeat { self.ao = !self.ao; println!("Ambient occlusion: {}", if self.ao { "ON" } else { "OFF" }); }
                        if key == KeyCode::KeyI { self.ao_strength = (self.ao_strength + 0.1).min(2.0); self.ao = true; }
                        if key == KeyCode::KeyK { self.ao_strength = (self.ao_strength - 0.1).max(0.0); self.ao = true; }
                        if key == KeyCode::KeyU { self.ao_radius = (self.ao_radius * 1.25).min(2.0); self.ao = true; }
                        if key == KeyCode::KeyJ { self.ao_radius = (self.ao_radius / 1.25).max(0.0001); self.ao = true; }
                        if matches!(key, KeyCode::KeyI | KeyCode::KeyK | KeyCode::KeyU | KeyCode::KeyJ) {
                            println!("AO strength: {:.1}, radius: {:.4}", self.ao_strength, self.ao_radius);
                        }
                        if key == KeyCode::KeyM && !event.repeat { self.material = (self.material + 1) % 10; println!("Material: {}", ["Original", "Pearl", "Copper", "Nebula", "Custom", "Gold", "Steel", "Clay", "Jade", "Aurora"][self.material as usize]); }
                        if matches!(key, KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4 | KeyCode::Digit5 | KeyCode::Digit6) && !event.repeat {
                            self.fractal = match key {
                                KeyCode::Digit1 => Fractal::Mandelbulb,
                                KeyCode::Digit2 => Fractal::Mandelbox,
                                KeyCode::Digit3 => Fractal::Menger,
                                KeyCode::Digit4 => Fractal::Kifs,
                                KeyCode::Digit5 => Fractal::AmazingSurface,
                                _ => Fractal::PseudoKleinian,
                            };
                            self.slice = None;
                            self.camera = Camera::default();
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
                if let Some(job) = &self.render_job {
                    let mut finished = None;
                    while let Ok(update) = job.receiver.try_recv() {
                        match update { export::Update::Progress(p) => { self.output.progress = p; }, export::Update::Finished(result) => finished = Some(result) }
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
                        let request = export::Request { width: self.output.width.clamp(64,8192), height: self.output.height.clamp(64,8192), samples: self.output.samples, uniforms };
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
                let mut scene = vec![self.camera.position[0], self.camera.position[1], self.camera.position[2], self.camera.yaw, self.camera.pitch, self.camera.roll, self.slice.is_some() as u8 as f32, self.slice.unwrap_or(0.0), self.power, self.fractal.value(), self.diagnostic as u8 as f32, self.material as f32, self.ao as u8 as f32, self.ao_strength, self.ao_radius, self.exposure, self.normal_smoothing, self.quality, self.antialias as u8 as f32, self.distant_filter, self.refine_idle as u8 as f32, self.lighting.color[0], self.lighting.color[1], self.lighting.color[2], self.lighting.roughness, self.lighting.metallic, self.lighting.light_color[0], self.lighting.light_color[1], self.lighting.light_color[2], self.lighting.azimuth, self.lighting.elevation, self.lighting.intensity, self.lighting.ambient, size.width as f32, size.height as f32];
                scene.extend(self.effects.preset());
                scene.push(self.matcap_rotation);
                scene.push(self.camera.look_roll);
                scene.extend(self.mandelbox_parameters.values());
                scene.extend(self.mandelbulb_parameters.values());
                scene.extend(self.menger_parameters.values());
                scene.extend(self.background_color);
                scene.extend(self.mutation.values());
                scene.extend(self.fold_parameters.values());
                scene.extend(self.mutation.fold_parameters.values());
                for stage in &self.extra_mutations {
                    scene.extend(stage.stacked_preset_values());
                }
                scene.push(self.base_weight);
                scene.extend(projection);
                let target = self.still_sample_target();
                scene.push(target as f32);
                let selected_mutation = self.mutation.selected(self.fractal);
                let extra_selected = self.extra_mutations.iter().any(|stage| stage.selected(self.fractal).is_some());
                let mutation_selected = selected_mutation.is_some() || extra_selected;
                let distance_blend = extra_selected || (self.mutation.weight > 0.0 && selected_mutation.is_some());
                let editing = mutation_selected && dragging && ctx.is_using_pointer();
                let navigating = moving || self.looking;
                let interacting = navigating || editing;
                if !interacting || frame_interval > 0.25 {
                    self.motion_samples = 1;
                    self.motion_recovery = 0;
                } else if frame_interval > 0.045 {
                    self.motion_samples = (self.motion_samples / 2).max(1);
                    self.motion_recovery = 0;
                } else if frame_interval < 0.027 {
                    self.motion_recovery += 1;
                    if self.motion_recovery >= 8 {
                        self.motion_samples = (self.motion_samples * 2).min(4);
                        self.motion_recovery = 0;
                    }
                } else {
                    self.motion_recovery = 0;
                }
                if scene != self.last_scene || interacting {
                    self.last_scene = scene;
                    self.idle_ready = false;
                    self.progress_samples = 0;
                    self.progress_tile = 0;
                    self.idle_deadline = None;
                }
                if !interacting && !self.idle_ready && self.idle_deadline.is_none() {
                    self.idle_deadline = Some(Instant::now() + Duration::from_millis(600));
                }
                let progressing = self.idle_ready && !interacting;
                let capture_slot = self.capture_slot.take();
                let composite_progress = progressing && capture_slot.is_none();
                let advance = composite_progress && self.progress_samples < target;
                let progress = composite_progress.then_some(self.progress_samples);
                let mut data = self.camera_uniforms([size.width.max(1) as f32, size.height.max(1) as f32], progress, projection, false);
                data[10] = frame.height() * output.pixels_per_point;
                let fixed_aa = self.antialias && !self.refine_idle && !self.diagnostic;
                let preview_divisor = if capture_slot.is_some() {
                    4
                } else if progressing {
                    1
                } else if mutation_selected || (self.refine_idle && !navigating) {
                    if navigating && !distance_blend { 2 } else { 4 }
                } else if fixed_aa {
                    2
                } else {
                    1
                };
                if preview_divisor > 1 && data[19] > 0.0 {
                    data[19] = match self.motion_samples { 4 => 1.0, 2 => 0.5, _ => 0.0 };
                }
                // Keep the preview visible while the full-resolution AA pass completes in tiles.
                let refine_tile = (advance && (self.refine_idle || fixed_aa)).then_some(self.progress_tile);
                let capture_result = self.renderer.as_mut().and_then(|renderer| renderer.render(data, progress, advance, refine_tile, preview_divisor, output, &ctx, capture_slot.is_some()));
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
                if advance && self.renderer.as_ref().is_some_and(|renderer| renderer.frame_submitted) {
                    let tile_count = size.width.max(1).div_ceil(REFINE_TILE_WIDTH) * size.height.max(1).div_ceil(REFINE_TILE_HEIGHT);
                    if refine_tile.is_some() && self.progress_tile + 1 < tile_count {
                        self.progress_tile += 1;
                    } else {
                        self.progress_tile = 0;
                        self.progress_samples += 1;
                    }
                }
                if moving || advance { window.request_redraw(); }
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
    fn png_uses_the_viewports_stationary_detail_settings() {
        let mut app = super::App::default();
        app.antialias = true;
        for (quality, stationary_quality) in [(0.7, 0.7 * 1.25), (4.0, 4.0), (8.0, 8.0)] {
            app.quality = quality;
            for refine in [false, true] {
                app.refine_idle = refine;
                let viewport = app.camera_uniforms([1380.0, 900.0], Some(0), [0.0; 4], false);
                let moving = app.camera_uniforms([1380.0, 900.0], None, [0.0; 4], false);
                let export = app.camera_uniforms([3840.0, 2160.0], None, [0.0; 4], true);
                assert_eq!([export[18], export[23]], [stationary_quality, 1.0]);
                assert_eq!([viewport[18], viewport[23]], [stationary_quality, 1.0]);
                assert_eq!([moving[18], moving[23]], [quality, 0.0]);
                assert_eq!(viewport[19], super::flag(!refine));
                assert_eq!(app.still_sample_target(), if refine { app.refine_target } else { 1 });
            }
        }
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
        for tab in super::ControlsTab::ALL {
            app.controls_tab = tab;
            ctx.data_mut(|data| {
                data.insert_persisted(panel_id, egui::containers::panel::PanelState {
                    rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, screen.height())),
                });
            });
            let _ = ctx.run(input(), |ctx| {
                app.ui(ctx);
                let width = ctx.available_rect().left();
                assert!((super::CONTROLS_MIN_WIDTH..=super::CONTROLS_MIN_WIDTH + 2.0).contains(&width), "{} tab width: {width}", tab.label());
                assert!(ctx.available_rect().width() >= 150.0);
            });
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
        assert!(app.fractal_title().starts_with("MengerVibes - Menger"));
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
    fn presets_round_trip_and_recover_duplicated_sample_count() {
        let path = std::env::temp_dir().join(format!("mengervibes-preset-test-{}.txt", std::process::id()));
        let path = path.to_str().unwrap();
        let mut original = super::App::default();
        original.camera.position = [1.0, 2.0, 3.0];
        original.camera.roll = 0.7;
        original.camera.look_roll = -0.3;
        original.camera.pitch = 1.56;
        original.matcap_rotation = 70.5;
        original.exposure = 2.2;
        original.distant_filter = 3.3;
        original.refine_target = 64;
        original.lighting.color = [0.1, 0.2, 0.3];
        original.effects.focal = 50.0;
        original.effects.focus = 0.15;
        original.effects.aperture = 0.002;
        original.effects.dof_enabled = true;
        original.contrast = 2.5;
        original.quality = 8.0;
        original.normal_smoothing = 0.000020;
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
        assert_eq!(loaded.matcap_rotation, original.matcap_rotation);
        assert_eq!(loaded.exposure, original.exposure);
        assert_eq!(loaded.distant_filter, original.distant_filter);
        assert_eq!(loaded.lighting.color, original.lighting.color);
        assert_eq!(loaded.refine_target, 64);
        assert_eq!(loaded.effects.preset(), original.effects.preset());
        assert_eq!(loaded.contrast, 2.5);
        assert_eq!(loaded.quality, 8.0);
        assert_eq!(loaded.normal_smoothing, 0.000020);
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
        assert_eq!(loaded.fold_parameters, original.fold_parameters);
        assert_eq!(loaded.mutation, original.mutation);
        restored.fractal = super::Fractal::PseudoKleinian;
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
        assert_eq!(loaded.mutation, original.mutation);
        assert_eq!(loaded.fold_parameters, super::FoldParameters::default());

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
        assert_eq!(loaded.matcap_rotation, 0.0);
        assert_eq!(loaded.camera.look_roll, 0.0);
        assert_eq!(loaded.effects.preset(), original.effects.preset());
        assert_eq!(loaded.contrast, 1.6);
        assert_eq!(loaded.mandelbox_parameters, super::MandelboxParameters::default());
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
        assert_eq!(recovered.refine_target, 64);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fractal_shader_parses_and_validates() {
        let source = super::FRACTAL_SHADER;
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("Shader validation failed");
    }

    #[test]
    fn progressive_pipelines_validate_on_gpu() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) else {
            eprintln!("No GPU adapter available; skipping GPU validation");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).expect("GPU device");
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
        let (focus_pipeline, focus_bind) = super::Renderer::create_focus_pipeline(&device, &shader, &camera);
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
        let camera_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera.as_entire_binding() }],
        });
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
    }
}
