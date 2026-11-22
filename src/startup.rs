use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, mpsc},
    time::{Duration, Instant},
};

use winit::{dpi::PhysicalSize, event_loop::EventLoopProxy, window::Window};

use crate::{APP_NAME, Renderer};

type SurfaceResult = Result<(wgpu::Instance, wgpu::Surface<'static>), String>;

pub(crate) enum Event {
    Stage(&'static str),
    CreateSurface { instance: wgpu::Instance, reply: mpsc::Sender<SurfaceResult> },
    Finished(Result<Box<Renderer>, String>),
    UpdateChecked(Result<crate::about::BuildStatus, crate::about::CheckError>),
}

pub(crate) fn create_surface(window: Arc<Window>, instance: wgpu::Instance, reply: mpsc::Sender<SurfaceResult>) {
    // Winit exposes the native window handle on its owning thread.
    let surface = instance.create_surface(window).map_err(|error| format!("GPU surface: {error}"));
    let _ = reply.send(surface.map(|surface| (instance, surface)));
}

pub(crate) struct Startup {
    pub(crate) started: Option<Instant>,
    stage: &'static str,
    error: Option<String>,
    pub(crate) next_repaint: Option<Instant>,
}

impl Default for Startup {
    fn default() -> Self {
        Self { started: None, stage: "Starting graphics…", error: None, next_repaint: None }
    }
}

impl Startup {
    pub(crate) fn start(&mut self, size: PhysicalSize<u32>, proxy: EventLoopProxy<Event>) {
        self.started = Some(Instant::now());
        self.next_repaint = Some(Instant::now() + Duration::from_secs(1));

        let worker = std::thread::Builder::new().name("graphics-startup".into()).spawn(move || {
            let stage = |name| {
                println!("Startup: {name}");
                let _ = proxy.send_event(Event::Stage(name));
            };
            let result = catch_unwind(AssertUnwindSafe(|| {
                stage("Connecting to the graphics driver");
                let instance = wgpu::Instance::default();
                let (reply, response) = mpsc::channel();
                proxy.send_event(Event::CreateSurface { instance, reply })
                    .map_err(|_| "Window closed during startup")?;
                let (instance, surface) = response.recv().map_err(|_| "Window closed during startup")??;
                pollster::block_on(Renderer::new(instance, surface, size, stage))
            })).unwrap_or_else(|panic| {
                let message = panic.downcast_ref::<String>().map(String::as_str)
                    .or_else(|| panic.downcast_ref::<&str>().copied())
                    .unwrap_or("Unknown graphics error");
                Err(message.into())
            });
            let _ = proxy.send_event(Event::Finished(result.map(Box::new)));
        });
        if let Err(error) = worker { self.fail(format!("Could not start the graphics worker: {error}")); }
    }

    pub(crate) fn set_stage(&mut self, stage: &'static str) {
        self.stage = stage;
    }

    pub(crate) fn fail(&mut self, error: String) {
        eprintln!("Graphics startup failed: {error}");
        self.error = Some(error);
        self.next_repaint = None;
    }

    pub(crate) fn paint(&self, window: &Window) {
        let elapsed = self.started.map_or(0, |start| start.elapsed().as_secs());
        let (status, detail) = if let Some(error) = &self.error {
            ("Graphics startup failed".into(), error.clone())
        } else {
            let detail = if elapsed >= 15 {
                "The graphics driver is taking longer than usual. You can close this window."
            } else {
                "Preparing the fractal viewer"
            };
            (format!("{}  ·  {elapsed}s", self.stage), detail.into())
        };
        window.set_title(&format!("{APP_NAME} - {status}"));
        #[cfg(windows)]
        paint_windows(window, &status, &detail);
        #[cfg(not(windows))]
        let _ = detail;
    }
}

#[cfg(windows)]
fn paint_windows(window: &Window, status: &str, detail: &str) {
    use windows_sys::Win32::{
        Foundation::RECT,
        Graphics::Gdi::{
            CreateSolidBrush, DEFAULT_GUI_FONT, DT_CENTER, DT_NOPREFIX, DT_SINGLELINE, DT_VCENTER,
            DT_WORDBREAK, DeleteObject, DrawTextW, FillRect, GetDC, GetStockObject, ReleaseDC,
            SelectObject, SetBkMode, SetTextColor, TRANSPARENT,
        },
        UI::WindowsAndMessaging::GetClientRect,
    };
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let Ok(handle) = window.window_handle() else { return; };
    let RawWindowHandle::Win32(handle) = handle.as_raw() else { return; };
    let hwnd = handle.hwnd.get() as _;
    // The borrowed HWND belongs to this live window. Painting and releasing its DC
    // both happen on the window's event-loop thread.
    unsafe {
        let mut rect = RECT::default();
        if GetClientRect(hwnd, &mut rect) == 0 { return; }
        let dc = GetDC(hwnd);
        if dc.is_null() { return; }
        let background = CreateSolidBrush(0x0021_1813);
        if !background.is_null() {
            FillRect(dc, &rect, background);
            DeleteObject(background);
        }
        let font = SelectObject(dc, GetStockObject(DEFAULT_GUI_FONT));
        SetBkMode(dc, TRANSPARENT as i32);

        let center = rect.bottom / 2;
        let margin = (rect.right / 16).min(80);
        let scale = window.scale_factor() as f32;
        let line = (26.0 * scale) as i32;
        let draw = |text: &str, top, bottom, color, format| {
            let wide: Vec<u16> = text.encode_utf16().collect();
            let mut bounds = RECT { left: margin, top, right: rect.right - margin, bottom };
            SetTextColor(dc, color);
            DrawTextW(dc, wide.as_ptr(), wide.len() as i32, &mut bounds, format | DT_CENTER | DT_NOPREFIX);
        };
        draw(APP_NAME, center - line * 2, center - line, 0x00ef_d459, DT_SINGLELINE | DT_VCENTER);
        draw(status, center - line, center, 0x00f2_ece4, DT_SINGLELINE | DT_VCENTER);
        draw(detail, center + line / 2, rect.bottom, 0x00bf_b1a2, DT_WORDBREAK);

        if !font.is_null() { SelectObject(dc, font); }
        ReleaseDC(hwnd, dc);
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use winit::{
        application::ApplicationHandler,
        event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
        platform::windows::EventLoopBuilderExtWindows,
    };

    fn compare_navigation(renderer: &mut crate::Renderer, ctx: &egui::Context, extent: [u32; 2]) -> Result<(), String> {
        use crate::{App, Camera, Fractal, MutationStage, adaptive_viewport::{AdaptiveViewport, PreviewWork}};

        let mut app = App::default();
        app.lighting.shadows = false;
        app.lighting.bounces = 1;
        let mut capture = |app: &App, divisor: u32, adaptive: bool, reference: bool, frame: u32| -> Result<_, String> {
            let progress = reference.then_some(0);
            let mut data = app.camera_uniforms(extent.map(|side| side as f32), progress, [0.0; 4], false);
            let size = extent.map(|side| side.div_ceil(divisor));
            if !reference {
                data[crate::MOVING_SAMPLES_UNIFORM] = 1.0;
                data[crate::MOVING_BOUNCES_UNIFORM] = 1.0;
                data[crate::NAVIGATION_SAMPLE_UNIFORM] = frame as f32;
            }
            let work = (adaptive && !reference).then_some(PreviewWork { size, divisor, samples: 1 });
            if work.is_some() { data[18] = (app.quality * 1.25).min(4.0).max(app.quality); }
            let gui = ctx.run(egui::RawInput::default(), |_| {});
            let start = Instant::now();
            let image = renderer.render(data, progress, reference, None, size, gui, ctx, true, adaptive, work)
                .ok_or("Navigation capture unavailable")??;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            let stats = renderer.poll_viewport_work().flatten();
            let small = image::imageops::resize(&image, 10, 6, image::imageops::FilterType::Triangle);
            Ok((small, ms, stats))
        };
        let coarse_error = |a: &image::RgbaImage, b: &image::RgbaImage| {
            a.pixels().zip(b.pixels()).flat_map(|(a, b)| (0..3).map(move |i| f64::from(a[i].abs_diff(b[i])))).sum::<f64>()
                / (f64::from(a.width() * a.height()) * 3.0)
        };
        for (name, formula, mutations, interior) in [
            ("Menger exterior", Fractal::Menger, &[][..], false),
            ("Mandelbox exterior", Fractal::Mandelbox, &[][..], false),
            ("Mandelbulb exterior", Fractal::Mandelbulb, &[][..], false),
            ("Bulb + three mutations", Fractal::Mandelbulb, &[Fractal::Mandelbox, Fractal::AmazingSurface, Fractal::Menger][..], false),
            ("Interior + paths", Fractal::Mandelbulb, &[Fractal::Mandelbox, Fractal::Menger][..], true),
        ] {
            app.fractal = formula;
            app.camera = Camera::framed_view(formula);
            app.lighting.path_traced = interior;
            if interior { app.camera.position = [0.0; 3]; }
            app.extra_mutations = mutations.iter().map(|&formula| MutationStage { formula: Some(formula), ..Default::default() }).collect();
            let legacy_divisor = if mutations.is_empty() { 1 } else { 4 };
            let mut scheduler = AdaptiveViewport::default();
            let mut elapsed = [Vec::new(), Vec::new()];
            let mut errors = [0.0; 2];
            let mut acceptable = [0; 2];
            let mut levels = Vec::new();
            let mut previous = None;
            let mut pose_changes = 0;
            let yaw = app.camera.yaw;
            // Warm up both paths, then alternate large turns within one session.
            capture(&app, legacy_divisor, false, false, 0)?;
            capture(&app, 4, true, false, 0)?;
            for (frame, offset) in [0.0, 0.43, -0.35, 0.64, -0.19].into_iter().enumerate() {
                app.camera.yaw = yaw + offset;
                let (reference, _, _) = capture(&app, 1, false, true, 0)?;
                let (legacy, legacy_ms, _) = capture(&app, legacy_divisor, false, false, frame as u32)?;
                let divisor = scheduler.divisor();
                let (adaptive, adaptive_ms, stats) = capture(&app, divisor, true, false, frame as u32)?;
                levels.push(divisor);
                scheduler.observe(stats.ok_or("Adaptive navigation did not report completion")?);
                if previous.as_ref().is_some_and(|prior| prior != &adaptive) { pose_changes += 1; }
                previous = Some(adaptive.clone());
                elapsed[0].push(legacy_ms);
                elapsed[1].push(adaptive_ms);
                for (i, image) in [legacy, adaptive].iter().enumerate() {
                    let error = coarse_error(image, &reference);
                    errors[i] += error / 5.0;
                    acceptable[i] += usize::from(error <= 12.0);
                }
            }
            for times in &mut elapsed { times.sort_by(f64::total_cmp); }
            if pose_changes < 2 { return Err(format!("{name}: the adaptive preview did not track the scripted turns")); }
            println!("{name} turns: median capture {:.1}/{:.1} ms; coarse RGB error {:.2}/{:.2}; below threshold {}/5 and {}/5 (legacy/adaptive); divisors {levels:?}",
                elapsed[0][2], elapsed[1][2], errors[0], errors[1], acceptable[0], acceptable[1]);
        }
        Ok(())
    }

    #[test]
    #[ignore = "Requires a Windows desktop and a graphics adapter"]
    fn starts_renderer_on_worker_with_windows_event_loop_running() {
        struct Probe {
            startup: Startup,
            proxy: EventLoopProxy<Event>,
            window: Option<Arc<Window>>,
            deadline: Instant,
            result: Option<Result<(), String>>,
        }
        impl ApplicationHandler<Event> for Probe {
            fn resumed(&mut self, event_loop: &ActiveEventLoop) {
                let window = event_loop.create_window(Window::default_attributes()
                    .with_visible(false).with_inner_size(winit::dpi::PhysicalSize::new(300, 180))).unwrap();
                self.startup.start(window.inner_size(), self.proxy.clone());
                self.window = Some(Arc::new(window));
            }

            fn user_event(&mut self, event_loop: &ActiveEventLoop, event: Event) {
                match event {
                    Event::Stage(stage) => self.startup.set_stage(stage),
                    Event::UpdateChecked(_) => unreachable!("The startup test does not check for updates"),
                    Event::CreateSurface { instance, reply } => create_surface(self.window.as_ref().unwrap().clone(), instance, reply),
                    Event::Finished(result) => {
                        self.result = Some(result.and_then(|mut renderer| {
                            let mut app = crate::App::default();
                            app.fractal = crate::Fractal::Mandelbox;
                            app.camera.position = [0.93724036, -0.018842345, 1.1496642];
                            app.camera.yaw = -0.8167231;
                            app.camera.pitch = 0.024;
                            app.mandelbox_parameters.scale = -1.5567619;
                            app.mandelbox_parameters.fold_limit = 0.9502837;
                            app.mandelbox_parameters.inner_radius = 0.23;
                            app.mandelbox_parameters.outer_radius = 0.95;
                            app.mutation.formula = Some(crate::Fractal::Menger);
                            app.mutation.weight = 0.72;
                            app.mutation.menger_parameters.depth = 10;
                            app.material = 3;
                            app.exposure = 1.8;
                            app.quality = 1.1;
                            app.normal_smoothing = 0.005;
                            let extent = [renderer.config.width, renderer.config.height];
                            let size = extent.map(|side| side as f32);
                            let ctx = egui::Context::default();
                            renderer.device.push_error_scope(wgpu::ErrorFilter::Validation);
                            let tiles = renderer.config.width.div_ceil(crate::REFINE_TILE_WIDTH)
                                * renderer.config.height.div_ceil(crate::REFINE_TILE_HEIGHT);
                            let mut capture = |progress: Option<u32>, tile, divisor, adaptive| -> Result<image::RgbaImage, String> {
                                let _ = renderer.poll_viewport_work();
                                let mut data = app.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                let render_size = extent.map(|side| side.div_ceil(divisor));
                                let preview = (adaptive && progress.is_none()).then_some(crate::adaptive_viewport::PreviewWork {
                                    size: render_size, divisor, samples: 1,
                                });
                                if preview.is_some() { data[18] = (app.quality * 1.25).min(4.0).max(app.quality); }
                                renderer.render(data, progress, progress.is_some(), tile, render_size, gui, &ctx, true, adaptive, preview)
                                    .ok_or_else(|| "Viewport capture unavailable".to_string())?
                            };
                            let legacy_preview = capture(None, None, 4, false)?;
                            let preview = capture(None, None, 4, true)?;
                            let mut refined = capture(Some(0), Some(0..1), 1, true)?;
                            if tiles > 1 && refined != preview {
                                return Err("The preview changed before the first sample completed".into());
                            }
                            if tiles > 1 { refined = capture(Some(0), Some(1..tiles), 1, true)?; }
                            if refined != capture(Some(0), None, 1, false)? {
                                return Err("Tiled refinement differs from a full pass".into());
                            }
                            let reduced = image::imageops::resize(&preview, 10, 6, image::imageops::FilterType::Triangle);
                            let old = image::imageops::resize(&legacy_preview, 10, 6, image::imageops::FilterType::Triangle);
                            let still = image::imageops::resize(&refined, 10, 6, image::imageops::FilterType::Triangle);
                            let difference = |image: &image::RgbaImage| image.as_raw().chunks_exact(4).zip(still.as_raw().chunks_exact(4))
                                .flat_map(|(a, b)| (0..3).map(move |channel| f32::from(a[channel].abs_diff(b[channel]))))
                                .sum::<f32>() / (image.width() * image.height() * 3) as f32;
                            let mean_difference = difference(&reduced);
                            println!("Mandelbox + Menger at 300×180, coarse RGB error (0–255): legacy {:.2}, adaptive {mean_difference:.2}", difference(&old));
                            if mean_difference > 26.0 {
                                return Err(format!("Menger blend lost its overall structure during navigation ({mean_difference:.1})"));
                            }
                            let _ = renderer.poll_viewport_work();
                            crate::intelligent_sampling::compare_viewport_sampling(&mut renderer, &ctx, &mut app, extent)?;
                            compare_navigation(&mut renderer, &ctx, extent)?;
                            let settings = crate::noise_sampling::Settings { enabled: true, threshold: 0.01, min_samples: 2 };
                            renderer.noise_plan = settings.plan(8);
                            let noise_pipelines = renderer.noise_pipelines.get_or_init(|| crate::noise_sampling::Pipelines::new(&renderer.device, false));
                            noise_pipelines.sample_pipeline(&renderer.device, &renderer.camera_layout, false);
                            let mut empty = crate::App::default();
                            empty.camera.position = [100.0; 3];
                            empty.camera.yaw = 0.0;
                            empty.camera.pitch = 0.0;
                            for index in 0..2 {
                                let data = empty.camera_uniforms(size, Some(index), [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, Some(index), true, None, extent, gui, &ctx, true, false, None)
                                    .ok_or("Adaptive viewport capture unavailable")??;
                            }
                            renderer.poll_noise_sampling();
                            if renderer.noise_remaining != Some(0)
                                || renderer.noise_tiles.as_ref().is_none_or(|tiles| tiles.len() != 0)
                            {
                                return Err("The uniform background did not converge after two samples".into());
                            }
                            renderer.noise_remaining = None;
                            renderer.noise_tiles = None;
                            empty.camera.position = [0.0, 0.0, 10.0];
                            empty.effects.dof_enabled = true;
                            empty.effects.aperture = 0.04;
                            empty.effects.focus = 2.0;
                            for index in 0..2 {
                                let data = empty.camera_uniforms(size, Some(index), [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, Some(index), true, None, extent, gui, &ctx, true, false, None)
                                    .ok_or("Adaptive viewport capture unavailable")??;
                            }
                            renderer.poll_noise_sampling();
                            let [columns, rows] = crate::noise_sampling::viewport_tile_grid(extent);
                            let total = columns * rows;
                            let tiles = renderer.noise_tiles.take().ok_or("No active viewport tiles reported")?;
                            if !(1..total).contains(&tiles.len()) {
                                return Err(format!("The viewport did not isolate noisy tiles: {} of {total} remain", tiles.len()));
                            }
                            let render_pass = |renderer: &mut crate::Renderer| -> Result<_, String> {
                                let tile_count = renderer.noise_tiles.as_ref().map_or(total, crate::noise_sampling::ActiveTiles::len);
                                let start = Instant::now();
                                let mut submissions = 0;
                                let mut cursor = 0;
                                while cursor < tile_count {
                                    let end = renderer.noise_tiles.as_ref().map_or(cursor + 1, |active| active.batch_end(cursor, 1));
                                    let data = empty.camera_uniforms(size, Some(2), [0.0; 4], false);
                                    let gui = ctx.run(egui::RawInput::default(), |_| {});
                                    let capture = end == tile_count;
                                    let result = renderer.render(data, Some(2), true, Some(cursor..end), extent, gui, &ctx, capture, false, None);
                                    if capture { result.ok_or("Adaptive viewport frame unavailable")??; }
                                    if !renderer.frame_submitted { return Err("Adaptive pass was not submitted".into()); }
                                    submissions += 1;
                                    cursor = end;
                                }
                                Ok((submissions, start.elapsed().as_secs_f64() * 1000.0))
                            };
                            let active_count = tiles.len();
                            let mut active = Some(tiles);
                            let mut old_times = Vec::new();
                            let mut new_times = Vec::new();
                            let mut old_submissions = 0;
                            let mut new_submissions = 0;
                            for _ in 0..4 {
                                renderer.noise_tiles = None;
                                let (old_count, old_ms) = render_pass(&mut renderer)?;
                                renderer.noise_tiles = active.take();
                                let (new_count, new_ms) = render_pass(&mut renderer)?;
                                active = renderer.noise_tiles.take();
                                old_submissions = old_count;
                                new_submissions = new_count;
                                old_times.push(old_ms);
                                new_times.push(new_ms);
                            }
                            old_times.sort_by(f64::total_cmp);
                            new_times.sort_by(f64::total_cmp);
                            if new_submissions >= old_submissions { return Err("Sparse viewport work did not reduce submissions".into()); }
                            println!("Viewport: {active_count}/{total} active tiles, {old_submissions} to {new_submissions} frame submissions; {:.1} to {:.1} ms per pass",
                                (old_times[1] + old_times[2]) * 0.5, (new_times[1] + new_times[2]) * 0.5);
                            renderer.noise_plan = None;
                            renderer.noise_remaining = None;
                            renderer.noise_tiles = None;
                            app.effects.aperture = 0.04;
                            app.effects.focus = 0.15;
                            let mut capture_dof = |app: &crate::App, sample, advance| -> Result<image::RgbaImage, String> {
                                let progress = Some(sample);
                                let data = app.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, progress, advance, None, extent, gui, &ctx, true, false, None)
                                    .ok_or_else(|| "Depth of field preview unavailable".to_string())?
                            };
                            app.effects.dof_enabled = false;
                            let pinhole = capture_dof(&app, 0, true)?;
                            app.effects.dof_enabled = true;
                            let first_lens_sample = capture_dof(&app, 0, true)?;
                            if pinhole == first_lens_sample {
                                return Err("The first viewport sample did not use the camera aperture".into());
                            }
                            let blurred = capture_dof(&app, 1, true)?;
                            if first_lens_sample == blurred || capture_dof(&app, 2, false)? != blurred {
                                return Err("The full-resolution DOF preview did not accumulate or retain its samples".into());
                            }
                            let mut ao_scene = crate::App::default();
                            ao_scene.camera.position = [0.0, 0.0, 3.2];
                            ao_scene.camera.yaw = 0.0;
                            ao_scene.camera.pitch = 0.0;
                            ao_scene.effects.focal = 21.6;
                            ao_scene.ao_strength = 1.0;
                            ao_scene.ao_radius = 0.75;
                            let mut capture_ao = |enabled| -> Result<image::RgbaImage, String> {
                                ao_scene.ao = enabled;
                                let reduced_extent = extent.map(|side| side.div_ceil(2));
                                let data = ao_scene.camera_uniforms(size, Some(0), [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, Some(0), true, None, reduced_extent, gui, &ctx, true, false, None)
                                    .ok_or_else(|| "Ray-traced AO preview unavailable".to_string())?
                            };
                            if capture_ao(false)? == capture_ao(true)? {
                                return Err("Ray-traced AO did not affect the reduced-resolution viewport".into());
                            }
                            ao_scene.lighting.path_traced = true;
                            ao_scene.select_material(crate::DIFFUSE_GRAY_MATERIAL);
                            ao_scene.lighting.enabled = false;
                            let mut capture_path = |environment, sample| -> Result<image::RgbaImage, String> {
                                ao_scene.lighting.environment = environment;
                                let progress = Some(sample);
                                let data = ao_scene.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, progress, true, None, extent.map(|side| side.div_ceil(2)), gui, &ctx, true, false, None)
                                    .ok_or_else(|| "Path-traced preview unavailable".to_string())?
                            };
                            let uniform = capture_path(0, 0)?;
                            let copper = capture_path(3, 0)?;
                            if uniform == copper || copper == capture_path(3, 1)? {
                                return Err("Path-traced viewport did not respond to the environment or accumulate a new sample".into());
                            }
                            let mut capture_moving = |samples, bounces| -> Result<image::RgbaImage, String> {
                                let mut data = ao_scene.camera_uniforms(size, None, [0.0; 4], false);
                                data[crate::MOVING_SAMPLES_UNIFORM] = samples;
                                data[crate::MOVING_BOUNCES_UNIFORM] = bounces;
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, None, false, None, extent.map(|side| side.div_ceil(4)), gui, &ctx, true, false, None)
                                    .ok_or_else(|| "Moving viewport preview unavailable".to_string())?
                            };
                            let four_samples = capture_moving(4.0, 1.0)?;
                            if capture_moving(1.0, 1.0)? == four_samples {
                                return Err("Moving viewport sample count did not change the preview".into());
                            }
                            if capture_moving(4.0, 4.0)? == four_samples {
                                return Err("Moving viewport bounce count did not change the preview".into());
                            }
                            let mut pixel_previews = Vec::new();
                            for mode in crate::pixel_sampling::PixelSampling::ALL {
                                ao_scene.pixel_sampling = mode;
                                let data = ao_scene.camera_uniforms(size, None, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                let image = renderer.render(data, None, false, None,
                                    extent.map(|side| side.div_ceil(4)), gui, &ctx, true, false, None)
                                    .ok_or_else(|| "Pixel sampling preview unavailable".to_string())??;
                                if pixel_previews.contains(&image) {
                                    return Err(format!("{} did not affect the viewport", mode.label()));
                                }
                                pixel_previews.push(image);
                            }
                            ao_scene.pixel_sampling = crate::pixel_sampling::PixelSampling::R2;
                            let denoise_extent = extent.map(|side| side.div_ceil(2));
                            let capture_denoise = |renderer: &mut crate::Renderer, sample, advance| -> Result<image::RgbaImage, String> {
                                let progress = Some(sample);
                                let data = ao_scene.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, progress, advance, None, denoise_extent, gui, &ctx, true, false, None)
                                    .ok_or_else(|| "Denoised viewport unavailable".to_string())?
                            };
                            for backend in [crate::denoise::Backend::Cpu, crate::denoise::Backend::Gpu]
                                .into_iter().filter(|backend| backend.available())
                            {
                                crate::intelligent_sampling::wait_for_pipelines(&mut renderer, true)?;
                                renderer.intelligent_sampling.enabled = true;
                                renderer.viewport_denoise.configure(Some(backend), true);
                                let start_denoiser = |renderer: &mut crate::Renderer, samples| {
                                    let (done, finished) = std::sync::mpsc::channel();
                                    renderer.viewport_denoise.start(&renderer.device, &renderer.queue, &renderer.accumulation,
                                        denoise_extent, samples, 6, crate::viewport_denoise::DEFAULT_INTERVAL, move || { let _ = done.send(()); });
                                    finished
                                };
                                let present_denoised = |renderer: &mut crate::Renderer, finished: std::sync::mpsc::Receiver<()>, samples, traced| -> Result<image::RgbaImage, String> {
                                    finished.recv_timeout(Duration::from_secs(15)).map_err(|error| error.to_string())?;
                                    renderer.poll_viewport_denoise();
                                    if let Some(error) = &renderer.viewport_denoise.error { return Err(error.clone()); }
                                    let image = capture_denoise(renderer, traced, false)?;
                                    let timing = renderer.viewport_denoise.presented.as_ref()
                                        .filter(|timing| timing.samples == samples && timing.traced_samples == traced)
                                        .ok_or_else(|| format!("{} did not present {samples} spp with {traced} spp traced", backend.label()))?;
                                    println!("{} | {} spp filtered, {} traced at presentation | readback {:.1} ms | prepare {:.1} ms | OIDN {:.1} ms | to present {:.1} ms",
                                        backend.label(), timing.samples, timing.traced_samples, timing.readback_ms,
                                        timing.preparation_ms, timing.oidn_ms, timing.presentation_ms);
                                    Ok(image)
                                };
                                let noisy = capture_denoise(&mut renderer, 0, true)?;
                                let first = start_denoiser(&mut renderer, 1);
                                assert!(renderer.viewport_denoise.awaiting_first_preview());
                                let early = present_denoised(&mut renderer, first, 1, 1)?;
                                if noisy == early {
                                    return Err(format!("{} did not filter the first viewport sample", backend.label()));
                                }
                                for sample in 1..4 {
                                    capture_denoise(&mut renderer, sample, true)?;
                                }
                                let intermediate = start_denoiser(&mut renderer, 4);
                                assert!(!renderer.viewport_denoise.awaiting_first_preview());
                                capture_denoise(&mut renderer, 4, true)?;
                                if early != capture_denoise(&mut renderer, 5, true)? {
                                    return Err("First filtered viewport was lost during refinement".into());
                                }
                                let intermediate = present_denoised(&mut renderer, intermediate, 4, 6)?;
                                let finish = start_denoiser(&mut renderer, 6);
                                let final_image = present_denoised(&mut renderer, finish, 6, 6)?;
                                if early == intermediate || intermediate == final_image {
                                    return Err(format!("{} did not refine the filtered viewport", backend.label()));
                                }
                                renderer.viewport_denoise.configure(None, true);
                                let raw = capture_denoise(&mut renderer, 6, false)?;
                                if raw == final_image || renderer.viewport_denoise.display.is_some() {
                                    return Err("The final viewport was still filtered after disabling denoising".into());
                                }
                                renderer.intelligent_sampling.enabled = false;
                            }
                            match pollster::block_on(renderer.device.pop_error_scope()) {
                                Some(error) => Err(error.to_string()),
                                None => Ok(()),
                            }
                        }));
                        event_loop.exit();
                    }
                }
            }

            fn window_event(&mut self, _: &ActiveEventLoop, _: winit::window::WindowId, _: winit::event::WindowEvent) {}

            fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
                if self.result.is_some() {
                    event_loop.exit();
                    return;
                }
                if Instant::now() >= self.deadline {
                    self.result = Some(Err(format!("Graphics startup stalled at: {}", self.startup.stage)));
                    event_loop.exit();
                } else {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(self.deadline));
                }
            }
        }

        let event_loop = EventLoop::<Event>::with_user_event().with_any_thread(true).build().unwrap();
        let mut probe = Probe {
            startup: Startup::default(),
            proxy: event_loop.create_proxy(),
            window: None,
            deadline: Instant::now() + Duration::from_secs(30),
            result: None,
        };
        event_loop.run_app(&mut probe).unwrap();
        probe.result.unwrap().unwrap();
    }
}
