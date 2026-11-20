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
                            let mut capture = |progress, tile, divisor| -> Result<image::RgbaImage, String> {
                                let data = app.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, progress, progress.is_some(), tile, extent.map(|side| side.div_ceil(divisor)), gui, &ctx, true)
                                    .ok_or_else(|| "Viewport capture unavailable".to_string())?
                            };
                            let preview = capture(None, None, 4)?;
                            let mut refined = capture(Some(0), Some(0..1), 1)?;
                            if tiles > 1 && refined != preview {
                                return Err("The preview changed before the first sample completed".into());
                            }
                            if tiles > 1 { refined = capture(Some(0), Some(1..tiles), 1)?; }
                            if refined != capture(Some(0), None, 1)? {
                                return Err("Tiled refinement differs from a full pass".into());
                            }
                            let reduced = image::imageops::resize(&preview, 10, 6, image::imageops::FilterType::Triangle);
                            let still = image::imageops::resize(&refined, 10, 6, image::imageops::FilterType::Triangle);
                            let mean_difference = reduced.as_raw().iter().zip(still.as_raw())
                                .map(|(a, b)| f32::from(a.abs_diff(*b))).sum::<f32>() / reduced.as_raw().len() as f32;
                            if mean_difference > 26.0 {
                                return Err(format!("Menger blend lost its overall structure during navigation ({mean_difference:.1})"));
                            }
                            app.effects.dof_enabled = true;
                            app.effects.aperture = 0.04;
                            app.effects.focus = 0.15;
                            let mut capture_dof = |sample, advance| -> Result<image::RgbaImage, String> {
                                let progress = Some(sample);
                                let data = app.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, progress, advance, None, extent, gui, &ctx, true)
                                    .ok_or_else(|| "Depth of field preview unavailable".to_string())?
                            };
                            let pinhole = capture_dof(0, true)?;
                            let blurred = capture_dof(1, true)?;
                            if pinhole == blurred || capture_dof(2, false)? != blurred {
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
                                renderer.render(data, Some(0), true, None, reduced_extent, gui, &ctx, true)
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
                                renderer.render(data, progress, true, None, extent.map(|side| side.div_ceil(2)), gui, &ctx, true)
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
                                renderer.render(data, None, false, None, extent.map(|side| side.div_ceil(4)), gui, &ctx, true)
                                    .ok_or_else(|| "Moving viewport preview unavailable".to_string())?
                            };
                            let four_samples = capture_moving(4.0, 1.0)?;
                            if capture_moving(1.0, 1.0)? == four_samples {
                                return Err("Moving viewport sample count did not change the preview".into());
                            }
                            if capture_moving(4.0, 4.0)? == four_samples {
                                return Err("Moving viewport bounce count did not change the preview".into());
                            }
                            let denoise_extent = extent.map(|side| side.div_ceil(2));
                            let capture_denoise = |renderer: &mut crate::Renderer, sample, advance| -> Result<image::RgbaImage, String> {
                                let progress = Some(sample);
                                let data = ao_scene.camera_uniforms(size, progress, [0.0; 4], false);
                                let gui = ctx.run(egui::RawInput::default(), |_| {});
                                renderer.render(data, progress, advance, None, denoise_extent, gui, &ctx, true)
                                    .ok_or_else(|| "Denoised viewport unavailable".to_string())?
                            };
                            for backend in [crate::denoise::Backend::Cpu, crate::denoise::Backend::Gpu]
                                .into_iter().filter(|backend| backend.available())
                            {
                                renderer.viewport_denoise.configure(Some(backend), true);
                                capture_denoise(&mut renderer, 0, true)?;
                                let noisy = capture_denoise(&mut renderer, 1, true)?;
                                let (done, finished) = std::sync::mpsc::channel();
                                renderer.viewport_denoise.start(&renderer.device, &renderer.queue, &renderer.accumulation,
                                    denoise_extent, move || { let _ = done.send(()); });
                                finished.recv_timeout(Duration::from_secs(15)).map_err(|error| error.to_string())?;
                                renderer.poll_viewport_denoise();
                                let filtered = capture_denoise(&mut renderer, 2, false)?;
                                if let Some(error) = &renderer.viewport_denoise.error {
                                    return Err(error.clone());
                                }
                                if noisy == filtered || renderer.viewport_denoise.display.is_none() {
                                    return Err(format!("{} did not filter the viewport", backend.label()));
                                }
                                renderer.viewport_denoise.configure(None, true);
                                if noisy != capture_denoise(&mut renderer, 2, false)? {
                                    return Err("The original viewport was not restored after disabling denoising".into());
                                }
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
