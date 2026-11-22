use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

const LIBRARY_NAME: &str = "OpenImageDenoise.dll";
pub const INSTALL_HINT: &str = "Extract the official Windows x64 OIDN 2.x package as oidn next to MengerVibes (oidn/bin/OpenImageDenoise.dll).";

fn library_path() -> Option<PathBuf> {
    let executable = std::env::current_exe().ok()?;
    Some(executable.parent()?.join("oidn/bin").join(LIBRARY_NAME))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self { Self::Cpu => "CPU", Self::Gpu => "GPU" }
    }

    pub fn available(self) -> bool {
        cfg!(windows) && library_path().is_some_and(|path| {
            path.is_file() && match self {
                Self::Cpu => path.with_file_name("OpenImageDenoise_device_cpu.dll").is_file(),
                Self::Gpu => path.with_file_name("OpenImageDenoise_device_cuda.dll").is_file(),
            }
        })
    }
}

pub fn selector(ui: &mut egui::Ui, label: &str, mode: &mut Option<Backend>) {
    egui::ComboBox::from_label(label).selected_text(mode.map_or("Off", Backend::label)).show_ui(ui, |ui| {
        ui.selectable_value(mode, None, "Off");
        for backend in [Backend::Cpu, Backend::Gpu] {
            ui.add_enabled_ui(backend.available(), |ui| {
                ui.selectable_value(mode, Some(backend), backend.label());
            }).response.on_disabled_hover_text(INSTALL_HINT);
        }
    });
}

pub fn guide_textures(device: &wgpu::Device, size: wgpu::Extent3d) -> [wgpu::Texture; 2] {
    ["Denoise albedo", "Denoise normals"].map(|label| device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label), size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    }))
}

pub struct Frame {
    width: u32,
    height: u32,
    planes: [Vec<[f32; 3]>; 3],
}

impl Frame {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        let pixels = width as usize * height as usize;
        let mut planes = std::array::from_fn(|_| Vec::new());
        for plane in &mut planes {
            plane.try_reserve_exact(pixels).map_err(|_| "Not enough memory for denoising passes".to_string())?;
            plane.resize(pixels, [0.0; 3]);
        }
        Ok(Self { width, height, planes })
    }

    pub fn write_tile(&mut self, mapped: &[u8], array_stride: usize, row_stride: usize, origin: [u32; 2], size: [u32; 2]) {
        let [width, height] = size.map(|x| x as usize);
        let [x, y] = origin.map(|x| x as usize);
        for (plane_index, plane) in self.planes.iter_mut().enumerate() {
            for local_y in 0..height {
                let source = plane_index * array_stride + local_y * row_stride;
                let dest = (y + local_y) * self.width as usize + x;
                for (pixel, packed) in plane[dest..dest + width].iter_mut().zip(mapped[source..source + width * 8].chunks_exact(8)) {
                    for (output, channel) in pixel.iter_mut().zip(packed.chunks_exact(2)) {
                        let value = half::f16::from_le_bytes([channel[0], channel[1]]).to_f32();
                        // OIDN expects nonnegative HDR and albedo, but signed world-space normals.
                        *output = if !value.is_finite() { 0.0 } else {
                            match plane_index {
                                0 => value.max(0.0),
                                1 => value.clamp(0.0, 1.0),
                                _ => value.clamp(-1.0, 1.0),
                            }
                        };
                    }
                }
            }
        }
    }

    pub fn finish(mut self, runtime: &Runtime, contrast: f32, cancel: &AtomicBool) -> Result<image::RgbaImage, String> {
        runtime.filter(&mut self, cancel)?;
        self.into_png(contrast)
    }

    pub fn finish_viewport(mut self, backend: Backend, cancel: &AtomicBool) -> Result<(Vec<u8>, std::time::Instant), String> {
        Runtime::load(backend)?.filter(&mut self, cancel)?;
        let filtered_at = std::time::Instant::now();
        let [color, _, _] = self.planes;
        let mut packed = Vec::new();
        packed.try_reserve_exact(color.len() * 8).map_err(|_| "Not enough memory for the denoised viewport".to_string())?;
        for [r, g, b] in color {
            for value in [r, g, b, 1.0] {
                packed.extend(half::f16::from_f32(value.clamp(0.0, 65504.0)).to_le_bytes());
            }
        }
        Ok((packed, filtered_at))
    }

    pub(super) fn into_png(self, contrast: f32) -> Result<image::RgbaImage, String> {
        let [color, _, _] = self.planes;
        let mut rgba = Vec::new();
        rgba.try_reserve_exact(color.len() * 4).map_err(|_| "Not enough memory for the denoised PNG".to_string())?;
        for pixel in color {
            rgba.extend(display_pixel(pixel, contrast));
        }
        image::RgbaImage::from_raw(self.width, self.height, rgba).ok_or_else(|| "Invalid output dimensions".into())
    }

    #[cfg(test)]
    pub(super) fn guides(&self) -> [&[[f32; 3]]; 2] {
        [&self.planes[1], &self.planes[2]]
    }
}

// Keep this conversion in sync with contrast.wgsl and the sRGB export target.
fn display_pixel(mut rgb: [f32; 3], contrast: f32) -> [u8; 4] {
    if contrast > 0.0 && contrast != 1.0 {
        let luma = (rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722).max(0.00001);
        let t = ((luma - 0.35) / 0.65).clamp(0.0, 1.0);
        let blend = t * t * (3.0 - 2.0 * t);
        let ratio = (luma / 0.18).clamp(0.0001, 3.0) * (1.0 - blend) + blend;
        rgb = rgb.map(|channel| channel * ratio.powf(contrast - 1.0));
    }
    let [r, g, b] = rgb.map(|channel| {
        let linear = channel.clamp(0.0, 1.0);
        let srgb = if linear <= 0.0031308 { 12.92 * linear } else { 1.055 * linear.powf(1.0 / 2.4) - 0.055 };
        (srgb * 255.0).round() as u8
    });
    [r, g, b, 255]
}

#[cfg(windows)]
mod native {
    use super::*;
    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};
    use std::ffi::{CStr, c_char, c_int, c_void};
    use std::sync::atomic::Ordering;

    const CPU_DEVICE: c_int = 1;
    const CUDA_DEVICE: c_int = 3;
    const FLOAT3: c_int = 3;
    type OidnHandle = *mut c_void;
    type Release = unsafe extern "C" fn(OidnHandle);
    type Monitor = unsafe extern "C" fn(*mut c_void, f64) -> bool;

    struct Handle(OidnHandle, Release);
    impl Drop for Handle {
        fn drop(&mut self) { unsafe { (self.1)(self.0); } }
    }

    unsafe extern "C" fn monitor(data: *mut c_void, _: f64) -> bool {
        // oidnExecuteFilter completes its callbacks before returning.
        !unsafe { &*data.cast::<AtomicBool>() }.load(Ordering::Relaxed)
    }

    pub struct Runtime {
        library: Library,
        backend: Backend,
    }

    impl Runtime {
        pub fn load(backend: Backend) -> Result<Self, String> {
            if !backend.available() { return Err(INSTALL_HINT.into()); }
            let path = library_path().filter(|path| path.is_file()).ok_or(INSTALL_HINT)?;
            // Load the chosen package by absolute path, with dependencies resolved beside its DLL.
            let library = unsafe { Library::load_with_flags(path, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) }
                .map_err(|error| format!("Could not load OIDN: {error}"))?;
            Ok(Self { library, backend })
        }

        pub fn filter(&self, frame: &mut Frame, cancel: &AtomicBool) -> Result<(), String> {
            macro_rules! symbol {
                ($name:literal, $ty:ty) => {
                    *unsafe { self.library.get::<$ty>(concat!($name, "\0").as_bytes()) }
                        .map_err(|error| format!("OIDN 2.x API ({}): {error}", $name))?
                };
            }
            let new_device = symbol!("oidnNewDevice", unsafe extern "C" fn(c_int) -> OidnHandle);
            let commit_device = symbol!("oidnCommitDevice", unsafe extern "C" fn(OidnHandle));
            let release_device = symbol!("oidnReleaseDevice", Release);
            let get_error = symbol!("oidnGetDeviceError", unsafe extern "C" fn(OidnHandle, *mut *const c_char) -> c_int);
            let new_filter = symbol!("oidnNewFilter", unsafe extern "C" fn(OidnHandle, *const c_char) -> OidnHandle);
            let release_filter = symbol!("oidnReleaseFilter", Release);
            let set_image = symbol!("oidnSetSharedFilterImage", unsafe extern "C" fn(OidnHandle, *const c_char, *mut c_void, c_int, usize, usize, usize, usize, usize));
            let set_bool = symbol!("oidnSetFilterBool", unsafe extern "C" fn(OidnHandle, *const c_char, bool));
            let set_monitor = symbol!("oidnSetFilterProgressMonitorFunction", unsafe extern "C" fn(OidnHandle, Monitor, *mut c_void));
            let commit_filter = symbol!("oidnCommitFilter", unsafe extern "C" fn(OidnHandle));
            let execute_filter = symbol!("oidnExecuteFilter", unsafe extern "C" fn(OidnHandle));
            let new_buffer = symbol!("oidnNewBuffer", unsafe extern "C" fn(OidnHandle, usize) -> OidnHandle);
            let release_buffer = symbol!("oidnReleaseBuffer", Release);
            let write_buffer = symbol!("oidnWriteBuffer", unsafe extern "C" fn(OidnHandle, usize, usize, *const c_void));
            let read_buffer = symbol!("oidnReadBuffer", unsafe extern "C" fn(OidnHandle, usize, usize, *mut c_void));
            let set_buffer = symbol!("oidnSetFilterImage", unsafe extern "C" fn(OidnHandle, *const c_char, OidnHandle, c_int, usize, usize, usize, usize, usize));

            let check = |device| {
                let mut message = std::ptr::null();
                let code = unsafe { get_error(device, &mut message) };
                if code == 0 { return Ok(()); }
                let error = if message.is_null() { format!("error {code}") } else {
                    unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned()
                };
                Err(format!("OIDN {}: {error}", self.backend.label()))
            };

            if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
            // The three packed Float3 buffers stay allocated for the entire synchronous filter run.
            unsafe {
                let kind = if self.backend == Backend::Cpu { CPU_DEVICE } else { CUDA_DEVICE };
                let device = new_device(kind);
                if device.is_null() { return check(device).and(Err(format!("Could not create the OIDN {} device", self.backend.label()))); }
                let device = Handle(device, release_device);
                commit_device(device.0);
                check(device.0)?;
                let filter = new_filter(device.0, c"RT".as_ptr());
                if filter.is_null() { return check(device.0).and(Err("Could not create the OIDN RT filter".into())); }
                let filter = Handle(filter, release_filter);
                let [width, height] = [frame.width as usize, frame.height as usize];
                let mut buffers = Vec::new();
                let bytes = width * height * std::mem::size_of::<[f32; 3]>();
                if self.backend == Backend::Gpu {
                    for plane in &frame.planes {
                        let buffer = new_buffer(device.0, bytes);
                        if buffer.is_null() { return check(device.0).and(Err("Could not allocate an OIDN GPU buffer".into())); }
                        let buffer = Handle(buffer, release_buffer);
                        write_buffer(buffer.0, 0, bytes, plane.as_ptr().cast());
                        check(device.0)?;
                        buffers.push(buffer);
                    }
                    for (name, index) in [(c"color", 0), (c"albedo", 1), (c"normal", 2), (c"output", 0)] {
                        set_buffer(filter.0, name.as_ptr(), buffers[index].0, FLOAT3, width, height, 0, 0, 0);
                        check(device.0)?;
                    }
                } else {
                    let [color, albedo, normal] = &mut frame.planes;
                    let color = color.as_mut_ptr().cast();
                    for (name, data) in [(c"color", color), (c"albedo", albedo.as_mut_ptr().cast()),
                        (c"normal", normal.as_mut_ptr().cast()), (c"output", color)] {
                        set_image(filter.0, name.as_ptr(), data, FLOAT3, width, height, 0, 0, 0);
                        check(device.0)?;
                    }
                }
                set_bool(filter.0, c"hdr".as_ptr(), true);
                set_monitor(filter.0, monitor, std::ptr::from_ref(cancel).cast_mut().cast());
                commit_filter(filter.0);
                check(device.0)?;
                execute_filter(filter.0);
                if cancel.load(Ordering::Relaxed) { return Err("Render cancelled".into()); }
                check(device.0)?;
                if let Some(color) = buffers.first() {
                    read_buffer(color.0, 0, bytes, frame.planes[0].as_mut_ptr().cast());
                }
                check(device.0)
            }
        }
    }
}

#[cfg(windows)]
pub use native::Runtime;

#[cfg(not(windows))]
pub struct Runtime;

#[cfg(not(windows))]
impl Runtime {
    pub fn load(_: Backend) -> Result<Self, String> { Err(INSTALL_HINT.into()) }
    fn filter(&self, _: &mut Frame, _: &AtomicBool) -> Result<(), String> { Err(INSTALL_HINT.into()) }
}
