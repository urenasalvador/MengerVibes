// SPDX-License-Identifier: GPL-3.0-only

use std::path::{Path, PathBuf};

#[cfg(windows)]
pub(super) fn choose_file(window: Option<&winit::window::Window>) -> Result<Option<PathBuf>, String> {
    use std::os::windows::ffi::OsStringExt;
    use windows_sys::Win32::UI::Controls::Dialogs::{
        CommDlgExtendedError, GetOpenFileNameW, OFN_EXPLORER, OFN_FILEMUSTEXIST, OFN_NOCHANGEDIR,
        OFN_PATHMUSTEXIST, OPENFILENAMEW,
    };
    use winit::raw_window_handle::{HasWindowHandle, RawWindowHandle};

    let owner = match window.and_then(|window| window.window_handle().ok()).map(|handle| handle.as_raw()) {
        Some(RawWindowHandle::Win32(handle)) => handle.hwnd.get() as _,
        _ => std::ptr::null_mut(),
    };
    let filter: Vec<u16> = "Radiance HDR (*.hdr)\0*.hdr\0\0".encode_utf16().collect();
    let title: Vec<u16> = "Open HDRI panorama\0".encode_utf16().collect();
    let mut file = vec![0u16; 32768];
    let mut dialog = OPENFILENAMEW {
        lStructSize: std::mem::size_of::<OPENFILENAMEW>() as u32,
        hwndOwner: owner, lpstrFilter: filter.as_ptr(), nFilterIndex: 1,
        lpstrFile: file.as_mut_ptr(), nMaxFile: file.len() as u32, lpstrTitle: title.as_ptr(),
        Flags: OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR,
        ..Default::default()
    };
    // The dialog uses the UTF-16 buffers synchronously and writes the selected path into file.
    if unsafe { GetOpenFileNameW(&mut dialog) } == 0 {
        let error = unsafe { CommDlgExtendedError() };
        return if error == 0 { Ok(None) } else { Err(format!("File dialog error 0x{error:04X}")) };
    }
    let end = file.iter().position(|&unit| unit == 0).unwrap_or(file.len());
    Ok(Some(std::ffi::OsString::from_wide(&file[..end]).into()))
}

#[derive(Clone, PartialEq)]
pub(super) struct Panorama {
    pub path: PathBuf,
    pub texture: Environment,
}

#[derive(Clone, PartialEq)]
pub(super) struct Environment {
    view: wgpu::TextureView,
    importance: wgpu::TextureView,
}

#[cfg(test)]
impl Environment {
    pub fn without_importance(&self, fallback: &Self) -> Self {
        Self { view: self.view.clone(), importance: fallback.importance.clone() }
    }
}

impl Panorama {
    pub fn load(device: &wgpu::Device, queue: &wgpu::Queue, path: &Path) -> Result<Self, String> {
        let path = std::path::absolute(path).map_err(|error| error.to_string())?;
        let mut reader = image::ImageReader::open(&path).map_err(|error| error.to_string())?;
        reader.set_format(image::ImageFormat::Hdr);
        let cap = device.limits().max_texture_dimension_2d.min(8192);
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(cap);
        limits.max_image_height = Some(cap / 2);
        reader.limits(limits);
        let pixels = reader.decode().map_err(|error| error.to_string())?.into_rgb32f();
        let (width, height) = pixels.dimensions();
        if width == 0 || height == 0 || width != height * 2 {
            return Err("Expected a 2:1 equirectangular Radiance HDR image.".into());
        }
        Ok(Self { path, texture: upload(device, queue, width, height, pixels.as_raw()) })
    }
}

pub(super) fn upload(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, rgb: &[f32]) -> Environment {
    let [columns, rows] = [width.min(512), height.min(256)];
    let table = importance_table(width, height, columns, rows, rgb);
    let importance = match table {
        Some(entries) => upload_rgba(device, queue, "Environment importance", columns, rows, entries.into_iter().flatten()),
        None => upload_rgba(device, queue, "Environment importance disabled", 1, 1, [0.0; 4]),
    };
    let rgba = rgb.chunks_exact(3).flat_map(|pixel| [pixel[0], pixel[1], pixel[2], 1.0]);
    let view = upload_rgba(device, queue, "Environment panorama", width, height, rgba);
    Environment { view, importance }
}

fn upload_rgba(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, width: u32, height: u32,
    values: impl IntoIterator<Item = f32>,
) -> wgpu::TextureView {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label), size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let bytes: Vec<u8> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    queue.write_texture(texture.as_image_copy(), &bytes, wgpu::TexelCopyBufferLayout {
        offset: 0, bytes_per_row: Some(width * 16), rows_per_image: Some(height),
    }, size);
    texture.create_view(&Default::default())
}

fn cell_solid_angle(row: u32, columns: u32, rows: u32) -> f64 {
    let step = std::f64::consts::PI / f64::from(rows);
    let polar = f64::from(row) * step;
    (std::f64::consts::TAU / f64::from(columns)) * (polar.cos() - (polar + step).cos())
}

fn importance_table(width: u32, height: u32, columns: u32, rows: u32, rgb: &[f32]) -> Option<Vec<[f32; 4]>> {
    let count = (columns * rows) as usize;
    let mut flux = vec![0.0; count];
    let mut max_luminance = 0.0f64;
    for (y, scanline) in rgb.chunks_exact(width as usize * 3).enumerate() {
        let solid_angle = cell_solid_angle(y as u32, width, height);
        let row = y as u32 * rows / height;
        for (x, color) in scanline.chunks_exact(3).enumerate() {
            let luminance = f64::from(color[0]) * 0.2126 + f64::from(color[1]) * 0.7152 + f64::from(color[2]) * 0.0722;
            max_luminance = max_luminance.max(luminance);
            let column = x as u32 * columns / width;
            flux[(row * columns + column) as usize] += luminance * solid_angle;
        }
    }
    let total: f64 = flux.iter().sum();
    let sphere = 4.0 * std::f64::consts::PI;
    if count == 1 || !total.is_finite() || total <= 0.0 || max_luminance <= 1.25 * total / sphere {
        return None;
    }
    // Leave a small uniform component for directions outside bright cells.
    let mut entries = Vec::with_capacity(count);
    let mut scaled = Vec::with_capacity(count);
    let mut small = Vec::new();
    let mut large = Vec::new();
    for (index, flux) in flux.into_iter().enumerate() {
        let solid_angle = cell_solid_angle(index as u32 / columns, columns, rows);
        let mass = 0.98 * flux / total + 0.02 * solid_angle / sphere;
        entries.push([1.0, index as f32, (mass / solid_angle) as f32, 0.0]);
        let scaled_mass = mass * count as f64;
        scaled.push(scaled_mass);
        if scaled_mass < 1.0 { &mut small } else { &mut large }.push(index);
    }
    // Vose alias table: each bucket stores its acceptance probability, alias and solid-angle PDF.
    while !small.is_empty() && !large.is_empty() {
        let lower = small.pop().unwrap();
        let upper = large.pop().unwrap();
        entries[lower][0] = scaled[lower] as f32;
        entries[lower][1] = upper as f32;
        scaled[upper] -= 1.0 - scaled[lower];
        if scaled[upper] < 1.0 { &mut small } else { &mut large }.push(upper);
    }
    Some(entries)
}

pub(super) fn layout(device: &wgpu::Device, dynamic: bool) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Camera and HDRI"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: dynamic,
                    min_binding_size: wgpu::BufferSize::new(super::CAMERA_UNIFORM_BYTES),
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::FRAGMENT, count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false,
                },
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3, visibility: wgpu::ShaderStages::FRAGMENT, count: None,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                },
            },
        ],
    })
}

pub(super) fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, camera: &wgpu::Buffer, environment: &Environment, blue_noise: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera and HDRI"), layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: camera, offset: 0, size: wgpu::BufferSize::new(super::CAMERA_UNIFORM_BYTES),
            }) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&environment.view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(blue_noise) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&environment.importance) },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn importance_alias_matches_solid_angle_pdf() {
        let (columns, rows) = (8, 4);
        let count = (columns * rows) as usize;
        let mut panorama = vec![0.0; 3 * count];
        let bright = 10;
        panorama[bright * 3..bright * 3 + 3].fill(100.0);
        let table = importance_table(columns, rows, columns, rows, &panorama).unwrap();
        let mut sampled_mass = vec![0.0; count];
        for (index, entry) in table.iter().enumerate() {
            sampled_mass[index] += f64::from(entry[0]) / count as f64;
            sampled_mass[entry[1] as usize] += (1.0 - f64::from(entry[0])) / count as f64;
        }
        let mut integrated = 0.0;
        for (index, entry) in table.iter().enumerate() {
            let mass = f64::from(entry[2]) * cell_solid_angle(index as u32 / columns, columns, rows);
            assert!(mass > 0.0);
            assert!((mass - sampled_mass[index]).abs() < 1e-6);
            integrated += mass;
        }
        assert!((integrated - 1.0).abs() < 1e-6);
        assert!(sampled_mass[bright] > 0.98);
        assert!(sampled_mass[0] < sampled_mass[8]);
        panorama.fill(1.0);
        assert!(importance_table(columns, rows, columns, rows, &panorama).is_none());
    }
}
