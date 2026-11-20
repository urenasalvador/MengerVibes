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
    pub view: wgpu::TextureView,
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
        Ok(Self { path, view: upload(device, queue, width, height, pixels.as_raw()) })
    }
}

pub(super) fn upload(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, rgb: &[f32]) -> wgpu::TextureView {
    let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Environment panorama"), size, mip_level_count: 1, sample_count: 1,
        dimension: wgpu::TextureDimension::D2, format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let bytes: Vec<u8> = rgb.chunks_exact(3).flat_map(|pixel| {
        [pixel[0], pixel[1], pixel[2], 1.0].into_iter().flat_map(f32::to_le_bytes)
    }).collect();
    queue.write_texture(texture.as_image_copy(), &bytes, wgpu::TexelCopyBufferLayout {
        offset: 0, bytes_per_row: Some(width * 16), rows_per_image: Some(height),
    }, size);
    texture.create_view(&Default::default())
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
        ],
    })
}

pub(super) fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, camera: &wgpu::Buffer, view: &wgpu::TextureView) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera and HDRI"), layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: camera, offset: 0, size: wgpu::BufferSize::new(super::CAMERA_UNIFORM_BYTES),
            }) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
        ],
    })
}
