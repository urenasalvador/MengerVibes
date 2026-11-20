pub const SENSOR_HEIGHT_MM: f32 = 24.0;
pub const DEFAULT_FOCAL_MM: f32 = 21.6;
const MAX_APERTURE_RADIUS: f32 = 0.05;
const MAX_F_STOP: f32 = 128.0;
// The lens uses one scene unit per metre. Aperture is stored as a radius.
const DIAMETER_MM_PER_SCENE_RADIUS: f32 = 2_000.0;

pub struct CameraEffects {
    pub focal: f32,
    pub aperture: f32,
    pub dof_enabled: bool,
    pub focus: f32,
    pub fog_density: f32,
    pub fog_start: f32,
    pub fog_color: [f32; 3],
}

impl Default for CameraEffects {
    fn default() -> Self {
        Self {
            focal: DEFAULT_FOCAL_MM,
            aperture: 0.001,
            dof_enabled: false,
            focus: 0.25,
            fog_density: 0.0,
            fog_start: 0.0,
            fog_color: [0.16, 0.20, 0.25],
        }
    }
}

impl CameraEffects {
    pub fn f_stop(&self) -> f32 {
        if self.aperture > 0.0 {
            self.focal / (DIAMETER_MM_PER_SCENE_RADIUS * self.aperture)
        } else {
            MAX_F_STOP
        }
    }

    pub fn f_stop_range(&self) -> std::ops::RangeInclusive<f32> {
        let min = self.focal / (DIAMETER_MM_PER_SCENE_RADIUS * MAX_APERTURE_RADIUS);
        min.min(self.f_stop())..=MAX_F_STOP.max(self.f_stop())
    }

    pub fn set_f_stop(&mut self, f_stop: f32) {
        self.aperture = (self.focal / (DIAMETER_MM_PER_SCENE_RADIUS * f_stop))
            .min(MAX_APERTURE_RADIUS);
    }

    pub fn set_focal(&mut self, focal: f32) {
        let f_stop = self.f_stop();
        self.focal = focal;
        if self.aperture > 0.0 {
            self.set_f_stop(f_stop);
        }
    }

    pub fn uniforms(&self, frame: [f32; 4]) -> [f32; 11] {
        [
            self.focal * 2.0 / SENSOR_HEIGHT_MM, self.focus,
            if self.dof_enabled { self.aperture } else { 0.0 },
            self.fog_density, self.fog_color[0], self.fog_color[1], self.fog_color[2],
            self.fog_start, frame[0], frame[1], frame[2],
        ]
    }

    pub fn preset(&self) -> [f32; 9] {
        [
            self.focal, self.aperture, self.focus, self.fog_density,
            self.fog_start, self.fog_color[0], self.fog_color[1], self.fog_color[2],
            f32::from(u8::from(self.dof_enabled)),
        ]
    }

    pub fn from_preset(values: &[f32]) -> Self {
        let [focal, aperture, focus, fog_density, fog_start, r, g, b, rest @ ..] = values else {
            return Self::default();
        };
        Self {
            focal: focal.clamp(8.0, 120.0),
            aperture: aperture.clamp(0.0, MAX_APERTURE_RADIUS),
            dof_enabled: rest.first().map_or(*aperture > 0.0, |enabled| *enabled != 0.0),
            focus: focus.clamp(0.0001, 30.0),
            fog_density: fog_density.clamp(0.0, 20.0),
            fog_start: fog_start.clamp(0.0, 30.0),
            fog_color: [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)],
        }
    }
}

pub fn render_frame(available: egui::Rect, aspect: f32) -> egui::Rect {
    let width = available.width().max(1.0).min(available.height().max(1.0) * aspect);
    egui::Rect::from_center_size(available.center(), egui::vec2(width, width / aspect))
}

pub fn viewport_projection(screen: egui::Rect, frame: egui::Rect) -> [f32; 4] {
    let center = frame.center() - screen.center();
    [
        center.x * 2.0 / screen.width().max(1.0),
        -center.y * 2.0 / screen.height().max(1.0),
        screen.height().max(1.0) / frame.height().max(1.0),
        0.0,
    ]
}
