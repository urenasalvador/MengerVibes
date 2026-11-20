use std::{
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
};

const MIN_CARD_WIDTH: f32 = 64.0;
const CARD_GAP: f32 = 8.0;
const DEFAULT_CARD_WIDTH: f32 = 164.0;
const CARD_SCROLL_ID: &str = "preset_cards";

pub enum Action {
    Add,
    ResetDefaults,
    Load(u64),
    Update(u64),
    Remove(u64),
    Rename(u64, String),
}

struct Preset {
    id: u64,
    name: String,
    has_preview: bool,
    thumbnail: Option<egui::TextureHandle>,
}

pub struct Gallery {
    directory: PathBuf,
    entries: Vec<Preset>,
    last_id: u64,
    card_width: f32,
    editing_name: Option<(u64, String)>,
    confirming_removal: Option<u64>,
    scroll_to_end: bool,
    visible_cards: std::ops::Range<usize>,
}

impl Default for Gallery {
    fn default() -> Self {
        Self {
            directory: PathBuf::from("presets"),
            entries: Vec::new(),
            last_id: 0,
            card_width: DEFAULT_CARD_WIDTH,
            editing_name: None,
            confirming_removal: None,
            scroll_to_end: false,
            visible_cards: 0..0,
        }
    }
}

impl Gallery {
    fn id_from_path(path: &Path) -> Option<u64> {
        let id: u64 = path
            .file_stem()?
            .to_str()?
            .strip_prefix("slot_")?
            .parse()
            .ok()?;
        (id > 0 && path.file_stem()?.to_str()? == format!("slot_{id}")).then_some(id)
    }

    pub fn scan(&mut self) -> io::Result<()> {
        let files = match fs::read_dir(&self.directory) {
            Ok(files) => files,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => return Err(error),
        };
        let mut entries = Vec::new();
        let mut last_id = 0;
        for file in files {
            let file = file?;
            if !file.file_type()?.is_file() {
                continue;
            }
            let path = file.path();
            let Some(id) = Self::id_from_path(&path) else {
                continue;
            };
            last_id = last_id.max(id);
            if path.extension().and_then(|extension| extension.to_str()) == Some("txt") {
                let name = fs::read_to_string(self.file(id, "name")).unwrap_or_default();
                entries.push(Preset {
                    id,
                    name: name.trim().to_owned(),
                    has_preview: self.file(id, "png").is_file(),
                    thumbnail: None,
                });
            }
        }
        entries.sort_unstable_by_key(|preset| preset.id);
        self.entries = entries;
        self.visible_cards = 0..0;
        self.last_id = last_id;
        self.card_width = fs::read_to_string(self.directory.join("card_width.txt"))
            .ok()
            .and_then(|width| width.trim().parse::<f32>().ok())
            .filter(|width| width.is_finite())
            .map(|width| width.max(MIN_CARD_WIDTH))
            .unwrap_or(DEFAULT_CARD_WIDTH);
        Ok(())
    }

    pub fn file(&self, id: u64, extension: &str) -> PathBuf {
        self.directory.join(format!("slot_{id}.{extension}"))
    }

    pub fn add(&mut self, contents: &str) -> io::Result<u64> {
        fs::create_dir_all(&self.directory)?;
        loop {
            let id = self
                .last_id
                .checked_add(1)
                .ok_or_else(|| io::Error::other("Preset identifiers exhausted"))?;
            self.last_id = id;
            let path = self.file(id, "txt");
            if self.file(id, "removed").exists() {
                continue;
            }
            let mut file = match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => file,
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            };
            if let Err(error) = file.write_all(contents.as_bytes()) {
                drop(file);
                let _ = fs::remove_file(&path);
                return Err(error);
            }
            self.entries.push(Preset {
                id,
                name: String::new(),
                has_preview: false,
                thumbnail: None,
            });
            self.scroll_to_end = true;
            return Ok(id);
        }
    }

    pub fn remove(&mut self, id: u64) -> io::Result<()> {
        // Moving the settings out of the gallery keeps the original preset recoverable.
        let removed = self.file(id, "removed");
        if removed.exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "A removed copy already exists",
            ));
        }
        fs::rename(self.file(id, "txt"), removed)?;
        self.keep_previews(0..0);
        self.entries.retain(|entry| entry.id != id);
        Ok(())
    }

    pub fn rename(&mut self, id: u64, name: String) -> io::Result<()> {
        let name = name.trim();
        fs::write(self.file(id, "name"), name)?;
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.name = name.to_owned();
        }
        Ok(())
    }

    pub fn preview_saved(&mut self, id: u64) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.has_preview = true;
            entry.thumbnail = None;
        }
    }

    fn keep_previews(&mut self, visible: std::ops::Range<usize>) {
        for index in self.visible_cards.clone() {
            if !visible.contains(&index) {
                self.entries[index].thumbnail = None;
            }
        }
        self.visible_cards = visible;
    }

    fn save_card_width(&self) -> io::Result<()> {
        fs::create_dir_all(&self.directory)?;
        fs::write(
            self.directory.join("card_width.txt"),
            self.card_width.to_string(),
        )
    }

    pub fn hide(&mut self) {
        self.keep_previews(0..0);
    }

    pub fn ui(&mut self, ui: &mut egui::Ui, status: &str) -> Option<Action> {
        let mut action = None;
        ui.horizontal_wrapped(|ui| {
            if ui.button("+ Add preset").on_hover_text("Save the current view as a new preset").clicked() {
                action = Some(Action::Add);
            }
            if ui.button("Reset to Defaults")
                .on_hover_text("Restore the starting scene, camera and render settings")
                .clicked()
            {
                action = Some(Action::ResetDefaults);
            }
        });

        // Reserve the scrollbar's width even during its first frame.
        ui.spacing_mut().scroll.floating = false;
        let scrollbar_width = ui.spacing().scroll.allocated_width();
        let max_width = (ui.available_width() - scrollbar_width).floor().max(1.0);
        let mut card_width = self.card_width.min(max_width);
        ui.horizontal(|ui| {
            ui.label("Card width");
            ui.spacing_mut().slider_width = 76.0;
            let width = ui.add(
                egui::Slider::new(&mut card_width, MIN_CARD_WIDTH.min(max_width)..=max_width)
                    .step_by(1.0)
                    .suffix(" px"),
            );
            if width.changed() {
                self.card_width = card_width;
            }
            if width.drag_stopped() || (width.changed() && !width.dragged()) {
                if let Err(error) = self.save_card_width() {
                    eprintln!("Could not save the preset card width: {error}");
                }
            }
        });

        if let Some(id) = self.confirming_removal {
            ui.horizontal_wrapped(|ui| {
                ui.label("Remove preset?");
                if ui.small_button("Remove").clicked() {
                    action = Some(Action::Remove(id));
                    self.confirming_removal = None;
                }
                if ui.small_button("Cancel").clicked() {
                    self.confirming_removal = None;
                }
            });
        } else if let Some((id, text)) = &mut self.editing_name {
            let mut finish_rename = false;
            ui.horizontal(|ui| {
                let edit = ui.add(
                    egui::TextEdit::singleline(text)
                        .desired_width((ui.available_width() - 55.0).max(1.0))
                        .char_limit(80),
                );
                if ui.small_button("OK").clicked()
                    || (edit.lost_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter)))
                {
                    action = Some(Action::Rename(*id, std::mem::take(text)));
                    finish_rename = true;
                } else if ui
                    .small_button("×")
                    .on_hover_text("Cancel rename")
                    .clicked()
                    || ui.input(|input| input.key_pressed(egui::Key::Escape))
                {
                    finish_rename = true;
                }
            });
            if finish_rename {
                self.editing_name = None;
            }
        }

        if !status.is_empty() {
            ui.add(egui::Label::new(egui::RichText::new(status).small()).truncate())
                .on_hover_text(status);
        }
        ui.separator();

        if self.entries.is_empty() {
            ui.weak("No presets yet");
            return action;
        }

        let columns = ((max_width + CARD_GAP) / (card_width + CARD_GAP))
            .floor()
            .max(1.0) as usize;
        let rows = self.entries.len().div_ceil(columns);
        let thumbnail_height = card_width * 0.6;
        let card_size = egui::vec2(card_width, thumbnail_height + 26.0);
        let row_stride = card_size.y + CARD_GAP;
        let content_height = row_stride * rows as f32 - CARD_GAP;
        let max_offset = (content_height - ui.available_height()).max(0.0);
        let scroll_id = ui.make_persistent_id(egui::Id::new(CARD_SCROLL_ID));
        let offset =
            egui::scroll_area::State::load(ui.ctx(), scroll_id).map_or(0.0, |state| state.offset.y);
        let mut scroll = egui::ScrollArea::vertical()
            .id_salt(CARD_SCROLL_ID)
            .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysVisible)
            .auto_shrink([false, false]);
        if std::mem::take(&mut self.scroll_to_end) || offset > max_offset {
            scroll = scroll.vertical_scroll_offset(max_offset);
        }
        scroll.show_viewport(ui, |ui, viewport| {
            ui.set_min_height(content_height);
            let first_row = (viewport.min.y / row_stride).floor().max(0.0) as usize;
            let last_row = ((viewport.max.y / row_stride).ceil().max(0.0) as usize + 1).min(rows);
            let end = (last_row * columns).min(self.entries.len());
            let visible = (first_row * columns).min(end)..end;
            self.keep_previews(visible.clone());
            let origin = ui.max_rect().min;
            for index in visible {
                let offset = egui::vec2(
                    (index % columns) as f32 * (card_width + CARD_GAP),
                    (index / columns) as f32 * row_stride,
                );
                let rect = egui::Rect::from_min_size(origin + offset, card_size);
                let id = self.entries[index].id;
                ui.scope_builder(egui::UiBuilder::new().id_salt(id).max_rect(rect), |ui| {
                    ui.set_clip_rect(ui.clip_rect().intersect(rect));
                    if let Some(card_action) = self.card_ui(ui, index, card_width, thumbnail_height)
                    {
                        action = Some(card_action);
                    }
                });
            }
        });
        action
    }

    fn card_ui(
        &mut self,
        ui: &mut egui::Ui,
        index: usize,
        width: f32,
        height: f32,
    ) -> Option<Action> {
        let id = self.entries[index].id;
        let image_path = self.file(id, "png");
        let entry = &mut self.entries[index];
        if entry.has_preview && entry.thumbnail.is_none() {
            if let Ok(image) = image::open(image_path) {
                let pixels = image.to_rgba8();
                let color = egui::ColorImage::from_rgba_unmultiplied(
                    [pixels.width() as usize, pixels.height() as usize],
                    pixels.as_raw(),
                );
                entry.thumbnail = Some(ui.ctx().load_texture(
                    format!("preset_{id}"),
                    color,
                    egui::TextureOptions::LINEAR,
                ));
            } else {
                entry.has_preview = false;
            }
        }
        let name = if entry.name.is_empty() {
            format!("Preset {id}")
        } else {
            entry.name.clone()
        };
        let size = egui::vec2(width, height);
        let (rect, preview) = ui.allocate_exact_size(size, egui::Sense::click());
        let style = *ui.style().interact(&preview);
        ui.painter().rect_filled(rect, 4.0, style.weak_bg_fill);
        if let Some(texture) = &entry.thumbnail {
            let scale = (width / texture.size_vec2().x).min(height / texture.size_vec2().y);
            let fit = egui::Rect::from_center_size(rect.center(), scale * texture.size_vec2());
            ui.painter().image(
                texture.id(),
                fit,
                egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                egui::Color32::WHITE,
            );
        } else {
            ui.painter().text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "No preview",
                egui::TextStyle::Small.resolve(ui.style()),
                ui.visuals().weak_text_color(),
            );
        }
        ui.painter()
            .rect_stroke(rect, 4.0, style.bg_stroke, egui::StrokeKind::Inside);
        let mut action = preview
            .on_hover_text(format!("Load {name}"))
            .clicked()
            .then_some(Action::Load(id));
        ui.allocate_ui_with_layout(
            egui::vec2(width, 0.0),
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| {
                ui.menu_button("⋯", |ui| {
                    if ui.button("Load").clicked() {
                        action = Some(Action::Load(id));
                        ui.close_menu();
                    }
                    if ui.button("Rename").clicked() {
                        self.confirming_removal = None;
                        self.editing_name = Some((id, name.clone()));
                        ui.close_menu();
                    }
                    if ui.button("Update from current view").clicked() {
                        action = Some(Action::Update(id));
                        ui.close_menu();
                    }
                    if ui.button("Remove…").clicked() {
                        self.editing_name = None;
                        self.confirming_removal = Some(id);
                        ui.close_menu();
                    }
                });
                ui.add(egui::Label::new(&name).truncate())
                    .on_hover_text(&name);
            },
        );
        action
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gallery_keeps_legacy_presets_and_persists_add_remove_name_and_width() {
        let directory =
            std::env::temp_dir().join(format!("sdf-preset-gallery-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let mut gallery = Gallery {
            directory: directory.clone(),
            ..Gallery::default()
        };
        for id in 1..=5 {
            fs::write(gallery.file(id, "txt"), format!("legacy {id}")).unwrap();
        }
        gallery.scan().unwrap();
        assert_eq!(gallery.entries.len(), 5);
        for id in 6..=35 {
            assert_eq!(gallery.add(&format!("view {id}")).unwrap(), id);
        }
        gallery.rename(35, "Tunnel".into()).unwrap();
        gallery.remove(3).unwrap();
        gallery.remove(34).unwrap();
        gallery.card_width = 238.0;
        gallery.save_card_width().unwrap();

        let mut reopened = Gallery {
            directory: directory.clone(),
            ..Gallery::default()
        };
        reopened.scan().unwrap();
        let ids: Vec<_> = reopened.entries.iter().map(|entry| entry.id).collect();
        assert_eq!(
            ids,
            (1..=35)
                .filter(|id| ![3, 34].contains(id))
                .collect::<Vec<_>>()
        );
        assert_eq!(reopened.entries.last().unwrap().name, "Tunnel");
        assert_eq!(reopened.card_width, 238.0);
        assert_eq!(
            fs::read_to_string(reopened.file(3, "removed")).unwrap(),
            "legacy 3"
        );
        assert_eq!(
            fs::read_to_string(reopened.file(5, "txt")).unwrap(),
            "legacy 5"
        );
        assert_eq!(reopened.add("another view").unwrap(), 36);

        let ctx = egui::Context::default();
        let render = |gallery: &mut Gallery, panel_width: f32| {
            let mut panel_right = 0.0;
            let output = ctx.run(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800.0, 450.0),
                    )),
                    ..Default::default()
                },
                |ctx| {
                    egui::SidePanel::left("test_presets")
                        .exact_width(panel_width)
                        .show(ctx, |ui| {
                            gallery.ui(ui, "");
                        });
                    panel_right = ctx.available_rect().left();
                },
            );
            assert!(
                (panel_right - panel_width).abs() <= 1.0,
                "Panel grew to {panel_right}"
            );
            output
                .shapes
                .iter()
                .filter_map(|shape| match &shape.shape {
                    egui::Shape::Rect(rect)
                        if rect.rect.width() > 0.0
                            && (rect.rect.height() / rect.rect.width() - 0.6).abs() < 0.01 =>
                    {
                        assert!(
                            rect.rect.right() <= panel_right,
                            "Card overflowed the left panel"
                        );
                        Some(rect.rect)
                    }
                    _ => None,
                })
                .collect::<Vec<_>>()
        };
        let wide_cards = render(&mut reopened, 240.0);
        assert!(!wide_cards.is_empty());
        assert!(
            wide_cards
                .iter()
                .all(|card| card.width() > 188.0 && card.width() <= 240.0)
        );
        assert_eq!(reopened.visible_cards.end, reopened.entries.len());
        assert!(reopened.visible_cards.len() < 10);

        reopened.card_width = MIN_CARD_WIDTH;
        let small_cards = render(&mut reopened, 240.0);
        assert!(
            small_cards
                .iter()
                .all(|card| (card.width() - MIN_CARD_WIDTH).abs() < 1.0)
        );
        assert!(
            small_cards
                .iter()
                .any(|card| card.left() > small_cards[0].left()),
            "Expected multiple columns"
        );
        assert!(
            small_cards
                .iter()
                .any(|card| card.top() != small_cards[0].top()),
            "Expected multiple rows"
        );

        reopened.card_width = 500.0;
        let resized_cards = render(&mut reopened, 360.0);
        assert!(
            resized_cards
                .iter()
                .all(|card| card.width() > 300.0 && card.width() <= 360.0)
        );
        assert!(!resized_cards.is_empty());
        assert_eq!(reopened.card_width, 500.0);

        for entry in fs::read_dir(&directory).unwrap() {
            fs::remove_file(entry.unwrap().path()).unwrap();
        }
        fs::remove_dir(&directory).unwrap();
    }
}
