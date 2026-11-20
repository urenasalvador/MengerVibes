use std::collections::VecDeque;

use super::{App, PRESET_VALUE_COUNT};

const HISTORY_LIMIT: usize = 128;

#[derive(Clone, PartialEq)]
struct Snapshot {
    scene: [f32; PRESET_VALUE_COUNT],
    output_size: [u32; 2],
    output_samples: u32,
    show_frame: bool,
}

impl Snapshot {
    fn capture(app: &App) -> Self {
        Self {
            scene: app.preset_values(),
            output_size: [app.output.width, app.output.height],
            output_samples: app.output.samples,
            show_frame: app.show_frame,
        }
    }

    fn restore(self, app: &mut App) {
        app.restore_scene(&self.scene);
        [app.output.width, app.output.height] = self.output_size;
        app.output.samples = self.output_samples;
        app.show_frame = self.show_frame;
        app.keys.clear();
        app.looking = false;
        app.focus_point = None;
    }
}

#[derive(Clone, Copy)]
pub(super) enum Action {
    Undo,
    Redo,
}

#[derive(Default)]
pub(super) struct History {
    current: Option<Snapshot>,
    gesture_start: Option<Snapshot>,
    undo: VecDeque<Snapshot>,
    redo: Vec<Snapshot>,
}

impl History {
    fn observe(&mut self, next: Snapshot, continuous: bool) {
        if self.current.as_ref() == Some(&next) {
            if !continuous { self.finish_gesture(); }
            return;
        }
        let previous = self.current.replace(next);
        if let Some(previous) = previous {
            self.gesture_start.get_or_insert(previous);
        }
        if !continuous { self.finish_gesture(); }
    }

    fn finish_gesture(&mut self) {
        if let Some(start) = self.gesture_start.take() {
            if self.current.as_ref() != Some(&start) {
                self.push_undo(start);
                self.redo.clear();
            }
        }
    }

    fn push_undo(&mut self, snapshot: Snapshot) {
        if self.undo.len() == HISTORY_LIMIT { self.undo.pop_front(); }
        self.undo.push_back(snapshot);
    }

    pub(super) fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.has_gesture_change()
    }

    pub(super) fn can_redo(&self) -> bool {
        !self.redo.is_empty() && !self.has_gesture_change()
    }

    fn has_gesture_change(&self) -> bool {
        self.gesture_start.is_some() && self.gesture_start != self.current
    }

    fn step(&mut self, action: Action) -> Option<Snapshot> {
        self.finish_gesture();
        let target = match action {
            Action::Undo => self.undo.pop_back(),
            Action::Redo => self.redo.pop(),
        }?;
        if let Some(previous) = self.current.replace(target.clone()) {
            match action {
                Action::Undo => self.redo.push(previous),
                Action::Redo => self.push_undo(previous),
            }
        }
        Some(target)
    }
}

impl App {
    pub(super) fn observe_history(&mut self, continuous: bool) {
        let snapshot = Snapshot::capture(self);
        self.history.observe(snapshot, continuous);
    }

    pub(super) fn apply_history_actions(&mut self) -> bool {
        let mut target = None;
        for action in self.history_actions.drain(..) {
            if let Some(snapshot) = self.history.step(action) {
                target = Some(snapshot);
            }
        }
        let Some(snapshot) = target else { return false; };
        snapshot.restore(self);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(app: &mut App, action: Action) -> bool {
        app.history_actions.push(action);
        app.apply_history_actions()
    }

    #[test]
    fn history_groups_a_drag_and_preserves_the_redo_branch_until_an_edit() {
        let mut app = App::default();
        let original_scale = app.mandelbox_parameters.scale;
        app.observe_history(false);
        for scale in [-1.6, -1.7, -1.8] {
            app.mandelbox_parameters.scale = scale;
            app.observe_history(true);
        }
        app.observe_history(false);
        assert!(step(&mut app, Action::Undo));
        assert_eq!(app.mandelbox_parameters.scale, original_scale);
        assert!(!step(&mut app, Action::Undo));

        app.mandelbox_parameters.scale = -1.65;
        app.observe_history(true);
        app.mandelbox_parameters.scale = original_scale;
        app.observe_history(false);
        assert!(step(&mut app, Action::Redo));
        assert_eq!(app.mandelbox_parameters.scale, -1.8);

        assert!(step(&mut app, Action::Undo));
        app.mandelbox_parameters.scale = -1.9;
        app.observe_history(false);
        assert!(!step(&mut app, Action::Redo));
    }

    #[test]
    fn history_restores_scene_navigation_and_export_settings() {
        let mut app = App::default();
        let initial_position = app.camera.position;
        let initial_width = app.output.width;
        app.observe_history(false);

        app.camera.position = [0.5, 1.0, 2.0];
        app.camera.roll = 0.3;
        app.camera.look_roll = 0.1;
        app.observe_history(true);
        app.observe_history(false);
        app.output.width = 720;
        app.output.samples = 8;
        app.show_frame = false;
        app.observe_history(false);
        app.effects.focus = 1.5;
        app.observe_history(false);

        assert!(step(&mut app, Action::Undo));
        assert_eq!(app.effects.focus, crate::CameraEffects::default().focus);
        assert_eq!(app.output.width, 720);
        assert!(step(&mut app, Action::Undo));
        assert_eq!(app.output.width, initial_width);
        assert_eq!(app.output.samples, 16);
        assert!(app.show_frame);
        assert_eq!(app.camera.position, [0.5, 1.0, 2.0]);
        assert!(step(&mut app, Action::Undo));
        assert_eq!(app.camera.position, initial_position);
        assert!(step(&mut app, Action::Redo));
        assert_eq!(app.camera.roll, 0.3);
        assert_eq!(app.camera.look_roll, 0.1);
    }
}
