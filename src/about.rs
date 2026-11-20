use std::thread;

use semver::Version;
use winit::event_loop::EventLoopProxy;

use crate::{APP_NAME, startup::Event};

const CURRENT_VERSION: &str = env!("CARGO_PKG_VERSION");
const RELEASES_URL: &str = "https://github.com/urenasalvador/MengerVibes/releases";
const PROJECT_LICENSE: &str = include_str!("../COPYING");

#[derive(Default)]
pub(crate) struct About {
    pub(crate) open: bool,
    status: UpdateStatus,
}

#[derive(Default)]
enum UpdateStatus {
    #[default]
    Idle,
    Checking,
    Finished(Result<BuildStatus, CheckError>),
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum BuildStatus {
    Newer(Version),
    UpToDate,
    NoBuilds,
    NoCompatibleBuilds,
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CheckError {
    Unavailable,
    Access,
    Connection,
}

impl About {
    pub(crate) fn show(&mut self, ctx: &egui::Context, proxy: Option<&EventLoopProxy<Event>>) {
        if !self.open {
            return;
        }

        let dialog = egui::Modal::new(egui::Id::new("about_mengervibes")).show(ctx, |ui| {
            ui.set_width((ctx.screen_rect().width() - 40.0).clamp(200.0, 480.0));
            ui.heading(format!("About {APP_NAME}"));
            ui.label(format!("Version {CURRENT_VERSION}"));
            ui.label("By Salvador Ureña, with help from OpenAI through vibe coding.");
            ui.separator();
            ui.label("Licenses");
            ui.collapsing("MengerVibes · GNU GPL v3.0 only", |ui| {
                egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                    ui.label(PROJECT_LICENSE);
                });
            });
            ui.collapsing("Third-party credits and licenses", |ui| {
                ui.hyperlink_to("Mandelbulber v2", "https://github.com/buddhi1980/mandelbulber2");
                ui.label("Pseudo-Kleinian iteration adapted from Mandelbulber v2. Copyright © 2017 Mandelbulber Team (GPLv3). Its Amazing Surface implementation was also used as a reference.");
                ui.separator();
                ui.label("Direct dependencies");
                ui.label("MIT or Apache-2.0: egui, egui-wgpu, egui-winit, image, pollster, semver, wgpu, windows-sys");
                ui.label("Apache-2.0: winit · MIT: embed-resource");
            });
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                let can_check = cfg!(target_os = "windows") && proxy.is_some()
                    && !matches!(self.status, UpdateStatus::Checking);
                if ui.add_enabled(can_check, egui::Button::new("Check for updates")).clicked()
                    && let Some(proxy) = proxy
                {
                    self.check(proxy.clone());
                }
                ui.hyperlink_to("GitHub releases", RELEASES_URL);
            });
            match &self.status {
                UpdateStatus::Idle => { ui.small("Checks published builds on GitHub. No automatic downloads."); }
                UpdateStatus::Checking => { ui.label("Checking for updates…"); }
                UpdateStatus::Finished(Ok(BuildStatus::Newer(version))) => {
                    ui.label(format!("New build available: {version}"));
                }
                UpdateStatus::Finished(Ok(BuildStatus::UpToDate)) => { ui.label("You're up to date."); }
                UpdateStatus::Finished(Ok(BuildStatus::NoBuilds)) => { ui.label("No builds have been published yet."); }
                UpdateStatus::Finished(Ok(BuildStatus::NoCompatibleBuilds)) => {
                    ui.label("No compatible builds found with a version tag.");
                }
                UpdateStatus::Finished(Err(CheckError::Access)) => {
                    ui.label("GitHub did not grant access. For a private repository, set MENGERVIBES_GITHUB_TOKEN with read access before starting MengerVibes, or open GitHub releases to check manually.");
                }
                UpdateStatus::Finished(Err(CheckError::Connection)) => {
                    ui.label("Could not check for updates. Check your connection and try again.");
                }
                UpdateStatus::Finished(Err(CheckError::Unavailable)) => {
                    ui.label("The update checker could not start on this system.");
                }
            };
            ui.button("Close").clicked()
        });
        if dialog.inner || dialog.should_close() {
            self.open = false;
        }
    }

    fn check(&mut self, proxy: EventLoopProxy<Event>) {
        self.status = UpdateStatus::Checking;
        let worker = thread::Builder::new()
            .name("check-for-updates".into())
            .spawn(move || {
                let _ = proxy.send_event(Event::UpdateChecked(check_releases()));
            });
        if worker.is_err() {
            self.finish(Err(CheckError::Unavailable));
        }
    }

    pub(crate) fn finish(&mut self, result: Result<BuildStatus, CheckError>) {
        self.status = UpdateStatus::Finished(result);
    }
}

fn compare_builds(current: &Version, tags: &str) -> BuildStatus {
    if tags.trim().is_empty() {
        return BuildStatus::NoBuilds;
    }
    let newest = tags
        .lines()
        .filter_map(|tag| {
            let tag = tag.trim();
            Version::parse(tag.strip_prefix('v').unwrap_or(tag)).ok()
        })
        .max_by(Version::cmp_precedence);
    match newest {
        Some(version) if version.cmp_precedence(current).is_gt() => BuildStatus::Newer(version),
        Some(_) => BuildStatus::UpToDate,
        None => BuildStatus::NoCompatibleBuilds,
    }
}

#[cfg(target_os = "windows")]
const QUERY_RELEASES: &str = r#"
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$headers = @{ Accept = 'application/vnd.github+json' }
if ($env:MENGERVIBES_GITHUB_TOKEN) {
    $headers.Authorization = 'Bearer ' + $env:MENGERVIBES_GITHUB_TOKEN
}
try {
    $uri = 'https://api.github.com/repos/urenasalvador/MengerVibes/releases?per_page=100'
    $releases = Invoke-RestMethod -Uri $uri -Headers $headers -UserAgent 'MengerVibes' -TimeoutSec 10 -MaximumRedirection 0 -UseBasicParsing
    foreach ($release in $releases) {
        if ($release.draft -or -not $release.tag_name) { continue }
        foreach ($asset in $release.assets) {
            if ($asset.state -eq 'uploaded' -and $asset.name -match '\.(zip|exe|msi|msix|msixbundle|7z)$') {
                [Console]::WriteLine($release.tag_name)
                break
            }
        }
    }
} catch {
    $status = [int]$_.Exception.Response.StatusCode
    if ($status -in 401, 404 -or ($status -eq 403 -and $_.Exception.Response.Headers['X-RateLimit-Remaining'] -ne '0')) {
        exit 2
    }
    exit 1
}
"#;

#[cfg(target_os = "windows")]
fn check_releases() -> Result<BuildStatus, CheckError> {
    use std::{os::windows::process::CommandExt, path::PathBuf, process::Command};
    use windows_sys::Win32::System::Threading::CREATE_NO_WINDOW;

    let system_root = std::env::var_os("SystemRoot").ok_or(CheckError::Unavailable)?;
    let powershell =
        PathBuf::from(system_root).join("System32/WindowsPowerShell/v1.0/powershell.exe");
    // Read GitHub release metadata only. The program never requests a build asset.
    let output = Command::new(powershell)
        .args([
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            QUERY_RELEASES,
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|_| CheckError::Unavailable)?;
    match output.status.code() {
        Some(0) => {
            let tags = String::from_utf8_lossy(&output.stdout);
            let current =
                Version::parse(CURRENT_VERSION).expect("Cargo package version is valid SemVer");
            Ok(compare_builds(&current, &tags))
        }
        Some(2) => Err(CheckError::Access),
        _ => Err(CheckError::Connection),
    }
}

#[cfg(not(target_os = "windows"))]
fn check_releases() -> Result<BuildStatus, CheckError> {
    Err(CheckError::Unavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_the_highest_build_by_version_precedence() {
        let current = Version::new(1, 9, 0);
        let tags = "v1.9.1\r\nother-tag\r\n1.10.0-rc.2\r\nv1.10.0-rc.10\r\n";
        assert_eq!(
            compare_builds(&current, tags),
            BuildStatus::Newer(Version::parse("1.10.0-rc.10").unwrap())
        );
        assert_eq!(
            compare_builds(&current, "v1.10.0-rc.10\nv1.10.0"),
            BuildStatus::Newer(Version::new(1, 10, 0))
        );
    }

    #[test]
    fn distinguishes_no_builds_from_current_and_unversioned_builds() {
        let current = Version::new(1, 2, 0);
        assert_eq!(compare_builds(&current, "\n"), BuildStatus::NoBuilds);
        assert_eq!(
            compare_builds(&current, "nightly"),
            BuildStatus::NoCompatibleBuilds
        );
        assert_eq!(
            compare_builds(&current, "v1.2.0-rc.1\nv1.2.0+ci.5\nv1.1.9"),
            BuildStatus::UpToDate
        );
    }
}
