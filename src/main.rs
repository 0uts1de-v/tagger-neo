#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        initial_window_size: Some(eframe::egui::vec2(1280.0, 800.0)),
        min_window_size: Some(eframe::egui::vec2(900.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "tagger-neo",
        options,
        Box::new(|cc| Box::new(tagger_neo::app::TaggerNeoApp::new(cc))),
    )
}
