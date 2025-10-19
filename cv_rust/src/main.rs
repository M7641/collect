use headless_chrome::{Browser, LaunchOptions};

use headless_chrome::types::PrintToPdfOptions;

fn print_cv_html() {
    // Launch headless Chrome
    let browser = Browser::new(LaunchOptions::default()).expect("Failed to launch browser");
    // Open cv.html
    let tab = browser.new_tab().expect("Failed to open new tab");
    tab.navigate_to("file:///home/mike/github/collect/cv_rust/src/cv.html")
        .expect("Failed to navigate to cv.html");
    tab.wait_until_navigated()
        .expect("Failed to wait for navigation");

    let pdf_data = tab
        .print_to_pdf(Some(PrintToPdfOptions {
            margin_top: Some(0.0),
            ..Default::default()
        }))
        .expect("Failed to generate PDF");
    std::fs::write("cv.pdf", &pdf_data).expect("Failed to write PDF file");
}

fn main() {
    print_cv_html();
}
