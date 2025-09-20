use reqwest;
use tokio;


#[tokio::main]
async fn fetch_and_print() -> Result<(), Box<dyn std::error::Error>> {
    let url = "https://fantasy.premierleague.com/api/bootstrap-static/";
    let resp = reqwest::get(url).await?.text().await?;
    println!("{}", resp);
    Ok(())
}

fn _run_fetch_and_print() {
    let _ = fetch_and_print();
}

fn main() {
    _run_fetch_and_print();
}
