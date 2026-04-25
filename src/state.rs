use std::{
    collections::{HashSet, VecDeque},
    fs,
    sync::Arc,
};

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex;
use wasmtime::{Engine, Module};
use wreq::Client;
use wreq_util::Emulation;

use crate::models::{Account, Config};

#[derive(Clone)]
pub struct AppState {
    pub safari_client: Client,
    pub chrome_client: Client,
    pub engine: Engine,
    pub module: Module,
    pub keys: Arc<HashSet<String>>,
    pub accounts: Arc<Mutex<VecDeque<Account>>>,
}

impl AppState {
    pub fn from_files(config_path: &str, wasm_path: &str) -> Result<Self> {
        let cfg: Config = fs::read_to_string(config_path)
            .with_context(|| format!("failed reading {config_path}"))
            .and_then(|raw| serde_json::from_str(&raw).context("invalid config.json"))?;

        let safari_client = Client::builder()
            .emulation(Emulation::Safari26)
            .build()
            .context("failed to build Safari-emulated client")?;

        let chrome_client = Client::builder()
            .emulation(Emulation::Chrome136)
            .build()
            .context("failed to build Chrome-emulated client")?;

        let engine = Engine::default();
        let module = Module::from_file(&engine, wasm_path)
            .map_err(|e| anyhow!("failed loading wasm module {wasm_path}: {e}"))?;

        Ok(Self {
            safari_client,
            chrome_client,
            engine,
            module,
            keys: Arc::new(cfg.keys.into_iter().collect()),
            accounts: Arc::new(Mutex::new(cfg.accounts.into())),
        })
    }

    pub async fn checkout_account(&self) -> Option<Account> {
        self.accounts.lock().await.pop_front()
    }

    pub async fn release_account(&self, account: Account) {
        self.accounts.lock().await.push_back(account);
    }
}
