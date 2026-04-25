use anyhow::{Result, anyhow};
use wasmtime::{AsContextMut, Linker, Memory, Module, Store};

fn checked_write(
    store: impl AsContextMut,
    memory: &Memory,
    offset: usize,
    data: &[u8],
) -> Result<()> {
    let mut store = store;
    let size = memory.data_size(store.as_context_mut());
    let end = offset
        .checked_add(data.len())
        .ok_or_else(|| anyhow!("wasm memory offset overflow"))?;
    if end > size {
        return Err(anyhow!("wasm write out of bounds"));
    }
    memory.write(store.as_context_mut(), offset, data)?;
    Ok(())
}

fn checked_read(
    store: impl AsContextMut,
    memory: &Memory,
    offset: usize,
    out: &mut [u8],
) -> Result<()> {
    let mut store = store;
    let size = memory.data_size(store.as_context_mut());
    let end = offset
        .checked_add(out.len())
        .ok_or_else(|| anyhow!("wasm memory offset overflow"))?;
    if end > size {
        return Err(anyhow!("wasm read out of bounds"));
    }
    memory.read(store.as_context_mut(), offset, out)?;
    Ok(())
}

pub fn solve_pow(
    engine: &wasmtime::Engine,
    module: &Module,
    challenge: &str,
    salt: &str,
    difficulty: i64,
    expire_at: i64,
) -> Result<i64> {
    let mut store = Store::new(engine, ());
    let linker = Linker::new(engine);
    let instance = linker
        .instantiate(&mut store, module)
        .map_err(|e| anyhow!("failed to instantiate wasm module: {e}"))?;

    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| anyhow!("missing wasm memory export"))?;

    let add_stack = instance
        .get_typed_func::<i32, i32>(&mut store, "__wbindgen_add_to_stack_pointer")
        .map_err(|e| anyhow!("missing __wbindgen_add_to_stack_pointer: {e}"))?;
    let alloc = instance
        .get_typed_func::<(i32, i32), i32>(&mut store, "__wbindgen_export_0")
        .map_err(|e| anyhow!("missing __wbindgen_export_0: {e}"))?;
    let wasm_solve = instance
        .get_typed_func::<(i32, i32, i32, i32, i32, f64), ()>(&mut store, "wasm_solve")
        .map_err(|e| anyhow!("missing wasm_solve: {e}"))?;

    let prefix = format!("{salt}_{expire_at}_");
    let challenge_bytes = challenge.as_bytes();
    let prefix_bytes = prefix.as_bytes();

    let retptr = add_stack.call(&mut store, -16)?;

    let challenge_ptr = alloc.call(&mut store, (challenge_bytes.len() as i32, 1))?;
    checked_write(&mut store, &memory, challenge_ptr as usize, challenge_bytes)?;

    let prefix_ptr = alloc.call(&mut store, (prefix_bytes.len() as i32, 1))?;
    checked_write(&mut store, &memory, prefix_ptr as usize, prefix_bytes)?;

    wasm_solve.call(
        &mut store,
        (
            retptr,
            challenge_ptr,
            challenge_bytes.len() as i32,
            prefix_ptr,
            prefix_bytes.len() as i32,
            difficulty as f64,
        ),
    )?;

    let mut status = [0u8; 4];
    let mut value = [0u8; 8];
    checked_read(&mut store, &memory, retptr as usize, &mut status)?;
    checked_read(&mut store, &memory, (retptr + 8) as usize, &mut value)?;

    let _ = add_stack.call(&mut store, 16);

    let status = i32::from_le_bytes(status);
    if status == 0 {
        return Err(anyhow!("pow solve returned status 0"));
    }

    Ok(f64::from_le_bytes(value) as i64)
}
