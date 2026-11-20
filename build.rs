fn main() {
    println!("cargo:rerun-if-changed=assets/mengervibes.rc");
    println!("cargo:rerun-if-changed=assets/mengervibes.ico");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("assets/mengervibes.rc", embed_resource::NONE)
            .manifest_required()
            .expect("Embed the Windows application icon");
    }
}
