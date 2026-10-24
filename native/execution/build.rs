use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=control.cu");
    println!("cargo:rerun-if-env-changed=COZY_CUDA_TOOLKIT");
    let root = PathBuf::from(
        env::var_os("COZY_CUDA_TOOLKIT")
            .expect("COZY_CUDA_TOOLKIT must name the build-time CUDA 13 toolkit"),
    );
    let mut build = cc::Build::new();
    build
        .cuda(true)
        .cpp_link_stdlib(None)
        .cudart("static")
        .file("control.cu")
        .include(root.join("include"))
        .flag("-std=c++17")
        .flag("-O3");
    for arch in [80, 86, 89, 90, 100, 103, 120, 121] {
        build.flag(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    // Only AOT cubins: serving never compiles PTX for an unqualified architecture.
    if !cfg!(windows) {
        build.flag("-Xcompiler=-fPIC,-fvisibility=hidden,-fno-gnu-unique");
    }
    build.compile("cozy_execution_cuda");
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib64").display()
    );
    println!("cargo:rustc-link-lib=static=cudart_static");
    if !cfg!(windows) {
        let output = std::process::Command::new("c++")
            .arg("-print-file-name=libstdc++.a")
            .output()
            .expect("C++ toolchain");
        let archive = PathBuf::from(
            String::from_utf8(output.stdout)
                .expect("C++ archive path")
                .trim(),
        );
        println!(
            "cargo:rustc-link-search=native={}",
            archive.parent().expect("C++ archive parent").display()
        );
        println!("cargo:rustc-link-lib=static=stdc++");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=rt");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-arg=-Wl,--exclude-libs,ALL");
        println!("cargo:rustc-link-arg=-Wl,-Bsymbolic");
    }
}
