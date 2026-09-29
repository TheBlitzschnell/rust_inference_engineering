//! Trains the spiral classifier and saves it to models/spiral-mlp.safetensors.
//!
//! Run with: cargo run --release -p ch10-first-model --bin train

use ch07_threads::SpinPool;
use ch10_first_model::{Mlp, accuracy, model_path, spirals, train::train};

fn main() {
    let (train_x, train_y) = spirals(300, 3, 0.2, 7);
    let (test_x, test_y) = spirals(100, 3, 0.2, 99);
    let mut model = Mlp::random(&[2, 64, 64, 3], 42);
    println!(
        "training a {} -> 64 -> 64 -> {} MLP ({} parameters) on {} points",
        model.input_dim(),
        model.output_dim(),
        model.params(),
        train_y.len()
    );

    let start = std::time::Instant::now();
    train(&mut model, &train_x, &train_y, 2000, 0.2, |step, loss| {
        if step % 250 == 0 || step == 1999 {
            println!("   step {step:>4}  loss {loss:.4}");
        }
    });
    println!("trained in {:.1?}", start.elapsed());

    let mut pool = SpinPool::new(1);
    println!(
        "accuracy: training set {:.1}%, held-out test set {:.1}%",
        100.0 * accuracy(&model, &mut pool, &train_x, &train_y),
        100.0 * accuracy(&model, &mut pool, &test_x, &test_y)
    );

    let path = model_path();
    std::fs::create_dir_all(path.parent().expect("model path has a parent"))
        .expect("create models/");
    model.save(&path).expect("write model file");
    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!("saved {} ({size} bytes)", path.display());
}
