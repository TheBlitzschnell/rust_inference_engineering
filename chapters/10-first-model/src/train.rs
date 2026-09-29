//! Just enough training to produce real weights for the inference chapters.
//!
//! Full-batch gradient descent with momentum on the softmax cross-entropy
//! loss, with hand-written backpropagation. Nothing here is optimized: it runs
//! once, for a few seconds, and inference is the subject of this course.

use crate::Mlp;

/// Trains `model` on `points` (`[n × input_dim]`) and `labels` for `steps`
/// steps with learning rate `lr`. Calls `on_step(step, loss)` after each step.
pub fn train(
    model: &mut Mlp,
    points: &[f32],
    labels: &[usize],
    steps: usize,
    lr: f32,
    mut on_step: impl FnMut(usize, f32),
) {
    const MOMENTUM: f32 = 0.9;
    let n = labels.len();
    assert_eq!(points.len(), n * model.input_dim());
    // One velocity buffer per weight and bias, for momentum.
    let mut velocity: Vec<(Vec<f32>, Vec<f32>)> = model
        .layers
        .iter()
        .map(|l| (vec![0.0; l.weight.len()], vec![0.0; l.bias.len()]))
        .collect();

    for step in 0..steps {
        // Forward, keeping every layer's input (`inputs[l]`) for the backward pass.
        let mut inputs: Vec<Vec<f32>> = vec![points.to_vec()];
        let last = model.layers.len() - 1;
        for (l, layer) in model.layers.iter().enumerate() {
            let x = &inputs[l];
            let mut z = vec![0.0f32; n * layer.out_dim];
            for b in 0..n {
                let xb = &x[b * layer.in_dim..(b + 1) * layer.in_dim];
                for o in 0..layer.out_dim {
                    let w = &layer.weight[o * layer.in_dim..(o + 1) * layer.in_dim];
                    let dot: f32 = w.iter().zip(xb).map(|(a, c)| a * c).sum();
                    let v = dot + layer.bias[o];
                    z[b * layer.out_dim + o] = if l == last { v } else { v.max(0.0) };
                }
            }
            inputs.push(z);
        }

        // Loss and its gradient with respect to the logits:
        // d(cross-entropy)/d(logit) = softmax(logits) - one_hot(label).
        let classes = model.output_dim();
        let mut grad = inputs[last + 1].clone();
        let mut loss = 0.0f32;
        for (row, &label) in grad.chunks_exact_mut(classes).zip(labels) {
            crate::probabilities(row);
            loss -= row[label].max(1e-12).ln();
            row[label] -= 1.0;
            for g in row.iter_mut() {
                *g /= n as f32;
            }
        }
        loss /= n as f32;

        // Backward, from the last layer to the first.
        for l in (0..=last).rev() {
            let layer = &model.layers[l];
            let x = &inputs[l];
            let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
            let mut grad_w = vec![0.0f32; out_dim * in_dim];
            let mut grad_b = vec![0.0f32; out_dim];
            let mut grad_x = vec![0.0f32; n * in_dim];
            for b in 0..n {
                let xb = &x[b * in_dim..(b + 1) * in_dim];
                for o in 0..out_dim {
                    let g = grad[b * out_dim + o];
                    if g == 0.0 {
                        continue;
                    }
                    grad_b[o] += g;
                    for i in 0..in_dim {
                        grad_w[o * in_dim + i] += g * xb[i];
                        grad_x[b * in_dim + i] += g * layer.weight[o * in_dim + i];
                    }
                }
            }
            // Through the ReLU of the previous layer: zero where it was off.
            if l > 0 {
                for (gx, &xv) in grad_x.iter_mut().zip(x) {
                    if xv <= 0.0 {
                        *gx = 0.0;
                    }
                }
            }
            // Momentum update.
            let (vw, vb) = &mut velocity[l];
            let layer = &mut model.layers[l];
            for ((w, v), g) in layer.weight.iter_mut().zip(vw.iter_mut()).zip(&grad_w) {
                *v = MOMENTUM * *v + g;
                *w -= lr * *v;
            }
            for ((w, v), g) in layer.bias.iter_mut().zip(vb.iter_mut()).zip(&grad_b) {
                *v = MOMENTUM * *v + g;
                *w -= lr * *v;
            }
            grad = grad_x;
        }
        on_step(step, loss);
    }
}
