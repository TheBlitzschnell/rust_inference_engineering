//! Capacity planning with a queueing model.
//!
//! `c` replicas, each able to complete `mu` requests per second, receive
//! Poisson arrivals at `lambda` per second, and a request waits only if
//! every replica is busy. That is the M/M/c queue, and Erlang's C formula
//! gives the probability of waiting. It is a rough model of batching
//! replicas (each is treated as one server of rate `mu`), good for the
//! question capacity planning asks: how many replicas keep the wait small
//! at the expected peak?

/// Erlang C: the probability that a request has to wait, with `c` servers
/// of rate `mu` and arrivals at `lambda` (`lambda < c · mu`).
pub fn erlang_c(c: usize, lambda: f64, mu: f64) -> f64 {
    let a = lambda / mu; // offered load, in servers
    let rho = a / c as f64;
    assert!(rho < 1.0, "the replicas cannot keep up: utilization {rho}");
    // Σ a^k / k! for k < c, computed term by term to avoid overflow.
    let mut term = 1.0;
    let mut sum = 0.0;
    for k in 0..c {
        sum += term;
        term *= a / (k + 1) as f64;
    }
    // `term` is now a^c / c!.
    let top = term / (1.0 - rho);
    top / (sum + top)
}

/// The mean time a request waits for a free replica.
pub fn mean_wait(c: usize, lambda: f64, mu: f64) -> f64 {
    erlang_c(c, lambda, mu) / (c as f64 * mu - lambda)
}

/// The smallest number of replicas whose mean wait is at most `target`
/// seconds and whose probability of waiting is at most `p_wait`.
pub fn replicas_for(lambda: f64, mu: f64, target: f64, p_wait: f64) -> usize {
    let mut c = (lambda / mu).floor() as usize + 1;
    while erlang_c(c, lambda, mu) > p_wait || mean_wait(c, lambda, mu) > target {
        c += 1;
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_server_is_the_m_m_1_queue() {
        // M/M/1: P(wait) = ρ, mean wait = ρ / (μ − λ).
        for rho in [0.3, 0.7, 0.9] {
            assert!((erlang_c(1, rho, 1.0) - rho).abs() < 1e-12);
            assert!((mean_wait(1, rho, 1.0) - rho / (1.0 - rho)).abs() < 1e-12);
        }
    }

    #[test]
    fn known_erlang_c_values() {
        // 2 servers, offered load 1 (ρ = 0.5): C = 1/3.
        assert!((erlang_c(2, 1.0, 1.0) - 1.0 / 3.0).abs() < 1e-12);
        // 10 servers, offered load 8: C ≈ 0.4092.
        assert!((erlang_c(10, 8.0, 1.0) - 0.4092).abs() < 1e-4);
    }

    #[test]
    fn more_replicas_at_the_same_utilization_wait_less() {
        let (a, b) = (erlang_c(2, 1.4, 1.0), erlang_c(20, 14.0, 1.0));
        assert!(b < a, "pooling: {b} < {a}");
        assert_eq!(replicas_for(14.0, 1.0, 0.1, 0.2), 19);
    }
}
