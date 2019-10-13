extern crate rand;
#[macro_use]
extern crate serde_derive;
extern crate serde;
#[macro_use]
#[cfg(test)]
extern crate approx;

use rand::distributions::Distribution;
use rand::distributions::StandardNormal;
use rand::distributions::Uniform;
use rand::{thread_rng, Rng, SeedableRng, StdRng, ThreadRng};

#[derive(Serialize, Deserialize, Clone)]
pub struct UpperLower {
    pub lower: f64,
    pub upper: f64,
}

fn get_levy(alpha: f64, uniform_rand: f64) -> f64 {
    uniform_rand.powf(-1.0 / alpha)
}

fn get_levy_flight(
    curr_val: f64,
    step_size: f64,
    lambda: f64,
    uniform_rand: f64,
    norm_rand: f64,
) -> f64 {
    curr_val + step_size * get_levy(lambda, uniform_rand) * norm_rand
}

fn get_truncated_parameter(lower: f64, upper: f64, result: f64) -> f64 {
    if result > upper {
        upper
    } else if result < lower {
        lower
    } else {
        result
    }
}

fn get_random_parameter(lower: f64, upper: f64, rand: f64) -> f64 {
    let half = 0.5;
    //reflects middle more likely than edges
    get_truncated_parameter(
        lower,
        upper,
        (upper + lower) * half + (upper - lower) * half * rand,
    )
}

fn get_random_parameters<T, U>(ul: &[UpperLower], rng: &mut T, rand: &mut U) -> Vec<f64>
where
    T: Rng,
    U: Distribution<f64>,
{
    ul.iter()
        .map(|v| get_random_parameter(v.lower, v.upper, rand.sample(rng)))
        .collect()
}

fn get_new_parameter_and_fn<T, U, S>(
    ul: &[UpperLower],
    obj_fn: S,
    rng: &mut T,
    rand: &mut U,
) -> (Vec<f64>, f64)
where
    S: Fn(&[f64]) -> f64,
    T: Rng,
    U: Distribution<f64>,
{
    let parameters = get_random_parameters(ul, rng, rand);
    let fn_value_at_parameters = obj_fn(&parameters);
    (parameters, fn_value_at_parameters)
}

const STEP_INCREMENT: f64 = 0.01;
fn get_step_size(curr: f64, best: f64, lower: f64, upper: f64) -> f64 {
    STEP_INCREMENT * (upper - lower) * (curr - best)
}

fn get_new_nest<T, U, S>(
    ul: &[UpperLower],
    obj_fn: S,
    n: usize,
    rng: &mut T,
    rand: &mut U,
) -> Vec<(Vec<f64>, f64)>
where
    S: Fn(&[f64]) -> f64,
    T: Rng,
    U: Distribution<f64>,
{
    (0..n)
        .map(|_| get_new_parameter_and_fn(ul, &obj_fn, rng, rand))
        .collect()
}

fn sort_nest(nest: &mut Vec<(Vec<f64>, f64)>, //move nest
) {
    nest.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()); //smallest to largest
}

fn get_best_nest(
    new_nest: &[(Vec<f64>, f64)],
    curr_nest: &mut Vec<(Vec<f64>, f64)>, //move curr_nest
) {
    curr_nest
        .iter_mut()
        .zip(new_nest.iter())
        .for_each(|(curr_val, new_val)| {
            let (curr_params, curr_fn_val) = curr_val;
            let (new_params, new_fn_val) = new_val;
            if new_fn_val < curr_fn_val {
                *curr_params = new_params.to_vec();
                *curr_fn_val = *new_fn_val;
            }
        });
    sort_nest(curr_nest);
}

fn get_cuckoos<T, G, U>(
    new_nest: &mut Vec<(Vec<f64>, f64)>,
    curr_nest: &[(Vec<f64>, f64)],
    best_parameters: &[f64],
    ul: &[UpperLower],
    obj_fn: impl Fn(&[f64]) -> f64,
    lambda: f64,
    rng: &mut T,
    uniform: &mut U,
    normal: &mut G,
) where
    T: Rng,
    G: Distribution<f64>,
    U: Distribution<f64>,
{
    new_nest
        .iter_mut()
        .zip(curr_nest.iter())
        .for_each(|(new_val, curr_val)| {
            let (new_parameters, new_fn_val) = new_val;
            let (curr_parameters, _) = curr_val;
            *new_parameters = curr_parameters
                .iter()
                .zip(ul.iter())
                .zip(best_parameters.iter())
                .map(|((curr_param, v), bp)| {
                    get_truncated_parameter(
                        v.lower,
                        v.upper,
                        get_levy_flight(
                            *curr_param,
                            get_step_size(*curr_param, *bp, v.lower, v.upper),
                            lambda,
                            uniform.sample(rng),
                            normal.sample(rng),
                        ),
                    )
                })
                .collect();

            *new_fn_val = obj_fn(&new_parameters);
        });
}

fn get_pa(p_min: f64, p_max: f64, index: usize, n: usize) -> f64 {
    p_max - (p_max - p_min) * (index as f64) / (n as f64)
}

fn empty_nests<T, U>(
    new_nest: &mut Vec<(Vec<f64>, f64)>, //
    obj_fn: &impl Fn(&[f64]) -> f64,
    ul: &[UpperLower],
    p: f64,
    rng: &mut T,
    rand: &mut U,
) where
    T: Rng,
    U: Distribution<f64>,
{
    let n = new_nest.len();
    let num_to_keep = ((n as f64) * p) as usize;
    let start_num = n - num_to_keep;
    new_nest
        .iter_mut()
        .enumerate()
        .filter(|(index, _)| index >= &start_num)
        .for_each(|(_, new_val)| {
            *new_val = get_new_parameter_and_fn(ul, &obj_fn, rng, rand);
        });
}

pub fn get_rng_seed(seed: [u8; 32]) -> StdRng {
    SeedableRng::from_seed(seed)
}

pub fn get_rng_system_seed() -> ThreadRng {
    thread_rng()
}

pub fn optimize<T>(
    obj_fn: &impl Fn(&[f64]) -> f64,
    ul: &[UpperLower],
    n: usize,
    total_mc: usize,
    tol: f64,
    rng_inst: impl Fn() -> T,
) -> (Vec<f64>, f64)
where
    T: Rng,
{
    let lambda = 1.5; //controls size of levy moves
    let p_min = 0.05; //min percentage of nests to replace
    let p_max = 0.5; //max percentage of nests to replace

    //randomness
    let mut rng = rng_inst();
    let mut normal = StandardNormal;
    let mut uniform = Uniform::new(0.0f64, 1.0);

    //starting nests
    let mut curr_nest = get_new_nest(&ul, &obj_fn, n, &mut rng, &mut normal);
    sort_nest(&mut curr_nest);
    let mut new_nest = get_new_nest(&ul, &obj_fn, n, &mut rng, &mut normal);

    let mut index = 0;
    loop {
        get_cuckoos(
            &mut new_nest,
            &curr_nest,
            &curr_nest.first().unwrap().0, // best parameters
            &ul,
            &obj_fn,
            lambda,
            &mut rng,
            &mut uniform,
            &mut normal,
        );

        get_best_nest(&new_nest, &mut curr_nest);

        empty_nests(
            &mut curr_nest,
            &obj_fn,
            &ul,
            get_pa(p_min, p_max, index, total_mc),
            &mut rng,
            &mut normal,
        );

        sort_nest(&mut curr_nest);
        index = index + 1;

        if cfg!(feature = "VERBOSE_FLAG_ALL") {
            print!("Index: {}, Param Vals: ", index);
            for val in curr_nest[0].0.iter() {
                print!("{}, ", val);
            }
            println!("Objective Value: {}", curr_nest[0].1);
        }
        if index >= total_mc || curr_nest.first().unwrap().1 <= tol {
            break;
        }
    }
    if cfg!(feature = "VERBOSE_FLAG_SUMMARY") {
        print!("Index: {}, Param Vals: ", index);
        for val in curr_nest[0].0.iter() {
            print!("{}, ", val);
        }
        println!("Objective Value: {}", curr_nest[0].1);
    }
    let (optim_parameters, optim_fn_val) = curr_nest.first().unwrap();
    (optim_parameters.to_vec(), *optim_fn_val)
}

#[cfg(test)]
mod tests {
    #[cfg(test)]
    use std::f64::consts::PI;

    #[cfg(test)]
    struct DegenerateDistribution {
        value: f64,
    }
    #[cfg(test)]
    impl Distribution<f64> for DegenerateDistribution {
        fn sample<R: Rng + ?Sized>(&self, _rng: &mut R) -> f64 {
            self.value
        }
    }
    use super::*;
    #[test]
    fn sort_algorithm() {
        let mut vec_to_sort: Vec<(Vec<f64>, f64)> = vec![];
        vec_to_sort.push((vec![1.0], 3.0));
        vec_to_sort.push((vec![1.0], 2.0));
        vec_to_sort.push((vec![1.0], 5.0));
        sort_nest(&mut vec_to_sort);
        let expected: Vec<f64> = vec![2.0, 3.0, 5.0];
        for (val, expected) in vec_to_sort.iter().zip(expected.iter()) {
            let (_, v) = *val;
            assert_eq!(v, *expected);
        }
    }
    #[test]
    fn simple_fn_optim() {
        let seed: [u8; 32] = [2; 32];
        let mut ul = vec![];
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        let (result, fn_val) = optimize(
            &|inputs: &[f64]| {
                inputs[0].powi(2) + inputs[1].powi(2) + inputs[2].powi(2) + inputs[3].powi(2)
            },
            &ul,
            25,
            1000,
            0.00000001,
            || get_rng_seed(seed),
        );
        for res in result.iter() {
            assert_abs_diff_eq!(*res, 0.0, epsilon = 0.001);
        }
        assert_abs_diff_eq!(fn_val, 0.0, epsilon = 0.00001);
    }
    #[test]
    fn test_rosenbrok_function() {
        let seed: [u8; 32] = [2; 32];
        let mut ul = vec![];
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        let (result, fn_val) = optimize(
            &|inputs: &[f64]| {
                (1.0 - inputs[0]).powi(2) + 100.0 * (inputs[1] - inputs[0].powi(2)).powi(2)
            },
            &ul,
            20,
            10000,
            0.00000001,
            || get_rng_seed(seed),
        );
        for res in result.iter() {
            assert_abs_diff_eq!(*res, 1.0, epsilon = 0.001);
        }
        assert_abs_diff_eq!(fn_val, 0.0, epsilon = 0.00001);
    }
    #[test]
    fn test_u_2_function() {
        //16 parameters
        let seed: [u8; 32] = [2; 32];
        let mut ul = vec![];
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        ul.push(UpperLower {
            lower: -5.0,
            upper: 5.0,
        });
        let (result, fn_val) = optimize(
            &|inputs: &[f64]| {
                inputs
                    .iter()
                    .fold(0.0, |accum, curr| accum + (curr - 1.0).powi(2))
            },
            &ul,
            25,
            25000,
            0.00000001,
            || get_rng_seed(seed),
        );
        for res in result.iter() {
            assert_abs_diff_eq!(*res, 1.0, epsilon = 0.001);
        }
        assert_abs_diff_eq!(fn_val, 0.0, epsilon = 0.00001);
    }
    #[test]
    fn test_rastigrin_function() {
        let seed: [u8; 32] = [2; 32];
        let mut ul = vec![];
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        ul.push(UpperLower {
            lower: -4.0,
            upper: 4.0,
        });
        let rastigrin_scale = 10.0;
        let (result, fn_val) = optimize(
            &|inputs: &[f64]| {
                rastigrin_scale * (inputs.len() as f64)
                    + inputs.iter().fold(0.0, |accum, curr| {
                        accum + curr.powi(2) - rastigrin_scale * (2.0 * PI * curr).cos()
                    })
            },
            &ul,
            25,
            25000,
            0.00000001,
            || get_rng_seed(seed),
        );
        for res in result.iter() {
            assert_abs_diff_eq!(*res, 0.0, epsilon = 0.001);
        }
        assert_abs_diff_eq!(fn_val, 0.0, epsilon = 0.00001);
    }
    #[test]
    fn test_get_levy() {
        let alpha = 1.5;
        assert_abs_diff_eq!(get_levy(alpha, 0.5), 1.5874, epsilon = 0.0001);
    }
    #[test]
    fn test_get_levy_flight() {
        let alpha = 1.5;
        let step_size = 0.01;
        let curr_val = 0.5;
        let unif_rand = 0.2;
        let norm_rand = -0.5;
        assert_abs_diff_eq!(
            get_levy_flight(curr_val, step_size, alpha, unif_rand, norm_rand),
            0.48538,
            epsilon = 0.0001
        );
    }
    #[test]
    fn test_get_random_parameter() {
        let lower = -0.5;
        let upper = 0.5;
        let rand = 0.3;
        assert_abs_diff_eq!(
            get_random_parameter(lower, upper, rand),
            0.15,
            epsilon = 0.0001
        );
    }
    #[test]
    fn test_get_best_nest() {
        let v1 = vec![0.5, 0.6, 0.7];
        let v2 = vec![0.4, 0.5, 0.6];
        let mut curr_nest = vec![(v1.clone(), 0.8), (v2.clone(), 0.7)];
        let new_nest = vec![(v1, 0.9), (v2, 0.6)];
        get_best_nest(&new_nest, &mut curr_nest);

        assert_eq!(curr_nest[0].1, 0.6);
        assert_eq!(curr_nest[1].1, 0.8);
        assert_eq!(curr_nest[0].0[0], 0.4);
        assert_eq!(curr_nest[0].0[1], 0.5);
        assert_eq!(curr_nest[0].0[2], 0.6);
        assert_eq!(curr_nest[1].0[0], 0.5);
        assert_eq!(curr_nest[1].0[1], 0.6);
        assert_eq!(curr_nest[1].0[2], 0.7);
    }
    #[test]
    fn test_get_cuckoos() {
        let v1 = vec![0.5, 0.6, 0.7];
        let v2 = vec![0.4, 0.5, 0.6];
        let curr_nest = vec![(v1.clone(), 0.8), (v2.clone(), 0.7)];
        let mut new_nest = vec![(v1, 0.9), (v2, 0.6)];
        let best_parameters = vec![0.4, 0.7, 0.5];
        let ul = vec![
            UpperLower {
                lower: -4.0,
                upper: 4.0,
            },
            UpperLower {
                lower: -4.0,
                upper: 4.0,
            },
            UpperLower {
                lower: -4.0,
                upper: 4.0,
            },
        ];

        let mut normal = DegenerateDistribution { value: 0.5 };
        let mut uniform = DegenerateDistribution { value: 0.5 };
        let mut rng = get_rng_system_seed();
        get_cuckoos(
            &mut new_nest,
            &curr_nest,
            &best_parameters,
            &ul,
            |v| v[0] + v[1] + v[2],
            1.5,
            &mut rng,
            &mut uniform,
            &mut normal,
        );

        assert_abs_diff_eq!(new_nest[0].1, 1.8127, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[1].1, 1.49365, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[0].0[0], 0.50635, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[0].0[1], 0.59365, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[0].0[2], 0.712699, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[1].0[0], 0.4, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[1].0[1], 0.487301, epsilon = 0.0001);
        assert_abs_diff_eq!(new_nest[1].0[2], 0.60635, epsilon = 0.0001);
    }

    #[test]
    fn test_get_pa() {
        let p_min = 0.05;
        let p_max = 0.5;
        let index: usize = 50;
        let n: usize = 1000;
        assert_abs_diff_eq!(get_pa(p_min, p_max, index, n), 0.4775, epsilon = 0.0001);
    }
    #[test]
    fn test_empty_nests() {
        let v1 = vec![0.5, 0.6, 0.7];
        let v2 = vec![0.4, 0.5, 0.6];
        let mut new_nest = vec![(v1.clone(), 0.8), (v2.clone(), 0.7)];
        let ul = vec![
            UpperLower {
                lower: -4.0,
                upper: 4.0,
            },
            UpperLower {
                lower: -4.0,
                upper: 4.0,
            },
            UpperLower {
                lower: -4.0,
                upper: 4.0,
            },
        ];
        let mut normal = DegenerateDistribution { value: 0.5 };
        let mut rng = get_rng_system_seed();
        let obj_fn = |v: &[f64]| v[0] + v[1] + v[2];
        empty_nests(&mut new_nest, &obj_fn, &ul, 0.4, &mut rng, &mut normal);

        assert_eq!(new_nest[0].1, 0.8);
        assert_eq!(new_nest[1].1, 0.7);
        assert_eq!(new_nest[0].0[0], 0.5);
        assert_eq!(new_nest[0].0[1], 0.6);
        assert_eq!(new_nest[0].0[2], 0.7);
        assert_eq!(new_nest[1].0[0], 0.4);
        assert_eq!(new_nest[1].0[1], 0.5);
        assert_eq!(new_nest[1].0[2], 0.6);
    }
}
