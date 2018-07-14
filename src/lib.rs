extern crate rand;
extern crate byteorder;
use byteorder::{LittleEndian, WriteBytesExt};


use rand::{thread_rng, ThreadRng, SeedableRng, Rng, StdRng};
use rand::distributions::Uniform;
use rand::distributions::StandardNormal;
use rand::distributions::{Distribution};

#[macro_use]
#[cfg(test)]
extern crate approx;

pub struct Upper_lower {
    lower: f64,
    upper: f64,
}

fn get_levy(alpha:f64, uniform_rand:f64)->f64 {
    uniform_rand.powf(-1.0/alpha)
}

fn get_levy_flight(
    curr_val:f64,
    step_size:f64,
    lambda:f64,
    uniform_rand:f64,
    norm_rand:f64
)->f64 {
    curr_val+step_size*get_levy(lambda, uniform_rand)*norm_rand
}

fn get_random_parameter(
    lower:f64,
    upper:f64,
    uniform_rand:f64
)->f64{
    let half=0.5;
    (upper+lower)*half+(upper-lower)*half*uniform_rand //reflects middle more likely than edges
}

fn get_truncated_parameter(
    lower:f64,
    upper:f64,
    result:f64
)->f64 {
    if result>upper {upper} else if result<lower {lower} else {result}
}

fn get_random_parameters<T, U>(
    ul:&Vec<Upper_lower>,
    rng:&mut T,
    rand:&mut U
)->Vec<f64>
    where 
        T:Rng,
        U:Distribution<f64>
{
    ul.iter().map(|v|get_random_parameter(v.lower, v.upper, rand.sample(rng))).collect()
}

fn get_new_parameter_and_fn<T, U, S>(
    ul:&Vec<Upper_lower>,
    obj_fn:S,
    rng:&mut T,
    rand:&mut U
)->(Vec<f64>, f64) 
    where 
        S:Fn(&Vec<f64>)->f64,
        T:Rng,
        U:Distribution<f64>
{
    let parameters=get_random_parameters(ul, rng, rand);
    let fn_value_at_parameters=obj_fn(&parameters);
    (parameters, fn_value_at_parameters)
}

static step_increment:f64=0.01;
fn get_step_size(curr:f64, best:f64, lower:f64, upper:f64)->f64{
    step_increment*(upper-lower)*(curr-best)
}

fn get_new_nest<T, U, S>(
    ul:&Vec<Upper_lower>, 
    obj_fn:S,
    n:usize,
    rng:&mut T,
    rand:&mut U
)->Vec<(Vec<f64>, f64)>
    where 
        S: Fn(&Vec<f64>)->f64,
        T:Rng,
        U:Distribution<f64>
{
    (0..n).map(|_|get_new_parameter_and_fn(ul, &obj_fn, rng, rand)).collect()
}

fn sort_nest(
    nest:&mut Vec<(Vec<f64>, f64)>//move nest
){
    nest.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap());//smallest to largest
}

fn get_best_nest(
    new_nest:&Vec<(Vec<f64>, f64)>,
    curr_nest:&mut Vec<(Vec<f64>, f64)>//move curr_nest
){
    curr_nest.iter_mut().zip(new_nest.iter()).for_each(|(mut curr_val, new_val)|{
        let (curr_params, curr_fn_val)=curr_val;
        let (new_params, new_fn_val)=new_val;
        if new_fn_val< curr_fn_val { 
            curr_params=new_params.iter().map(|v|*v).collect();
            curr_fn_val=new_fn_val;
        } 
    });
    sort_nest(curr_nest);
}

fn get_cuckoos<T, G, U>(
    new_nest:&mut Vec<(Vec<f64>, f64)>, 
    curr_nest:&Vec<(Vec<f64>, f64)>,
    best_parameters:&Vec<f64>,
    ul:&Vec<Upper_lower>,
    obj_fn:impl Fn(&Vec<f64>)->f64,
    lambda:f64,
    rng:&mut T,
    uniform:&mut U,
    normal:&mut G
)
    where
        T:Rng,
        G:Distribution<f64>,
        U:Distribution<f64>
{
    new_nest.iter_mut()
        .zip(curr_nest.iter())
        .for_each(|(&mut new_val, curr_val)|{
            let (new_parameters, new_fn_val)=new_val;
            let (curr_parameters,_)=curr_val;
            new_parameters=curr_parameters.iter()
                .zip(ul.iter())
                .zip(best_parameters.iter())
                .map(|((curr_param, v), bp)|{
                    get_truncated_parameter(
                        v.lower, v.upper, 
                        get_levy_flight(
                            *curr_param, 
                            get_step_size(*curr_param, *bp, v.lower, v.upper), 
                            lambda, 
                            uniform.sample(rng), 
                            normal.sample(rng)
                        )
                    )
                }).collect();
            
            new_fn_val=obj_fn(&new_parameters);
            //(new_nest_parameters, new_nest_fn)
        });//.collect()
}

fn get_pa(
    p_min:f64,
    p_max:f64,
    index:usize,
    n:usize
)->f64{
    p_max-(p_max-p_min)*(index as f64)/(n as f64)
}

fn empty_nests<T, U>(
    new_nest:&mut Vec<(Vec<f64>, f64)>, //
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    ul:&Vec<Upper_lower>,
    p:f64,
    rng:&mut T,
    rand:&mut U
)
    where 
        T:Rng,
        U:Distribution<f64>
{
    let n=new_nest.len();
    let num_to_keep=((n as f64)*p) as usize;
    let start_num=n-num_to_keep;
    new_nest.iter_mut().enumerate().for_each(|(index, &mut new_val)|{
        //let (new_parameters, new_fn_val)=new_val;
        if index>=start_num {
            new_val=get_new_parameter_and_fn(ul, &obj_fn, rng, rand);
        }
    });
}

pub fn get_rng_seed(seed:i32)->StdRng{
    let mut wtr = vec![];
    wtr.write_i32::<LittleEndian>(seed).unwrap();

    let mut array = [0; 32];
    let bytes = &wtr[..array.len()]; // panics if not enough data
    array.copy_from_slice(bytes);


    SeedableRng::from_seed(array) 
}

pub fn get_rng_system_seed()->ThreadRng{
    thread_rng()
}

pub fn optimize<T>(
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    ul:&Vec<Upper_lower>,
    n:usize,
    total_mc:usize,
    tol:f64,
    rng_inst:impl Fn()->T
)->(Vec<f64>, f64)
    where T:Rng
{
    
    let lambda=1.5;//controls size of levy moves
    let p_min=0.05;//min percentage of nests to replace
    let p_max=0.5;//max percentage of nests to replace

    //randomness
    let mut rng=rng_inst();
    let mut normal=StandardNormal;
    let mut uniform=Uniform::new(0.0f64, 1.0);
    let mut curr_nest=get_new_nest(&ul, &obj_fn, n, &mut rng, &mut normal);
    sort_nest(&mut curr_nest);

    let mut new_nest=get_new_nest(&ul, &obj_fn, n, &mut rng, &mut normal);

    let mut done = false; // mut done: bool
    let mut index=0;
    while !done {
        let (curr_best_params, _)=curr_nest.first().unwrap();
        get_cuckoos(
            &mut new_nest, &curr_nest, 
            &curr_best_params, &ul, 
            &obj_fn, lambda, 
            &mut rng, &mut uniform, &mut normal
        );
        get_best_nest(&new_nest, &mut curr_nest);
        empty_nests(
            &mut new_nest, 
            &obj_fn, &ul,
            get_pa(p_min, p_max, index, total_mc),
            &mut rng,
            &mut normal
        );
        sort_nest(&mut new_nest);

        let (_, fn_min)=curr_nest.first().unwrap();
        index=index+1;
        done=index>=total_mc || fn_min<=&tol; 
    }
    *curr_nest.first().unwrap()
}


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_fn_optim() {
        let seed: &[_] = &[1, 2, 3, 4];
        let bounds:Upper_lower=Upper_lower{ lower:-4.0, upper:4.0};
        let ul=vec![bounds; 4];
        let (result, fn_val)=optimize(|&inputs|{
            inputs[0].powi(2)+inputs[1].powi(2)+inputs[2].powi(2)+inputs[3].powi(2)
        }, &ul, 25, 1000, 0.00000001, || get_rng_seed(seed));
        for res in result.iter(){
            assert_abs_diff_eq!(*res, 0.0, epsilon=0.00001);
        }
        
    }
}
