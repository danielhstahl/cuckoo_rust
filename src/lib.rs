extern crate rand;

use rand::prelude::*;
use rand::{thread_rng, Rng};
use rand::distributions::Normal;

struct upper_lower {
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

fn get_random_parameters(
    ul:&Vec<upper_lower>,
    uniform_rand_generator:impl Fn()->f64
)->Vec<f64>{
    ul.iter().map(|v|get_random_parameter(v.lower, v.upper, uniform_rand_generator())).collect()
}

fn get_new_paramater_and_fn(
    ul:&Vec<upper_lower>,
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    uniform_rand_generator:&impl Fn()->f64
)->(Vec<f64>, f64) {
    let parameters=get_random_parameters(ul, uniform_rand_generator);
    let fn_value_at_parameters=obj_fn(&parameters);
    (parameters, fn_value_at_parameters)
}

static step_increment:f64=0.01;
fn get_step_size(curr:f64, best:f64, lower:f64, upper:f64)->f64{
    step_increment*(upper-lower)*(curr-best)
}

fn get_new_nests(
    ul:&Vec<upper_lower>, 
    obj_fn:&impl Fn(&Vec<f64>)->f64,
    uniform_rand_generator:&impl Fn()->f64,
    n:usize
)->Vec<(Vec<f64>, f64)>{
    (0..n).map(|_|get_new_paramater_and_fn(ul, &obj_fn, &uniform_rand_generator)).collect()
}

/**void getCuckoos(
        Nest* newNest, const Nest& nest, 
        const BestParameter& bP,
        const ObjFun& objFun,
        const Array& ul, 
        const U& lambda, 
        const Unif& unif,
        const Norm& norm
    ){
        int n=nest.size(); //num nests
        int m=nest[0].first.size(); //num parameters
        Nest& nestRef= *newNest;
        for(int i=0; i<n;++i){
            for(int j=0; j<m; ++j){
                nestRef[i].first[j]=swarm_utils::getTruncatedParameter(
                    ul[j].lower, ul[j].upper, 
                    swarm_utils::getLevyFlight(
                        nest[i].first[j], 
                        getStepSize(nest[i].first[j], bP[j], ul[j].lower, ul[j].upper), 
                        lambda, unif(), norm()
                    )
                );
            }
            nestRef[i].second=objFun(nestRef[i].first);
        }
    }*/

fn get_cuckoos(
    new_nest:&Vec<(Vec<f64>, f64)>,
    curr_nest:&Vec<(Vec<f64>, f64)>,
    best_parameters:&Vec<f64>,
    obj_fn:impl Fn(&Vec<f64>)->f64,
    ul:&Vec<upper_lower>,
    lambda:f64,
    uniform_rand_generator:impl Fn()->f64,
    normal_rand_generator:impl Fn()->f64
)->Vec<(Vec<f64>, f64)>
{
    curr_nest.iter().map(|(parameters, _)|{
        let new_nest_parameters=parameters.iter().zip(ul.iter()).zip(best_parameters.iter()).map(|((param, v), bp)|{
            get_truncated_parameter(
                v.lower, v.upper, 
                get_levy_flight(
                    *param, 
                    get_step_size(*param, *bp, v.lower, v.upper), 
                    lambda, 
                    uniform_rand_generator(), 
                    normal_rand_generator()
                )
            )
        }).collect();
        let new_nest_fn=obj_fn(&new_nest_parameters);
        (new_nest_parameters, new_nest_fn)
    }).collect()
}
//    let normal = Normal::new(0.0, 1.0);
//let v = normal.sample(&mut rand::thread_rng());


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
